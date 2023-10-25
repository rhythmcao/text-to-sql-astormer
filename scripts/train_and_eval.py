#coding=utf8
import sys, os, time, json, gc, itertools, torch
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from argparse import Namespace
from contextlib import nullcontext
from utils.args import init_args
from utils.initialization import initialization_wrapper
from utils.example import Example
from utils.batch import Batch
from utils.optimization import set_optimizer
from model.model_utils import Registrable
from model.model_constructor import *
from torch.utils.data import DataLoader
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from scripts.eval_model import decode, print_ast, record_heatmap


# initialization params, output path, logger, random seed and torch.device
args = init_args(sys.argv[1:])


# setup class Example and load dataset
start_time = time.time()
if args.read_model_path: # load from checkpoints or testing mode
    params = json.load(open(os.path.join(args.read_model_path, 'params.json')), object_hook=lambda d: Namespace(**d))
    params.read_model_path, params.lazy_load = args.read_model_path, True
    params.load_optimizer, params.testing, params.device, params.ddp = args.load_optimizer, args.testing, args.device, args.ddp
    params.batch_size, params.grad_accumulate, params.test_batch_size = args.batch_size, args.grad_accumulate, args.test_batch_size
    params.beam_size, params.n_best = args.beam_size, args.n_best
    if not params.load_optimizer:
        params.max_iter, params.eval_after_iter = args.max_iter, args.eval_after_iter
        params.lr, params.l2, params.layerwise_decay = args.lr, args.l2, args.layerwise_decay
    args = params
exp_path, logger, device, local_rank, rank, world_size = initialization_wrapper(args)
is_master = (rank == 0)


# initialize model
Example.configuration(args.dataset, swv=args.swv, plm=args.plm, encode_method=args.encode_method, decode_method=args.decode_method)
model = Registrable.by_name('text2sql')(args, Example.tranx).to(device)
if args.read_model_path:
    check_point = torch.load(open(os.path.join(args.read_model_path, 'model.bin'), 'rb'), map_location=device)
    model.load_state_dict(check_point['model'])
    logger.info(f"Load saved model from path: {args.read_model_path:s}")
else: json.dump(vars(args), open(os.path.join(exp_path, 'params.json'), 'w'), indent=4)
if args.ddp: # add DDP wrapper for model
    model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
base_model = model.module if args.ddp else model


# read dataset
if not args.testing:
    train_dataset = Example.load_dataset('train')
    logger.info(f"Dataset size: train -> {len(train_dataset):d} ;")
dev_dataset = Example.load_dataset('dev')
logger.info(f"Dataset size: dev -> {len(dev_dataset):d} ;")
logger.info(f"Load dataset finished, cost {time.time() - start_time:.4f}s ...")


if not args.testing:
    assert args.batch_size % (world_size * args.grad_accumulate) == 0
    batch_size = args.batch_size // (world_size * args.grad_accumulate)

    # set training dataloader
    train_collate_fn = Batch.get_collate_fn(device=device, train=True, decode_order=args.decode_order) #, sample_size=4
    if args.ddp:
        train_sampler = DistributedSampler(train_dataset)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, shuffle=False, collate_fn=train_collate_fn)
    else: train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False, collate_fn=train_collate_fn)

    # set optimizer and scheduler
    eval_per_iter, loss_per_iter = 2000, 1000
    num_training_steps = args.max_iter * loss_per_iter
    num_warmup_steps = int(num_training_steps * args.warmup_ratio)
    optimizer, scheduler = set_optimizer(base_model, args, num_training_steps, module=base_model.encoder.input_layer, module_name='plm')

    iteration, start_epoch, best_result = 0, 0, { 'dev_em_acc': 0. , 'dev_ex_acc': .0}
    logger.info(f'Total training steps: {num_training_steps:d};\tWarmup steps: {num_warmup_steps:d}')
    if args.read_model_path and args.load_optimizer:
        optimizer.load_state_dict(check_point['optim'])
        scheduler.load_state_dict(check_point['scheduler'])
        iteration, start_epoch = check_point['iter'], check_point['epoch'] + 1
        best_result = check_point['result']
        logger.info(f'Previous Best Dev EM/EX Acc is {best_result["dev_em_acc"]:.4f}/{best_result["dev_ex_acc"]:.4f} .')
    logger.info(f'Start training from epoch {start_epoch:d} iteration({loss_per_iter:d}) {iteration // loss_per_iter:d} ......')

    model.train()
    terminate, count, start_time, loss_tracker = False, 0, time.time(), 0.
    for i in itertools.count(start_epoch, 1):
        if args.ddp: train_loader.sampler.set_epoch(i)
        for j, current_batch in enumerate(train_loader):
            count += 1
            update_flag = (count == args.grad_accumulate)
            cntx = model.no_sync() if args.ddp and not update_flag else nullcontext()
            with cntx:
                loss = model(current_batch)
                (world_size * loss).backward()
                loss_tracker += loss.item()
                if update_flag:
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    count = 0
                    iteration += 1

                    if iteration % loss_per_iter == 0:
                        logger.info(f'Training iteration({loss_per_iter:d}): {iteration // loss_per_iter:d}\tTime: {time.time() - start_time:.2f}s\tLoss: {loss_tracker:.4f}')
                        start_time, loss_tracker = time.time(), 0
                        torch.cuda.empty_cache()
                        gc.collect()

                    if iteration % eval_per_iter == 0 and iteration < args.eval_after_iter * loss_per_iter and is_master:
                        torch.save({
                            'epoch': i, 'iter': iteration,
                            'model': base_model.state_dict(),
                            'optim': optimizer.state_dict(),
                            'scheduler': scheduler.state_dict(),
                            'result': best_result
                        }, open(os.path.join(exp_path, 'model.bin'), 'wb'))
                        start_time = time.time()
                    elif iteration % eval_per_iter == 0 and iteration >= args.eval_after_iter * loss_per_iter and is_master:
                        start_time = time.time()
                        output_path = os.path.join(exp_path, 'dev.iter%s' % (str(iteration // loss_per_iter)))
                        dev_em_acc, dev_ex_acc = decode(base_model, dev_dataset, output_path, batch_size=args.test_batch_size,
                            beam_size=args.beam_size, n_best=args.n_best, decode_order=args.decode_order, device=device)
                        logger.info(f"Evaluation iteration({loss_per_iter:d}): {iteration // loss_per_iter:d}\tTime: {time.time() - start_time:.2f}s\tDev EM/EX acc: {dev_em_acc:.4f}/{dev_ex_acc:.4f}")
                        if dev_em_acc + dev_ex_acc >= best_result['dev_em_acc'] + best_result['dev_ex_acc']:
                            best_result['dev_em_acc'], best_result['dev_ex_acc'] = dev_em_acc, dev_ex_acc
                            torch.save({
                                'epoch': i, 'iter': iteration,
                                'model': base_model.state_dict(),
                                'optim': optimizer.state_dict(),
                                'scheduler': scheduler.state_dict(),
                                'result': best_result
                            }, open(os.path.join(exp_path, 'model.bin'), 'wb'))
                            logger.info(f"NEW BEST MODEL in iteration({loss_per_iter:d}): {iteration // loss_per_iter:d}\tDev EM/EX acc: {dev_em_acc:.4f}/{dev_ex_acc:.4f}")
                        start_time = time.time()
                        model.train()

                    if iteration >= num_training_steps:
                        terminate = True
                        break
        if terminate: break

    if is_master:
        check_point = torch.load(open(os.path.join(exp_path, 'model.bin'), 'rb'), map_location=device)
        del check_point['optim'], check_point['scheduler']
        base_model.load_state_dict(check_point['model'])
        logger.info(f"\nReload saved model in iteration({loss_per_iter:d}) {check_point['iter'] // loss_per_iter:d} from path: {exp_path:s}")


# eval model on test-suite database
if is_master:
    Example.use_database_testsuite()

    logger.info("Start evaluating dev dataset on testsuite database ......")
    start_time = time.time()
    dev_em_acc, dev_ex_acc = decode(base_model, dev_dataset, os.path.join(exp_path, 'dev.eval'), batch_size=args.test_batch_size,
        beam_size=args.beam_size, n_best=args.n_best, decode_order=args.decode_order, device=device)
    logger.info(f"EVALUATION costs {time.time() - start_time:.2f}s ; Dev EM/EXT acc: {dev_em_acc:.4f}/{dev_ex_acc:.4f} ;")
    check_point['result']['dev_em_acc'], check_point['result']['dev_ex_acc'] = dev_em_acc, dev_ex_acc

    logger.info("Start evaluating dev_syn dataset on testsuite database ......")
    dev_dataset = Example.load_dataset('dev_syn')
    start_time = time.time()
    dev_em_acc, dev_ex_acc = decode(base_model, dev_dataset, os.path.join(exp_path, 'dev_syn.eval'), batch_size=args.test_batch_size,
        beam_size=args.beam_size, n_best=args.n_best, decode_order=args.decode_order, device=device)
    logger.info(f"EVALUATION costs {time.time() - start_time:.2f}s ; Dev EM/EXT acc: {dev_em_acc:.4f}/{dev_ex_acc:.4f} ;")
    check_point['result']['dev_syn_em_acc'], check_point['result']['dev_syn_ex_acc'] = dev_em_acc, dev_ex_acc

    logger.info("Start evaluating dev_dk dataset on testsuite database ......")
    dev_dataset = Example.load_dataset('dev_dk')
    start_time = time.time()
    dev_em_acc, dev_ex_acc = decode(base_model, dev_dataset, os.path.join(exp_path, 'dev_dk.eval'), batch_size=args.test_batch_size,
        beam_size=args.beam_size, n_best=args.n_best, decode_order=args.decode_order, device=device)
    logger.info(f"EVALUATION costs {time.time() - start_time:.2f}s ; Dev EM/EXT acc: {dev_em_acc:.4f}/{dev_ex_acc:.4f} ;")
    check_point['result']['dev_dk_em_acc'], check_point['result']['dev_dk_ex_acc'] = dev_em_acc, dev_ex_acc

    logger.info("Start evaluating dev_realistic dataset on testsuite database ......")
    dev_dataset = Example.load_dataset('dev_realistic')
    start_time = time.time()
    dev_em_acc, dev_ex_acc = decode(base_model, dev_dataset, os.path.join(exp_path, 'dev_realistic.eval'), batch_size=args.test_batch_size,
        beam_size=args.beam_size, n_best=args.n_best, decode_order=args.decode_order, device=device)
    logger.info(f"EVALUATION costs {time.time() - start_time:.2f}s ; Dev EM/EXT acc: {dev_em_acc:.4f}/{dev_ex_acc:.4f} ;")
    check_point['result']['devi_realistic_em_acc'], check_point['result']['dev_realistic_ex_acc'] = dev_em_acc, dev_ex_acc

    # logger.info('Start evaluating and printing ASTs on the dev dataset ......')
    # start_time = time.time()
    # count = print_ast(base_model, dev_dataset, os.path.join(exp_path, 'dev.ast'), beam_size=args.beam_size, n_best=args.n_best, decode_order=args.decode_order, device=device)
    # logger.info(f"EVALUATION costs {time.time() - start_time:.2f}s ; Print {count:d} ASTs among {len(dev_dataset):d} samples ;")

    # print('Start recording attention heatmaps on the dev dataset ...... ')
    # start_time = time.time()
    # count = record_heatmap(base_model, dev_dataset, os.path.join(exp_path, 'dev.heatmap'), decode_order=args.decode_order, device=device)
    # logger.info(f"EVALUATION costs {time.time() - start_time:.2f}s ; Record {count:d} heatmaps among {len(dev_dataset):d} samples ;")

    torch.save(check_point, open(os.path.join(exp_path, 'model.bin'), 'wb'))

if args.ddp:
    dist.destroy_process_group()
