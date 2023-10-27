#coding=utf8
import numpy as np
import re, sys, os, logging, random, torch
from utils.hyperparams import hyperparam_path


def set_logger(exp_path, testing=False, is_master=True, append_mode=False):
    logFormatter = logging.Formatter('%(asctime)s - %(message)s') #('%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger('mylogger')
    level = logging.DEBUG #if is_master else logging.ERROR
    logger.setLevel(level)
    if is_master:
        mode = 'a' if append_mode else 'w'
        fileHandler = logging.FileHandler('%s/log_%s.txt' % (exp_path, ('test' if testing else 'train')), mode=mode)
        fileHandler.setFormatter(logFormatter)
        logger.addHandler(fileHandler)
    consoleHandler = logging.StreamHandler(sys.stdout)
    consoleHandler.setFormatter(logFormatter)
    logger.addHandler(consoleHandler)
    return logger


def set_random_seed(random_seed=999):
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(random_seed)


def set_torch_device(deviceId):
    deviceId = deviceId if torch.cuda.is_available() else -1
    if deviceId < 0:
        device = torch.device("cpu")
        print('[WARNING]: Use CPU , not GPU !')
    else:
        assert torch.cuda.device_count() >= deviceId + 1
        device = torch.device("cuda:%d" % (deviceId))
        torch.backends.cudnn.enabled = False
        # os.environ['CUDA_LAUNCH_BLOCKING'] = "1" # used when debug
        ## These two lines are used to ensure reproducibility with cudnn backend
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False
    return device


def initialization_wrapper(args):
    set_random_seed(args.seed)
    if args.ddp:
        try:
            torch.distributed.init_process_group("nccl")
            assert torch.distributed.is_initialized()
            print('Initialization succeed !')
        except:
            exit("[ERROR]: Distributed training initialization failed !")
        rank, world_size = torch.distributed.get_rank(), torch.distributed.get_world_size()
        device = set_torch_device(args.local_rank)
    else:
        rank, world_size = 0, 1
        device = set_torch_device(args.local_rank)
    is_master = (rank == 0)
    exp_path = hyperparam_path(args, create=is_master)
    logger = set_logger(exp_path, args.testing, is_master, args.load_optimizer)
    logger.info("Initialization finished ...")
    logger.info(f"Output path: {exp_path}")
    logger.info(f"Random seed: {args.seed}")
    logger.info(f"Local rank: {args.local_rank} ; Global rank: {rank} ; World size: {world_size}")
    return exp_path, logger, device, is_master, world_size
