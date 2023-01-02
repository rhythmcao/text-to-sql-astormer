#coding=utf8
import numpy as np
import re, sys, os, logging, random, torch
from utils.hyperparams import hyperparam_path


def set_logger(exp_path, testing=False, rank=0, append_mode=False):
    logFormatter = logging.Formatter('%(asctime)s - %(message)s') #('%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger('mylogger')
    level = logging.DEBUG if rank == 0 else logging.ERROR
    logger.setLevel(level)
    if rank == 0:
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
    if deviceId < 0:
        device = torch.device("cpu")
    else:
        assert torch.cuda.device_count() >= deviceId + 1
        device = torch.device("cuda:%d" % (deviceId))
        torch.backends.cudnn.enabled = False
        # os.environ['CUDA_LAUNCH_BLOCKING'] = "1" # used when debug
        ## These two lines are used to ensure reproducibility with cudnn backend
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False
    return device


def get_master_node_address():
    """ Select the master node in our slurm environment, machines are named like gqxx-01-104 """
    try:
        nodelist = os.environ['SLURM_STEP_NODELIST']
    except:
        nodelist = os.environ['SLURM_JOB_NODELIST']
    nodelist = nodelist.strip().split(',')[0]
    text = re.split('[-\[\]]', nodelist)
    if ('' in text):
        text.remove('')
    return text[0] + '-' + text[1] + '-' + text[2]


def distributed_init(host, rank, local_rank, world_size, port=23456):
    host_full = 'tcp://' + host + ':' + str(port)
    try:
        torch.distributed.init_process_group("nccl", init_method=host_full,
                                              rank=rank, world_size=world_size)
    except:
        print(f"Host addr {host_full}")
        print(f"Process id {int(os.environ['SLURM_PROCID'])}")
        exit("Distributed training initialization failed")
    assert torch.distributed.is_initialized()
    return set_torch_device(local_rank)


def initialization_wrapper(args):
    set_random_seed(args.seed)
    if args.ddp:
        host_address = get_master_node_address()
        local_rank = int(os.environ['SLURM_LOCALID'])
        rank = int(os.environ['SLURM_PROCID'])
        world_size = int(os.environ['SLURM_NTASKS'])
        port = '2' + os.environ['SLURM_JOBID'][-4:]
        print('host address | local_rank | rank | world_size | port:', host_address, local_rank, rank, world_size, port)
        device = distributed_init(host_address, rank, local_rank, world_size, port='2' + os.environ['SLURM_JOBID'][-4:])
    else:
        local_rank, rank, world_size = 0, 0, 1
        device = set_torch_device(args.device)
    exp_path = hyperparam_path(args, create=(rank == 0))
    logger = set_logger(exp_path, args.testing, rank, args.load_optimizer)
    logger.info("Initialization finished ...")
    logger.info(f"Output path is {exp_path}")
    logger.info(f"Random seed is set to {args.seed}")
    logger.info(f"World size is {world_size}")
    return exp_path, logger, device, local_rank, rank, world_size