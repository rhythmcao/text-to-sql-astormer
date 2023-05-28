#coding=utf-8
import argparse
import sys


def init_args(params=sys.argv[1:]):
    arg_parser = argparse.ArgumentParser()
    arg_parser = add_argument_base(arg_parser)
    arg_parser = add_argument_encoder(arg_parser)
    arg_parser = add_argument_decoder(arg_parser)
    args = arg_parser.parse_args(params)
    if args.decode_method == 'ast': args.no_select_schema = False
    if args.swv: assert args.encode_method != 'none', 'GNN should be adopted for swv!'
    return args


def add_argument_base(arg_parser):
    #### General configuration ####
    arg_parser.add_argument('--task', default='text2sql', help='Task name')
    arg_parser.add_argument('--dataset', type=str, default='spider', choices=['spider', 'sparc', 'cosql', 'dusql', 'chase'], help='Dataset name')
    arg_parser.add_argument('--seed', default=999, type=int, help='Random seed')
    arg_parser.add_argument('--device', type=int, default=0, help='Use which device: -1 -> cpu ; the index of gpu o.w.')
    arg_parser.add_argument('--ddp', action='store_true', help='use distributed data parallel training')
    arg_parser.add_argument('--testing', action='store_true', help='training or evaluation mode')
    arg_parser.add_argument('--read_model_path', type=str, help='read pretrained model path')
    arg_parser.add_argument('--local_rank', type=int, help='local rank for DDP training')
    #### Training Hyperparams ####
    arg_parser.add_argument('--dropout', type=float, default=0.2, help='feature dropout rate')
    arg_parser.add_argument('--batch_size', default=20, type=int, help='Batch size')
    arg_parser.add_argument('--test_batch_size', default=64, type=int, help='Test batch size')
    arg_parser.add_argument('--grad_accumulate', default=1, type=int, help='accumulate grad and update once every x steps')
    arg_parser.add_argument('--lr', type=float, default=5e-4, help='learning rate')
    arg_parser.add_argument('--layerwise_decay', type=float, default=1.0, help='layerwise decay rate for lr, used for PLM')
    arg_parser.add_argument('--l2', type=float, default=1e-4, help='weight decay coefficient')
    arg_parser.add_argument('--warmup_ratio', type=float, default=0.1, help='warmup steps proportion')
    arg_parser.add_argument('--lr_schedule', default='linear', choices=['constant', 'linear', 'ratsql', 'cosine'], help='lr scheduler')
    arg_parser.add_argument('--eval_after_iter', default=40, type=int, help='Start to evaluate after x epoch')
    arg_parser.add_argument('--load_optimizer', action='store_true', default=False, help='Whether to load optimizer state')
    arg_parser.add_argument('--max_iter', type=int, default=100, help='Terminate after maximum iter * 1000.')
    arg_parser.add_argument('--max_norm', default=5., type=float, help='Gradient clip norm size.')
    return arg_parser


def add_argument_encoder(arg_parser):
    # Encoder Hyperparams
    arg_parser.add_argument('--encode_method', choices=['irnet', 'rgatsql', 'none'], default='rgatsql', help='which graph encoder to use')
    arg_parser.add_argument('--swv', action='store_true', default=False, help='use static word vectors instead of PLM')
    arg_parser.add_argument('--plm', type=str, help='pretrained model name in Huggingface')
    arg_parser.add_argument('--encoder_num_layers', default=8, type=int, help='num of GNN layers in encoder')
    arg_parser.add_argument('--encoder_hidden_size', default=256, type=int, help='size of GNN layers hidden states')
    arg_parser.add_argument('--num_heads', default=8, type=int, help='num of heads in multihead attn')
    return arg_parser


def add_argument_decoder(arg_parser):
    # Decoder Hyperparams
    arg_parser.add_argument('--decode_method', choices=['seq', 'ast'], default='ast', help='method for decoding')
    arg_parser.add_argument('--decode_order', choices=['dfs+l2r', 'bfs+l2r', 'dfs+random', 'bfs+random', 'random'], default='dfs+l2r', help='Decoding order for AST generation')
    arg_parser.add_argument('--decoder_cell', choices=['transformer', 'onlstm', 'lstm'], default='transformer', help='Transformer cell, ONLSTM or traditional LSTM')
    arg_parser.add_argument('--decoder_num_layers', type=int, default=1, help='num_layers of decoder')
    arg_parser.add_argument('--decoder_hidden_size', default=512, type=int, help='Size of LSTM hidden states')
    arg_parser.add_argument('--no_select_schema', action='store_true', help='Whether directly copy DB schema or generating schema tokens')
    arg_parser.add_argument('--no_copy_mechanism', action='store_true', help='Whether use copy mechanism for generating value tokens')
    arg_parser.add_argument('--beam_size', default=5, type=int, help='Beam size for beam search')
    arg_parser.add_argument('--n_best', default=5, type=int, help='Top-k returns for beam search')
    return arg_parser