#coding=utf8
import os

EXP_PATH = 'exp'


def hyperparam_path(args, create=True):
    if args.read_model_path and args.testing:
        return args.read_model_path
    exp_path = hyperparam_path_text2sql(args)
    if create and not os.path.exists(exp_path):
        os.makedirs(exp_path)
    return exp_path


def hyperparam_path_text2sql(args):
    task = 'task_%s__dataset_%s__encoder_%s__decoder_%s' % (args.task, args.dataset, args.encode_method, args.decode_method)

    # encoder params
    exp_path = args.plm
    exp_path += '__enc_%s_x_%s__hd_%s' % (args.encoder_hidden_size, args.encoder_num_layers, args.num_heads)

    # decoder params
    exp_path += '__order_%s' % (args.decode_order)
    exp_path += '__dec_%s_x_%s__cell_%s' % (args.decoder_hidden_size, args.decoder_num_layers, args.decoder_cell)

    # training params
    exp_path += '__bs_%s__dp_%s' % (args.batch_size, args.dropout)
    exp_path += '__lr_%s_ld_%s' % (args.lr, args.layerwise_decay)
    exp_path += '__l2_%s' % (args.l2)
    # exp_path += '__wp_%s' % (args.warmup_ratio)
    # exp_path += '__sd_%s' % (args.lr_schedule)
    exp_path += '__mi_%s' % (args.max_iter)
    exp_path += '__mn_%s' % (args.max_norm)
    exp_path += '__bm_%s' % (args.beam_size)
    exp_path += '__nb_%s' % (args.n_best)
    exp_path += '__seed_%s' % (args.seed)
    exp_path = os.path.join(EXP_PATH, task, exp_path)
    return exp_path