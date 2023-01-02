#coding=utf8
import sys, os, gc, torch
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.example import Example
from utils.batch import Batch
from torch.utils.data import DataLoader


def decode(model, dataset, output_path=None, batch_size=64, beam_size=5, n_best=1, decode_order='dfs+l2r', etype='all', device=None):
    model.eval()
    evaluator, decode_method = Example.evaluator, Example.decode_method
    eval_collate_fn = Batch.get_collate_fn(device=device, train=False)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False, collate_fn=eval_collate_fn)

    pred_sqls = []
    with torch.no_grad():
        for cur_batch in data_loader:
            hyps = model.parse(cur_batch, beam_size=beam_size, n_best=n_best, decode_order=decode_order)
            sqls = evaluator.postprocess(hyps, cur_batch.examples, decode_method=decode_method, execution_guided=True)
            pred_sqls.extend(sqls)

    torch.cuda.empty_cache()
    gc.collect()
    return evaluator.accuracy(pred_sqls, dataset, output_path, etype=etype)