#coding=utf8
import sys, os, gc, torch
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.example import Example
from utils.batch import Batch
from torch.utils.data import DataLoader


def decode(model, dataset, output_path=None, batch_size=64, beam_size=5, n_best=5, decode_order='dfs+l2r', etype='all', device=None):
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


def print_ast(model, dataset, output_path, beam_size=5, n_best=5, decode_order='random', device=None, **kwargs):
    evaluator, decode_method = Example.evaluator, Example.decode_method
    assert 'random' in decode_order and decode_method == 'ast', "There is no need to analyze the canonical order or token-based method."
    model.eval()
    eval_collate_fn = Batch.get_collate_fn(device=device, train=False)
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False, drop_last=False, collate_fn=eval_collate_fn)

    count, tokenizer = 0, Example.tokenizer
    with torch.no_grad(), open(output_path, 'w') as of:
        for cur_batch in data_loader:
            if cur_batch.examples[0].ast.size > 30: continue

            hyps = model.parse(cur_batch, beam_size=beam_size, n_best=n_best, decode_order=decode_order)
            pred_sqls = evaluator.postprocess(hyps, cur_batch.examples, decode_method=decode_method, execution_guided=False)
            em, ex = evaluator.accuracy(pred_sqls, cur_batch.examples, output_path=None, etype='all')
            if int(em) == 1 and int(ex) == 1: # only print corrent predictions
                ex, tree = cur_batch.examples[0], hyps[0][0].tree
                if tree is None: continue
                count += 1
                of.write(f'DB: {ex.db["db_id"]}\n')
                of.write(f'Question: {ex.ex["question"]}\n')
                of.write(f'Gold SQL: {ex.query}\n')
                of.write(f'Pred SQL: {pred_sqls[0]}\n')
                of.write(f'AST[size={tree.size:d}]\n')
                of.write(tree.to_string(tables=ex.db['table_names_original'], columns=ex.db['column_names_original'], tokenizer=tokenizer))
                of.write('\n\n')
    return count


def print_heatmap(model, dataset, output_path=None, batch_size=64, beam_size=5, n_best=5, decode_order='dfs+l2r', device=None):
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