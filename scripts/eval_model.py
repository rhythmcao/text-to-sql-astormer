#coding=utf8
import sys, os, gc, torch, pickle, time
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.example import Example
from utils.batch import Batch
from torch.utils.data import DataLoader
from nsts.transition_system import GenerateTokenAction


def decode(model, dataset, output_path=None, batch_size=64, beam_size=5, n_best=5, decode_order='dfs+l2r', etype='all', device=None):
    model.eval()
    evaluator, decode_method = Example.evaluator, Example.decode_method
    eval_collate_fn = Batch.get_collate_fn(device=device, train=False)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False, collate_fn=eval_collate_fn)

    pred_sqls, time_elapse = [], 0
    with torch.no_grad():
        for cur_batch in data_loader:
            cur_time = time.time()
            hyps = model.parse(cur_batch, beam_size=beam_size, n_best=n_best, decode_order=decode_order)
            time_elapse += time.time() - cur_time
            sqls = evaluator.postprocess(hyps, cur_batch.examples, decode_method=decode_method, execution_guided=True)
            pred_sqls.extend(sqls)

    # print(f'[Evaluation]: total elapsed time for model.parse is {1000 * time_elapse * 1000 / len(dataset):.6f}ms/1000 samples .')
    torch.cuda.empty_cache()
    gc.collect()
    return evaluator.accuracy(pred_sqls, dataset, output_path, etype=etype)


def print_ast(model, dataset, output_path, beam_size=5, n_best=5, decode_order='dfs+random', device=None, **kwargs):
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
    torch.cuda.empty_cache()
    gc.collect()
    return count


def record_heatmap(model, dataset, output_path=None, decode_order='dfs+l2r', device=None):
    decode_method = Example.decode_method
    assert decode_method == 'ast', "We only analyze target side self-attention of ASTormer."
    model.eval()
    tranx, decode_method = Example.tranx, Example.decode_method
    eval_collate_fn = Batch.get_collate_fn(device=device, train=True, decode_order=decode_order)
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False, drop_last=False, collate_fn=eval_collate_fn)

    results, count = [], 0
    id2relation = tranx.ast_relation.id2relation
    with torch.no_grad():
        for cur_batch in data_loader:
            ex = cur_batch[0]
            tgt_len = len(ex.action)
            if len(ex.action) > 20 or any(action.action_type == GenerateTokenAction for action in ex.action): continue
            count += 1

            _, attention_weights = model(cur_batch, return_attention_weights=True) # use forward function instead of parse
            rel_mask = torch.tril(torch.ones((tgt_len, tgt_len), dtype=torch.bool))
            pad_idx = tranx.ast_relation.DECODER_RELATIONS.index('padding-padding')
            relation = cur_batch.decoder_relations[0].masked_fill(~ rel_mask, pad_idx).tolist()
            relation = [[id2relation[rid] for rid in rid_list] for rid_list in relation]
            records = {
                'db_id': ex.ex['db_id'], # database id~(string)
                'question': ex.ex['question'], # question~(string)
                'query': ex.query, # SQL query~(string)
                'ast': ex.ast, # object of class AbstractSyntaxTree, see nsts/asdl_ast.py
                'action': cur_batch.action_infos[0], # list of class ActionInfos, see nsts/transition_system.py
                'relation': relation, # tgt_len x tgt_len list, each entry is a relation name~(string), e.g., `0-0', `0-1', see DECODER_RELATIONS in nsts/relation_utils.py
                'weight': attention_weights[0][:, :, :len(ex.action), :len(ex.action)].numpy() # numpy float array, Layer x Head x tgt_len x tgt_len
            }
            results.append(records)

    pickle.dump(results, open(output_path, 'wb'))
    torch.cuda.empty_cache()
    gc.collect()
    return count
