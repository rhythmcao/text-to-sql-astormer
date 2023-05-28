#coding=utf8
import sys, os, json, argparse, time, torch
from argparse import Namespace
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.initialization import set_torch_device
from utils.example import Example
from utils.batch import Batch
from torch.utils.data import DataLoader
from model.model_utils import Registrable
from model.model_constructor import *


parser = argparse.ArgumentParser()
parser.add_argument('--read_model_path', required=True, help='path to saved model path, at least contain param.json and model.bin')
parser.add_argument('--db_dir', default='data/database', help='path to db dir')
parser.add_argument('--table_path', default='data/tables.json', help='path to tables json file')
parser.add_argument('--dataset_path', default='data/dev.json', help='path to raw dataset json file')
parser.add_argument('--output_path', default='predicted_sql.txt', help='output predicted sql file')
parser.add_argument('--batch_size', default=20, type=int, help='batch size for evaluation')
parser.add_argument('--beam_size', default=5, type=int, help='beam search size')
parser.add_argument('--n_best', default=5, type=int, help='return n_best size')
parser.add_argument('--device', type=int, default=-1, help='-1 -> CPU ; GPU index o.w.')
args = parser.parse_args(sys.argv[1:])

# load hyper-params
params = json.load(open(os.path.join(args.read_model_path, 'params.json'), 'r'), object_hook=lambda d: Namespace(**d))
params.lazy_load = True
device = set_torch_device(args.device)

# load dataset and preprocess
Example.configuration(params.dataset, swv=params.swv, plm=params.plm, encode_method=params.encode_method, decode_method=params.decode_method,
    table_path=args.table_path, db_dir=args.db_dir)
dataset = Example.load_dataset(dataset_path=args.dataset_path)
eval_collate_fn = Batch.get_collate_fn(device=device, train=False)
dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, drop_last=False, collate_fn=eval_collate_fn)

# load model params
model = Registrable.by_name('text2sql')(params, Example.tranx).to(device)
check_point = torch.load(open(os.path.join(args.read_model_path, 'model.bin'), 'rb'), map_location=device)['model']
model.load_state_dict(check_point)

model.eval()
start_time = time.time()
print(f'Start evaluating with {params.decode_method} method ...')
pred_sqls, evaluator, decode_method = [], Example.evaluator, Example.decode_method
with torch.no_grad():
    for cur_batch in dataloader:
        hyps = model.parse(cur_batch, beam_size=args.beam_size, n_best=args.n_best, decode_order=params.decode_order)
        sqls = evaluator.postprocess(hyps, cur_batch.examples, decode_method=decode_method, execution_guided=True)
        pred_sqls.extend(sqls)

print('Start writing predicted SQLs to file %s' % (args.output_path))
with open(args.output_path, 'w', encoding='utf8') as of:
    prev_id = 0
    for s, ex in zip(pred_sqls, dataset):
        prefix = '\n' if ex.turn_id != prev_id else ''
        if Example.dataset == 'dusql': s = ex.id + '\t' + s
        of.write(prefix + s + '\n')
        prev_id = ex.turn_id
print('Evaluation costs %.4fs .' % (time.time() - start_time))