#coding=utf-8
import sys, os, json, torch
import numpy as np
from collections import deque
from dataclasses import dataclass
from typing import Union, List, Tuple
from transformers import AutoTokenizer
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nsts.asdl_ast import AbstractSyntaxTree


CONFIG_PATHS = {
    'grammar': 'nsts/sql_grammar.txt',
    'plm_dir': 'pretrained_models',
    'spider': {
        'train': 'data/spider/train_spider.json',
        'dev': 'data/spider/dev.json',
        'tables': 'data/spider/tables.json',
        'db_dir': 'data/spider/database',
        'testsuite': 'data/spider/database-testsuite'
    },
    'sparc': {
        'train': 'data/sparc/train.json',
        'dev': 'data/sparc/dev.json',
        'tables': 'data/sparc/tables.json',
        'db_dir': 'data/sparc/database',
        'testsuite': 'data/spider/database-testsuite'
    },
    'cosql': {
        'train': 'data/cosql/sql_state_tracking/cosql_train.json',
        'dev': 'data/cosql/sql_state_tracking/cosql_dev.json',
        'tables': 'data/cosql/tables.json',
        'db_dir': 'data/cosql/database',
        'testsuite': 'data/spider/database-testsuite'
    }
}


class Action(object):

    def __init__(self, token: int):
        self.token = token


    def __repr__(self):
        return "%s[id=%s]" % (self.__class__.__name__, self.token)


class ApplyRuleAction(Action):

    @property
    def production_id(self):
        return self.token


class GenerateTokenAction(Action):

    EOV_ID = -1

    def is_stop_signal(self):
        return self.token == GenerateTokenAction.EOV_ID


class SelectColumnAction(Action):

    @property
    def column_id(self):
        return self.token


class SelectTableAction(Action):

    @property
    def table_id(self):
        return self.token


@dataclass
class ActionInfo(object):
    """ Sufficient statistics to record the information of one decoding action.
    Notice that, timestep records the canonical traverse order following depth-first-search and left-to-right.
    The real traverse order can be altered by permutating the action sequence and relation matrix as long as the top-down traversing criterion is obeyed.
    """
    timestep: int
    depth: int
    prod_id: int
    field_id: int
    parent_timestep: int
    action_type: type
    action_id: int


class TransitionSystem(object):

    primitive_type_to_action ={
        'tab_id': SelectTableAction,
        'col_id': SelectColumnAction,
        'val_id': GenerateTokenAction
    }

    def __init__(self, dataset: str, tokenizer: str = None, db_dir: str = None):
        super(TransitionSystem, self).__init__()
        from nsts.asdl import ASDLGrammar
        self.grammar = ASDLGrammar.from_filepath(CONFIG_PATHS['grammar'])
        from nsts.relation_utils import ASTRelation
        self.ast_relation = ASTRelation()

        tokenizer_path = os.path.join(CONFIG_PATHS['plm_dir'], tokenizer) if tokenizer is not None else os.path.join(CONFIG_PATHS['plm_dir'], 'grappa_large_jnt')
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, add_prefix_space=True)
        eov_token = self.tokenizer.eos_token if hasattr(self.tokenizer, 'eos_token') and self.tokenizer.eos_token is not None else self.tokenizer.sep_token
        GenerateTokenAction.EOV_ID = self.tokenizer.convert_tokens_to_ids(eov_token)

        from nsts.value_processor import ValueProcessor
        self.db_dir = CONFIG_PATHS[dataset]['db_dir'] if db_dir is None else db_dir
        self.value_processor = ValueProcessor(self.tokenizer, self.db_dir, eov_token=eov_token)
        
        from nsts.parse_json_to_ast import ASTParser
        self.ast_parser = ASTParser(self.grammar, self.value_processor)
        from nsts.parse_sql_to_json import JsonParser
        self.json_parser = JsonParser()

        from nsts.unparse_sql_from_ast import ASTUnParser
        self.ast_unparser = ASTUnParser(self.grammar, self.value_processor)
        from nsts.unparse_sql_from_json import JsonUnparser
        self.json_unparser = JsonUnparser()


    def change_database(self, db_dir):
        self.db_dir = db_dir
        self.value_processor.db_dir = db_dir


    def get_action_info_and_relation_from_ast(self, asdl_ast: AbstractSyntaxTree, relations: List[List[int]] = [], timestep: int = 0, depth: int = 0) -> Tuple[List[Union[List[ActionInfo], ActionInfo]], List[List[int]], int]:
        """ Given a complete SQL AST, extract the target output actions and relation matrix. Notice that, we maintain the structure with List or Tuple for output ActionInfo,
        to allow permutation-invariant top-down topological traversing.
        """
        prod2id, field2id = self.grammar.prod2id, self.grammar.field2id
        if timestep == 0: # root node
            parent_pid, parent_fid, parent_timestep = len(prod2id), len(field2id), -1
            relations, fill_relations = [[self.ast_relation.relation2id['0-0']]], True
        else:
            parent_pid, parent_fid, parent_timestep = prod2id[asdl_ast.parent_field.parent_node.production], field2id[asdl_ast.parent_field.field], asdl_ast.parent_field.parent_node.created_time
            relations.append(self.ast_relation.add_relation_for_child(relations, parent_timestep))
            fill_relations = False
        asdl_ast.created_time, pid = timestep, self.grammar.prod2id[asdl_ast.production]
        outputs = [ActionInfo(timestep, depth, parent_pid, parent_fid, parent_timestep, ApplyRuleAction, pid)]
        timestep += 1 # each time adding ActionInfo, add 1 to timestep

        depth += 1
        for field in asdl_ast.fields:
            realized_fields, fid = asdl_ast[field], field2id[field]
            children_actions = []

            for realized_field in realized_fields:
                if self.grammar.is_composite_type(realized_field.type): # nested subtree
                    current_actions, relations, timestep = self.get_action_info_and_relation_from_ast(realized_field.value, relations, timestep, depth)
                    children_actions.append(current_actions)
                else: # primitive types
                    action_type = TransitionSystem.primitive_type_to_action[realized_field.type.name]
                    if type(realized_field.value) in [list, tuple]: # GenerateTokenAction
                        action_list, token_start_timestep = [], timestep
                        for val in realized_field.value:
                            action_list.append(ActionInfo(timestep, depth, pid, fid, asdl_ast.created_time, action_type, int(val)))
                            relations.append(self.ast_relation.add_relation_for_child(relations, asdl_ast.created_time, token_start_timestep))
                            timestep += 1
                        children_actions.append(tuple(action_list)) # order sensitive for string tokens
                    else:
                        val = int(realized_field.value)
                        children_actions.append([ActionInfo(timestep, depth, pid, fid, asdl_ast.created_time, action_type, val)])
                        relations.append(self.ast_relation.add_relation_for_child(relations, asdl_ast.created_time))
                        timestep += 1
            # order sensitive for order by clause
            if 'OrderBy' in asdl_ast.production.constructor.name: children_actions = tuple(children_actions)
            outputs.append(children_actions)

        if fill_relations:
            relations = self.ast_relation.complete_triangular_relation_matrix(relations)

        return outputs, relations, timestep


    def flatten_action_info(self, action_info: List[Union[ActionInfo, List[ActionInfo]]], order: str = 'dfs+l2r') -> List[ActionInfo]:
        if order == 'dfs+l2r': # depth-first-search + left-to-right
            dfs_flatten = lambda x: [y for l in x for y in dfs_flatten(l)] if type(x) in [list, tuple] else [x]
            return dfs_flatten(action_info)
        elif order == 'bfs+l2r': # breadth first search + left-to-right
            def bfs_flatten(action_struct):
                actions, queue = [], deque([action_struct])
                while len(queue) > 0:
                    item = queue.popleft()
                    if isinstance(item, ActionInfo): actions.append(item)
                    elif item[0].action_type != ApplyRuleAction: actions.extend(item)
                    else:
                        actions.append(item[0])
                        for fields in item[1:]: queue.extend(fields)
                return actions

            return bfs_flatten(action_info)
        elif order == 'dfs+random': # depth-first-search + random horizontal order
            actions = []

            def dfs_random_flatten(action_struct):
                if type(action_struct) == tuple:
                    for item in action_struct: # keep the original order
                        dfs_random_flatten(item)
                elif type(action_struct) == list:
                    if isinstance(action_struct[0], ActionInfo): # AST node or primitive type
                        actions.append(action_struct[0])
                        action_struct = action_struct[1:]
                        indexes = np.arange(len(action_struct))
                        if len(indexes) > 1: np.random.shuffle(indexes)
                        for idx in indexes:
                            dfs_random_flatten(action_struct[idx])
                    else: # realized fields of the same Field
                        indexes = np.arange(len(action_struct))
                        if len(indexes) > 1: np.random.shuffle(indexes)
                        for idx in indexes:
                            dfs_random_flatten(action_struct[idx])
                else: actions.append(action_struct)

            dfs_random_flatten(action_info)
            return actions
        elif order == 'bfs+random': # breadth-first-earch + random horizontal order

            def bfs_random_flatten(action_struct):
                actions, queue = [], deque([action_struct])
                while len(queue) > 0:
                    item = queue.popleft()
                    if isinstance(item, ActionInfo): actions.append(item)
                    else: # iterable
                        if item[0].action_type != ApplyRuleAction:
                            actions.extend(item)
                        else:
                            actions.append(item[0])
                            item = item[1:]
                            indexes = np.arange(len(item))
                            if len(item) > 1: np.random.shuffle(indexes) # shuffle the order of different Fields
                            for idx in indexes:
                                fields = item[idx]
                                if len(fields) > 1 and type(fields) != tuple: # shuffle the order among the same Field
                                    field_indexes = np.arange(len(fields))
                                    np.random.shuffle(field_indexes)
                                    fields = [fields[jdx] for jdx in field_indexes]
                                queue.extend(fields) # append each child to the rightmost of the queue
                return actions

            return bfs_random_flatten(action_info)
        elif order == 'random': # randomly AST traverse as long as top-down topological order is obeyed
            
            def random_flatten(action_struct):
                actions, queue = [], [action_struct]
                while len(queue) > 0:
                    item = queue.pop(np.random.randint(len(queue))) # randomly sample one unexpanded nodes
                    if isinstance(item, ActionInfo): actions.append(item)
                    elif item[0].action_type != ApplyRuleAction: actions.extend(item)
                    else:
                        actions.append(item[0])
                        for fields in item[1:]: queue.extend(fields)
                return actions

            return random_flatten(action_info)
        else: raise ValueError(f'[ERROR]: traverse order not recognized {order:s} !')


    def get_outputs_from_ast(self, asdl_ast: AbstractSyntaxTree = None, action_infos: List[Union[ActionInfo, List[ActionInfo]]] = [], relations: List[List[int]] = [], order: str = 'dfs+l2r') -> Tuple[List[ActionInfo], torch.LongTensor]:
        """ Wrapper for func get_action_info_and_relation_from_ast, flatten the sequence of action_infos and permutate the row/columns for relations accordingly.
        """
        if asdl_ast:
            action_infos, relations, _ = self.get_action_info_and_relation_from_ast(asdl_ast)
        action_infos = self.flatten_action_info(action_infos, order=order)
        if len(action_infos) > 0:
            index = [action.timestep for action in action_infos]
            relations = relations[index][:, index]
        return action_infos, relations


    def unparse_sql_from_ast(self, asdl_ast, table, entry=None, **kwargs):
        """ table is used to retrieve column and table names by column_id and table_id;
        entry is used to provide extra information of the current sample, e.g., input question
        """
        return self.ast_unparser.unparse(asdl_ast, table, entry, **kwargs)


    def parse_sql_to_ast(self, sql: Union[str, dict], table: dict = None):
        """ if table is provided, parse the raw sql query~(str) into json dict
        """
        if type(sql) == str: # raw SQL query
            sql = self.json_parser.parse(sql, table)
        return self.ast_parser.parse(sql)


    def field_to_action(self, field):
        if field is not None: # typed constraint for grammar rules
            return TransitionSystem.primitive_type_to_action.get(field.type.name, ApplyRuleAction)
        else: return ApplyRuleAction


    def parse_sql_to_seq(self, sql: Union[str, dict], table: dict = None, select_schema: bool = True) -> List[int]:
        """ Convert the raw SQL into token/schema id sequence.
        If copy is True, schema is copied from the schema items instead of generated token-by-token~(the idea of constrained decoding),
        the corresponding schema index is schema_id + vocab_size.
        """
        do_lower_case = self.tokenizer.do_lower_case if hasattr(self.tokenizer, 'do_lower_case') else False
        assert not do_lower_case, 'The tokenizer of sequence decoder should be case-sensitive, o.w. unable to recover the upper/lower case information ...'

        # normalize the raw SQL query: parse to json and back to sql query to re-use the SQL-to-json parser
        if type(sql) == str:
            sql = self.json_parser.parse(sql, table)
        query = self.json_unparser.unparse(sql, table)
        if not select_schema: return [self.tokenizer.cls_token_id] + self.tokenzier.convert_tokens_to_ids(self.tokenizer.tokenize(query)) + [self.tokenizer.sep_token_id]

        tokens = query.split(' ')
        vocab_size = self.tokenizer.vocab_size
        table_names = table['table_names_original']
        column_names = list(map(lambda x: x[1] if x[0] == -1 else table_names[x[0]] + '.' + x[1], table['column_names_original']))
        masked_tokens, replacements = [], []
        mask_token, mask_token_id = self.tokenizer.pad_token, self.tokenizer.pad_token_id
        for tok in tokens:
            if tok in table_names:
                masked_tokens.append(mask_token)
                replacements.append(table_names.index(tok) + vocab_size)
            elif tok in column_names:
                masked_tokens.append(mask_token)
                replacements.append(column_names.index(tok) + vocab_size + len(table_names))
            else: masked_tokens.append(tok)
        masked_query = ' '.join(masked_tokens).replace(' ' + mask_token, mask_token)
        masked_token_ids = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(masked_query))
        replace_iter = iter(replacements)
        token_ids = list(map(lambda x: next(replace_iter) if x == mask_token_id else x, masked_token_ids))
        return [self.tokenizer.cls_token_id] + token_ids + [self.tokenizer.sep_token_id]


    def unparse_sql_from_seq(self, token_ids: List[int], db: dict, entry: dict = None) -> str:
        """ Translate the token_ids into the executable SQL query, take care of schema items if they are copied from the schema memory
        a.k.a., select_schema=True, instead of generated from the word vocabulary.
        """
        vocab_size = self.tokenizer.vocab_size

        if any(filter(lambda x: x >= vocab_size, token_ids)):
            table_names = db['table_names_original']
            try:
                if token_ids[0] == self.tokenizer.cls_token_id: token_ids = token_ids[1:]
                if token_ids[-1] == self.tokenizer.sep_token_id: token_ids = token_ids[:-1]
                column_names = list(map(lambda x: table_names[x[0]] + '.' + x[1] if x[1] != '*' else '*', db['column_names_original']))
                replacements = map(lambda idx: column_names[idx - vocab_size - len(table_names)] if idx >= vocab_size + len(table_names) else table_names[idx - vocab_size], filter(lambda idx: idx >= vocab_size, token_ids))
                mask_token, mask_token_id = self.tokenizer.pad_token, self.tokenizer.pad_token_id
                masked_token_ids = list(map(lambda idx: mask_token_id if idx >= vocab_size else idx, token_ids))
                masked_query = self.tokenizer.convert_tokens_to_string(self.tokenizer.convert_ids_to_tokens(masked_token_ids))
                for repl in replacements:
                    masked_query = masked_query.replace(mask_token, ' ' + repl + ' ', 1)
                return masked_query, True
            except:
                return f'select * from {table_names[0]}', False

        tokens = self.tokenizer.convert_ids_to_tokens(token_ids, skip_special_tokens=True)
        return self.tokenizer.convert_tokens_to_string(tokens), True


if __name__ == '__main__':

    import argparse
    from tqdm import tqdm
    from eval.evaluation import evaluate, build_foreign_key_map_from_json

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('-d', dest='dataset', choices=['spider', 'sparc', 'cosql'], help='dataset name')
    arg_parser.add_argument('-t', dest='tokenizer', default='grappa_large_jnt', help='tokenizer or PLM name, stored in ./pretrained_models/ directory')
    arg_parser.add_argument('-o', dest='output', default='ast', choices=['ast', 'seq'], help='decode method: AST or Sequence')
    arg_parser.add_argument('-e', dest='etype', default='all', choices=['match', 'exec', 'all'], help='evaluation metric, exact set match or execution accuracy or both')
    args = arg_parser.parse_args(sys.argv[1:])

    tranx = TransitionSystem(args.dataset, tokenizer=args.tokenizer)
    data_dir = os.path.dirname(CONFIG_PATHS[args.dataset]['train'])
    db_dir = CONFIG_PATHS[args.dataset]['db_dir']
    kmaps = build_foreign_key_map_from_json(CONFIG_PATHS[args.dataset]['tables'])
    tables = {db['db_id']: db for db in json.load(open(CONFIG_PATHS[args.dataset]['tables']))}
    train, dev = json.load(open(CONFIG_PATHS[args.dataset]['train'], 'r')), json.load(open(CONFIG_PATHS[args.dataset]['dev'], 'r'))


    def create_gold_sql(choice):
        gold_path = os.path.join(data_dir, choice + '_gold.sql')
        dataset = train if choice == 'train' else dev
        with open(gold_path, 'w') as of:
            for ex in dataset:
                if 'interaction' in ex:
                    for turn in ex['interaction']:
                        query = ' '.join(turn['query'].split('\n'))
                        of.write(' '.join(query.split('\t')) + '\t' + ex['database_id'] + '\n')
                    of.write('\n')
                else:
                    query = ' '.join(ex['query'].split('\n'))
                    of.write(' '.join(query.split('\t')) + '\t' + ex['db_id'] + '\n')
        return


    def sql_to_output_to_sql(dataset, output='ast'):
        recovered_sqls = []
        for ex in tqdm(dataset, desc='Parse SQL to AST and back to SQL', total=len(dataset)):
            if 'interaction' in ex:
                db = tables[ex['database_id']]
                turn_sqls = []
                for turn in ex['interaction']:
                    if output == 'ast':
                        sql_ast = tranx.parse_sql_to_ast(turn['sql'], db)
                        recovered_sql, flag = tranx.unparse_sql_from_ast(sql_ast, db, turn)
                        assert flag
                    else:
                        sql_seq = tranx.parse_sql_to_seq(turn['sql'], db)
                        recovered_sql, flag = tranx.unparse_sql_from_seq(sql_seq, db, turn)
                    turn_sqls.append(recovered_sql)
                recovered_sqls.append(turn_sqls)
            else:
                db = tables[ex['db_id']]
                if output == 'ast':
                    sql_ast = tranx.parse_sql_to_ast(ex['sql'], db)
                    recovered_sql, flag = tranx.unparse_sql_from_ast(sql_ast, db, ex)
                    assert flag
                else:
                    sql_seq = tranx.parse_sql_to_seq(ex['sql'], db)
                    recovered_sql, flag = tranx.unparse_sql_from_seq(sql_seq, db, ex)
                recovered_sqls.append(recovered_sql)
        return recovered_sqls


    def evaluate_sqls(recovered_sqls, choice='train', etype='all'):
        pred_path = os.path.join(data_dir, choice + '_pred.sql')
        with open(pred_path, 'w') as of:
            for each in recovered_sqls:
                if type(each) != str:
                    for turn in each: of.write(turn + '\n')
                    of.write('\n')
                else:
                    of.write(each + '\n')
        gold_path = os.path.join(data_dir, choice + '_gold.sql')
        output_path = os.path.join(data_dir, choice + '_eval.log')
        with open(output_path, 'w') as of:
            sys.stdout, old_print = of, sys.stdout
            evaluate(gold_path, pred_path, db_dir, etype, kmaps, False, False, False)
            sys.stdout = old_print


    # create_gold_sql('train')
    # train_sqls = sql_to_output_to_sql(train, args.output)
    # evaluate_sqls(train_sqls, 'train', args.etype)


    create_gold_sql('dev')
    dev_sqls = sql_to_output_to_sql(dev, args.output)
    evaluate_sqls(dev_sqls, 'dev', args.etype)