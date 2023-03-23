#coding=utf8
import torch
import torch.nn.functional as F
from functools import partial
from utils.example import Example, get_position_ids
from model.model_utils import lens2mask, cached_property, make_relative_positions
from asdl.transition_system import ApplyRuleAction, SelectTableAction, SelectColumnAction
from asdl.relation_utils import ENCODER_RELATIONS
from preprocess.preprocess_utils import get_question_relation


def from_example_list_encoder(ex_list, device='cpu', train=True, **kwargs):
    """
        question_lens: torch.long, bsize, number of tokens for each question
        schema_lens: torch.long, bsize, number of tables and columns for each example
        schema_token_lens: torch.long, number of tokens for each schema item
    """
    batch = Batch(ex_list, device)
    encode_method, pad_idx = Example.encode_method, Example.tokenizer.pad_token_id

    batch.schema_token_lens = torch.tensor(sum([ex.table_token_len + ex.column_token_len for ex in ex_list], []), dtype=torch.long, device=device)
    batch.schema_lens = torch.tensor([ex.schema_len for ex in ex_list], dtype=torch.long, device=device)

    # prepare inputs for pretrained models
    batch.inputs = {"input_ids": None, "attention_mask": None, "token_type_ids": None, "position_ids": None}
    input_lens = [len(ex.input_id) for ex in ex_list]
    max_len = max(input_lens)
    input_ids = [ex.input_id + [pad_idx] * (max_len - len(ex.input_id)) for ex in ex_list]
    batch.inputs["input_ids"] = torch.tensor(input_ids, dtype=torch.long, device=device)
    attention_mask = [[1] * l + [0] * (max_len - l) for l in input_lens]
    batch.inputs["attention_mask"] = torch.tensor(attention_mask, dtype=torch.float, device=device)
    token_type_ids = [ex.segment_id + [0] * (max_len - len(ex.segment_id)) for ex in ex_list]
    batch.inputs["token_type_ids"] = torch.tensor(token_type_ids, dtype=torch.long, device=device)
    position_ids = [get_position_ids(ex, shuffle=train) + [0] * (max_len - len(ex.input_id)) for ex in ex_list]
    batch.inputs["position_ids"] = torch.tensor(position_ids, dtype=torch.long, device=device)

    # extract representations after plm
    plm_question_mask = [ex.plm_question_mask + [0] * (max_len - len(ex.input_id)) for ex in ex_list]
    batch.plm_question_mask = torch.tensor(plm_question_mask, dtype=torch.bool, device=device)
    plm_schema_mask = [ex.plm_schema_mask + [0] * (max_len - len(ex.input_id)) for ex in ex_list]
    batch.plm_schema_mask = torch.tensor(plm_schema_mask, dtype=torch.bool, device=device)

    # for decoder memories
    if encode_method == 'rgatsql':
        batch.question_lens = torch.tensor([ex.question_len for ex in ex_list], dtype=torch.long, device=device)
        batch.mask = lens2mask(batch.question_lens + batch.schema_lens)
        max_len = batch.mask.size(-1)
        batch.select_schema_mask = torch.tensor([ex.select_schema_mask + [0] * (max_len - len(ex.select_schema_mask)) for ex in ex_list], dtype=torch.bool, device=device)
        batch.select_copy_mask = torch.tensor([ex.select_copy_mask + [0] * (max_len - len(ex.select_copy_mask)) for ex in ex_list], dtype=torch.bool, device=device)
        batch.copy_mask = batch.question_mask # only question token ids
        max_copy_len = batch.question_mask.size(-1)
        batch.copy_ids = torch.tensor([ex.copy_id + [pad_idx] * (max_copy_len - len(ex.copy_id)) for ex in ex_list], dtype=torch.long, device=device)
        pad_idx = ENCODER_RELATIONS.index('padding-padding')
        batch.encoder_relations = torch.stack([
            F.pad(torch.cat(
                    [
                        torch.cat([get_question_relation(ex.separator_pos), ex.encoder_relation[0]], dim=1),
                        torch.cat([ex.encoder_relation[1], ex.db['relation']], dim=1)
                    ], dim=0
                ), (0, max_len - len(ex.select_schema_mask), 0, max_len - len(ex.select_schema_mask)), value=pad_idx) for ex in ex_list
        ], dim=0).to(device)
        batch.encoder_relations_mask = batch.encoder_relations == pad_idx
    else:
        batch.mask = batch.inputs["attention_mask"].bool()
        batch.select_schema_mask = batch.plm_schema_mask
        batch.select_copy_mask = torch.tensor([ex.select_copy_mask + [0] * (max_len - len(ex.input_id)) for ex in ex_list], dtype=torch.bool, device=device)
        copy_lens = torch.tensor([len(ex.copy_id) for ex in ex_list], dtype=torch.long)
        batch.copy_mask = lens2mask(copy_lens).to(device)
        max_copy_len = batch.copy_mask.size(-1)
        batch.copy_ids = torch.tensor([ex.copy_id + [pad_idx] * (max_copy_len - len(ex.copy_id)) for ex in ex_list], dtype=torch.long, device=device)
    return batch


def from_example_list_decoder(ex_list, batch, device='cpu', train=True, decode_order='dfs+l2r', **kwargs):
    if train:
        decode_method = Example.decode_method

        if decode_method == 'ast':
            # action_ids, production_ids, field_ids, depth_ids, decoder_relations
            action_infos_list, relations_list = list(zip(*[Example.tranx.get_outputs_from_ast(action_infos=ex.action_info, relations=ex.decoder_relation, order=decode_order)
                if decode_order != 'dfs+l2r' else (ex.action, ex.decoder_relation) for ex in ex_list]))
            max_action_num = max([len(action) for action in action_infos_list])
            vocab_size, grammar_size = Example.tokenizer.vocab_size, len(Example.grammar.prod2id) + 1
            table_nums = [len(ex.db['table_names']) for ex in ex_list]
            ast_utils = Example.tranx.ast_relation
            rel_pad_idx = ast_utils.relation2id['padding-padding']

            def get_action_id(action_info, eid):
                if action_info.action_type == ApplyRuleAction:
                    return vocab_size + action_info.action_id
                elif action_info.action_type == SelectTableAction:
                    return vocab_size + grammar_size + action_info.action_id
                elif action_info.action_type == SelectColumnAction:
                    return vocab_size + grammar_size + table_nums[eid] + action_info.action_id
                else: return action_info.action_id

            def get_decoder_relation(relation):
                return F.pad(relation, (0, max_action_num - relation.size(0), 0, max_action_num - relation.size(0)), value=rel_pad_idx)

            def get_node_cross_action_relation(relation):
                root_rels, relation = relation[:, 0].tolist(), relation[:, :-1]
                root_rels = torch.tensor([ast_utils.child_relation_mappings[rel_id] for rel_id in root_rels], dtype=torch.long).unsqueeze(-1)
                relation = torch.cat([root_rels, relation], dim=-1)
                return F.pad(relation, (0, max_action_num - relation.size(0), 0, max_action_num - relation.size(0)), value=rel_pad_idx)

            def get_shift_decoder_relation(relation):
                shift_relation = relation[:-1, :-1]
                pad_relation = F.pad(shift_relation, (1, max_action_num - relation.size(0), 1, max_action_num - relation.size(0)), value=rel_pad_idx)
                root_rels = torch.tensor([ast_utils.child_relation_mappings[rel_id] for rel_id in shift_relation[:, 0].tolist()], dtype=torch.long)
                pad_relation[1:shift_relation.size(0) + 1, 0] = root_rels
                pad_relation[0, 0] = ast_utils.relation2id['0-0']
                return pad_relation

            batch.ast_actions = torch.tensor([[get_action_id(action_info, eid) for action_info in action_infos] + [0] * (max_action_num - len(action_infos)) for eid, action_infos in enumerate(action_infos_list)], dtype=torch.long, device=device)
            batch.production_ids = torch.tensor([[action_info.prod_id for action_info in action_infos] + [0] * (max_action_num - len(action_infos)) for action_infos in action_infos_list], dtype=torch.long, device=device)
            batch.field_ids = torch.tensor([[action_info.field_id for action_info in action_infos] + [0] * (max_action_num - len(action_infos)) for action_infos in action_infos_list], dtype=torch.long, device=device)
            batch.depth_ids = torch.tensor([[action_info.depth for action_info in action_infos] + [0] * (max_action_num - len(action_infos)) for action_infos in action_infos_list], dtype=torch.long, device=device)
            # relations_list = [make_relative_positions(rel.size(0)) for rel in relations_list] # use relation p_j - p_i
            batch.decoder_relations = torch.stack([get_decoder_relation(relation) for relation in relations_list]).to(device)
            # batch.shift_decoder_relations = torch.stack([get_shift_decoder_relation(relation) for relation in relations_list]).to(device)
            # batch.node_cross_action_relations = torch.stack([get_node_cross_action_relation(relation) for relation in relations_list]).to(device)
            batch.tgt_mask = lens2mask(torch.tensor([len(action_infos) for action_infos in action_infos_list], dtype=torch.long)).to(device)
            batch.action_infos = action_infos_list # to retrieve parent hidden state timestep, LSTM decoder
        else: # sequence decoder
            seq_actions = [ex.action for ex in ex_list]
            max_action_num = max([len(actions) for actions in seq_actions])
            pad_idx = Example.tranx.tokenizer.pad_token_id
            batch.seq_actions = torch.tensor([actions + [pad_idx] * (max_action_num - len(actions)) for actions in seq_actions], dtype=torch.long, device=device)
            # remember to minus 1 due to the shifted input operation
            batch.tgt_mask = lens2mask(torch.tensor([len(actions) - 1 for actions in seq_actions], dtype=torch.long)).to(device)
    else:
        batch.max_action_num = 150
        batch.database = [{'table': len(ex.db['table_names']), 'column': len(ex.db['column_names'])} for ex in ex_list]
    return batch


class Batch():

    def __init__(self, examples, device='cpu'):
        super(Batch, self).__init__()
        self.examples = examples
        self.device = device


    @classmethod
    def get_collate_fn(cls, **kwargs):
        return partial(cls.from_example_list, **kwargs)


    @classmethod
    def from_example_list(cls, ex_list, device='cpu', train=True, decode_order='dfs+l2r', **kwargs):
        batch = from_example_list_encoder(ex_list, device, train, **kwargs)
        batch = from_example_list_decoder(ex_list, batch, device, train, decode_order, **kwargs)
        return batch


    def __len__(self):
        return len(self.examples)


    def __getitem__(self, idx):
        return self.examples[idx]


    @cached_property
    def question_mask(self):
        return lens2mask(self.question_lens)


    @cached_property
    def schema_mask(self):
        return lens2mask(self.schema_lens)


    @cached_property
    def schema_token_mask(self):
        return lens2mask(self.schema_token_lens)


    def get_parent_state_ids(self, t):
        return [a[t].parent_timestep if t < len(a) else 0 for a in self.action_infos]
