#coding=utf8
import torch
import torch.nn.functional as F
from itertools import product
from typing import List, Dict


# relations: type_1-type_2-relation_name, r in relation_name represents reverse edge, b represents bidirectional edge
MAX_RELATIVE_DIST = 4
ENCODER_RELATIONS = ['padding-padding'] + [f'question-question-dist{i:d}' for i in range(- MAX_RELATIVE_DIST, MAX_RELATIVE_DIST + 1)] + \
    [f'question-question-previous', f'question-question-after'] + \
    ['table-table-identity', 'table-table-fk', 'table-table-fkr', 'table-table-fkb'] + \
    ['column-column-identity', 'column-column-sametable', 'column-column-fk', 'column-column-fkr'] + \
    ['table-column-pk', 'column-table-pk', 'table-column-has', 'column-table-has'] + \
    ['question-column-exactmatch', 'question-column-partialmatch', 'question-column-nomatch', 'question-column-valuematch',
    'column-question-exactmatch', 'column-question-partialmatch', 'column-question-nomatch', 'column-question-valuematch'] + \
    ['question-table-exactmatch', 'question-table-partialmatch', 'question-table-nomatch',
    'table-question-exactmatch', 'table-question-partialmatch', 'table-question-nomatch'] + \
    ['question-question-generic', 'table-table-generic', 'column-column-generic', 'table-column-generic', 'column-table-generic']


class ASTRelation():

    # relative relations are constructed by num-num, num is the distance to the common ancestor, such that
    # 0-0 means self-loop, 0-1 means parent-child relation, 1-1 means siblings and 0-2 means grandparent-grandchild relation
    # to avoid infinite distance, we restrict the maximum distance to 4, which can be interpreted as the ancestor or descendant
    MAX_ABSOLUTE_DEPTH = 20
    MAX_RELATIVE_DEPTH = 6
    MAX_TOKEN_DISTANCE = 4
    DECODER_RELATIONS = ['padding-padding'] + [str(i) + '-' + str(j) for i, j in product(range(MAX_RELATIVE_DEPTH), range(MAX_RELATIVE_DEPTH))] + \
        [f'left{i:d}' for i in range(1, MAX_TOKEN_DISTANCE)] # for multi-token SQL values, re-use 0-0 to mean the same token

    def __init__(self) -> None:
        super(ASTRelation, self).__init__()
        self.relation2id = {rel: idx for idx, rel in enumerate(ASTRelation.DECODER_RELATIONS)}
        self.id2relation = ASTRelation.DECODER_RELATIONS
        self.reverse_relation_mappings = self._construct_reverse_relation_mappings()
        self.child_relation_mappings = self._construct_child_relation_mappings()


    def __len__(self) -> int:
        return len(self.relation2id)


    def __getitem__(self, key: str) -> int:
        return self.relation2id[key]


    def _construct_child_relation_mappings(self) -> Dict[int, int]:
        """ Construct a dict which maps the relation of the parent to its children, e.g.,
        parent-child relation `0-1` -> grandparent-grandchild relation `0-2`
        """
        child_relation_mappings = {}
        for rel in self.relation2id:
            if 'padding' in rel or 'left' in rel: continue
            l, r = rel.split('-')
            r = int(r)
            new_r = r + 1 if r + 1 < ASTRelation.MAX_RELATIVE_DEPTH else ASTRelation.MAX_RELATIVE_DEPTH - 1
            new_rel = '-'.join([l, str(new_r)])
            child_relation_mappings[self.relation2id[rel]] = self.relation2id[new_rel]
        return child_relation_mappings


    def _construct_reverse_relation_mappings(self) -> Dict[int, int]:
        """ Construct a dict which maps a relation~(int) to its reverse relation~(int), e.g.,
        parent-child relation `0-1` -> child-parent relation `1-0`
        """
        reverse_relation_mappings = {}
        for rel in self.relation2id:
            rev_rel = '-'.join(rel.split('-')[::-1]) if 'left' not in rel and 'padding' not in rel else 'padding-padding'
            reverse_relation_mappings[self.relation2id[rel]] = self.relation2id[rev_rel]
        return reverse_relation_mappings


    def add_relation_for_child(self, history_relations: List[List[int]], parent_timestep: int, token_start_timestep: int = -1, **kwargs) -> List[int]:
        """ Given relations in history, construct the relations for the child of node expanded at parent_timestep.
        token_start_timestep is used~(>0) only when generating inner multi-token GenerateTokenAction. It denotes the first/starting timestep when generating the SQL value.
        In other words from token_start_timestep to the current timestep, the decoder is generating exactly the same SQL value.
        """
        if token_start_timestep < 0 or token_start_timestep == len(history_relations):
            child_relations = list(map(lambda idx: self.child_relation_mappings[idx], history_relations[parent_timestep]))
            for timestep in range(parent_timestep + 1, len(history_relations)):
                rel_id = self.child_relation_mappings[self.reverse_relation_mappings[history_relations[timestep][parent_timestep]]]
                child_relations.append(rel_id)
        else:
            child_relations = list(history_relations[token_start_timestep])[:-1]
            inner_token_relations = [self.relation2id[f'left{idx}'] if idx < ASTRelation.MAX_TOKEN_DISTANCE else self.relation2id[f'left{ASTRelation.MAX_TOKEN_DISTANCE - 1:d}']
                for idx in range(len(history_relations) - token_start_timestep, 0, -1)]
            child_relations.extend(inner_token_relations)
        child_relations.append(self.relation2id['0-0']) # add self-loop
        return child_relations


    def complete_triangular_relation_matrix(self, relation_matrix: List[List[int]]) -> torch.LongTensor:
        """ For permutation invariant training, we pre-calculate the full relation matrix and sort according to the key timestep.
        @input:
            relation_matrix: triangular relation matrix, [[1], [2, 1], [3, 2, 1], ...]
        @return:
            relation_matrix: fully filled relation matrix
        """
        max_len = len(relation_matrix[-1])
        for row_id, row in enumerate(relation_matrix[:-1]):
            row.extend([self.reverse_relation_mappings[relation_matrix[col_id][row_id]] for col_id in range(row_id + 1, max_len)])
        return torch.tensor(relation_matrix, dtype=torch.long)


    def padding_relation(self, relation_list: List[List[int]], device='cpu') -> torch.LongTensor:
        """ Given a triangular relation List, convert it into full relation tensor matrix.
        """
        t, pad_idx = len(relation_list), self.relation2id['padding-padding']
        relation_list = [rel + [pad_idx] * (t - len(rel)) for rel in relation_list]
        return torch.tensor(relation_list, dtype=torch.long, device=device)