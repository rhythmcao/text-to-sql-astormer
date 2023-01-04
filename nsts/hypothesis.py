#coding=utf-8
import torch
from copy import deepcopy
from typing import List, Tuple
from nsts.asdl import ASDLCompositeType, Field
from nsts.asdl_ast import AbstractSyntaxTree, RealizedField
from nsts.transition_system import Action, ApplyRuleAction, GenerateTokenAction, TransitionSystem


class Hypothesis(object):
    """ Given the sequence of actions, construct the target SQL AST step-by-step,
    and maintain necessary infos such as decoder relational matrix and frontier nodes.
    """
    def __init__(self, tranx: TransitionSystem, decode_order: str = 'dfs+l2r') -> None:
        self.tranx, self.decode_order = tranx, decode_order
        self.tree: AbstractSyntaxTree = None # root start with None
        self.actions: List[Action] = [] # list of Action
        self.relations: List[List[int]] = [] # list of triangular relational matrix
        self.frontier_info: List[RealizedField] = [None] # record the unrealized fields of all unexpanded nodes
        self.frontier_path: List[List[Tuple[Field, int]]] = [()] # record the path to each unrealized fields
        self._value_buffer: List[int] = [] # for GenerateTokenAction, store incomplete buffered tokens
        self._value_frontier: int = -1 # the frontier idx for val_id if len(self._value_buffer) > 0
        self.t: int = 0 # record the decoding timestep
        self.score: int = 0. # record the sum of logscores


    def get_relation(self, frontier_idx: int = 0, device = 'cpu') -> torch.LongTensor:
        """ Obtain the relation matrix for the frontier_idx frontier node.
        """
        return self.tranx.ast_relation.padding_relation(self.relations + [self.get_child_relation(self.frontier_info[frontier_idx])], device)


    def get_child_relation(self, frontier_field: RealizedField = None) -> List[int]:
        """ Obtain the child relations for the given frontier node.
        """
        if frontier_field is None: return [self.tranx.ast_relation.relation2id['0-0']]

        parent_timestep = frontier_field.parent_node.created_time
        if frontier_field.type.name == 'val_id' and len(self._value_buffer) > 0:
            token_start_timestep = self.t - len(self._value_buffer)
            child_relation = self.tranx.ast_relation.add_relation_for_child(self.relations, parent_timestep, token_start_timestep)
        else: child_relation = self.tranx.ast_relation.add_relation_for_child(self.relations, parent_timestep)
        return child_relation


    def get_next_frontier_ids(self) -> List[int]:
        """ Give the pre-defined traverse order~(self.decode_order), return the idxs of frontier nodes/fields to be expanded for the next timestep
        """
        if len(self._value_buffer) > 0: # GenerateTokenAction not stopped
            return [self._value_frontier]

        if 'l2r' in self.decode_order: return [0] # only the left one
        elif self.decode_order == 'random': return list(range(len(self.frontier_info)))
        else: # dfs+random / bfs+random
            next_frontier_ids = [0]
            parent_node_address = id(self.frontier_info[0].parent_node)
            for idx, realized_field in enumerate(self.frontier_info[1:]):
                if id(realized_field.parent_node) == parent_node_address:
                    next_frontier_ids.append(idx)
            return next_frontier_ids


    def apply_action(self, action: Action, frontier_idx: int = 0):
        """ Expand the frontier field of index frontier_idx with action, update the set of frontier nodes and store history information.
        frontier_idx~(int): index of the current node to be expanded in the self.frontier_info
        """
        grammar = self.tranx.grammar
        frontier_field = self.frontier_info[frontier_idx]
        self.relations.append(self.get_child_relation(frontier_field))

        if self.tree is None: # the first action must be ApplyRule
            assert isinstance(action, ApplyRuleAction) and self.t == 0 and frontier_idx == 0, 'Invalid action [%s], only ApplyRule action is valid ' \
                                                        'at the beginning of decoding' % (action)
            production = grammar.id2prod[action.production_id]
            self.tree = AbstractSyntaxTree(production, created_time=self.t, depth=0)
            self.update_frontier_info(frontier_idx)
        else:
            if isinstance(frontier_field.type, ASDLCompositeType):
                assert isinstance(action, ApplyRuleAction)
                depth = frontier_field.parent_node.depth + 1
                value = AbstractSyntaxTree(grammar.id2prod[action.production_id], created_time=self.t, depth=depth)
                frontier_field.add_value(value, realized_time=self.t)
                self.update_frontier_info(frontier_idx)
            else:
                if isinstance(action, GenerateTokenAction):
                    self._value_buffer.append(action.token)
                    if action.is_stop_signal():
                        frontier_field.add_value(self._value_buffer, realized_time=self.t)
                        self._value_buffer, self._value_frontier = [], -1
                        self.update_frontier_info(frontier_idx)
                    else: self._value_frontier = frontier_idx # not update frontier_info in this case
                else: # SelectTable or SelectColumn action
                    frontier_field.add_value(int(action.token), realized_time=self.t)
                    self.update_frontier_info(frontier_idx)

        self.t += 1
        self.actions.append(action)
        return self


    def update_frontier_info(self, frontier_idx: int = 0):
        """ Update the set of frontier info given the frontier field expanded at the current timestep.
        Concretely, it pushes all the children fields of the current node into the frontier set.
        """
        frontier_field, current_path = self.frontier_info[frontier_idx], self.frontier_path[frontier_idx]

        if frontier_field is None: # root node
            current_node = self.tree
            unrealized_fields = [current_node[field][0] for field in current_node.fields]
            self.frontier_info.pop(frontier_idx)
            self.frontier_info.extend(unrealized_fields)
            self.frontier_path.pop(frontier_idx)
            unrealized_paths = [current_path + ((field, 0),) for field in current_node.fields]
            self.frontier_path.extend(unrealized_paths)
            return

        # whether remove this RealizedField or replace with the next one
        parent_node, field = frontier_field.parent_node, frontier_field.field
        parent_node.tracker[field] += 1
        if parent_node.field_finished(field):
            self.frontier_info.pop(frontier_idx)
            self.frontier_path.pop(frontier_idx)
        else:
            self.frontier_info[frontier_idx] = parent_node[field][parent_node.tracker[field]]
            self.frontier_path[frontier_idx] = current_path[:-1] + ((current_path[-1][0], parent_node.tracker[field]), )


        # add children of frontier field to self.frontier_info
        current_node = frontier_field.value
        if isinstance(current_node, AbstractSyntaxTree) and len(current_node.fields) > 0:
            unrealized_fields = [current_node[field][0] for field in current_node.fields]
            unrealized_paths = [current_path + ((field, 0),) for field in current_node.fields]
            if self.decode_order.startswith('bfs'):
                self.frontier_info.extend(unrealized_fields)
                self.frontier_path.extend(unrealized_paths)
            else:
                self.frontier_info = unrealized_fields.extend(self.frontier_info)
                self.frontier_path = unrealized_paths.extend(self.frontier_path)
        else: pass # primitive types, no children
        return


    def clone_and_apply_action(self, action, frontier_idx=0):
        """ Clone the original Hypothesis and apply the action to frontier field with index frontier_idx.
        """
        new_hyp = self.copy()
        new_hyp.apply_action(action, frontier_idx=frontier_idx)
        return new_hyp


    def copy(self):
        new_hyp = Hypothesis(self.tranx, self.decode_order)
        if self.tree:
            new_hyp.tree = self.tree.copy()
            new_hyp.frontier_path = list(self.frontier_path)
            new_hyp.frontier_info = new_hyp.retrieve_frontier_info_from_path()
            new_hyp.relations = deepcopy(self.relations)

        new_hyp.actions = list(self.actions)
        new_hyp._value_buffer = list(self._value_buffer)
        new_hyp._value_frontier = self._value_frontier
        new_hyp.t = self.t
        new_hyp.score = self.score
        return new_hyp


    def retrieve_frontier_info_from_path(self):
        frontier_info = []
        for path in self.frontier_path:
            rfield, node = None, self.tree
            for f, idx in path:
                rfield = node[f][idx]
                node = rfield.value
            frontier_info.append(rfield)
        return frontier_info


    @property
    def completed(self):
        return self.tree is not None and len(self.frontier_info) == 0