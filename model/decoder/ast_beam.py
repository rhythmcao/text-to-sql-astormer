#coding=utf8
import torch
from typing import List
from nsts.hypothesis import Hypothesis
from nsts.transition_system import Action, ApplyRuleAction, SelectTableAction, SelectColumnAction, GenerateTokenAction, TransitionSystem


class ASTBeam():
    """ Maintain a beam of hypothesis during decoding for each example
    """
    def __init__(self, tranx: TransitionSystem, database: dict, beam_size: int = 5, n_best: int = 5, decode_order='dfs+l2r', device: torch.device = None, **kwargs) -> None:
        """
        @args:
            database: a dict store the number of tables and columns for the current sample
            decode_order: how to update the frontier fields for the partially constructed AST
        """
        assert beam_size >= 1
        self.tranx, self.grammar = tranx, tranx.grammar
        self.token_num, self.table_num, self.column_num = self.tranx.tokenizer.vocab_size, database['table'], database['column']
        self.beam_size, self.n_best, self.decode_order = beam_size, n_best, decode_order
        self.device = device
        # record the current hypothesis and current input fields
        self.hyps: List[Hypothesis] = [Hypothesis(self.tranx, self.decode_order)]
        self.frontier_ids: List[int] = [0] # one-to-one corresponding to self.hyps
        self.live_hyp_ids: List[int] = [0] # record the index of previous hyp in self.hyps
        self.completed_hyps: List[Hypothesis] = []


    def get_parent_prod_ids(self) -> torch.LongTensor:
        return torch.tensor([
            self.grammar.prod2id[hyp.frontier_info[fid].parent_node.production]
                for hyp, fid in zip(self.hyps, self.frontier_ids)], dtype=torch.long, device=self.device)


    def get_current_field_ids(self) -> torch.LongTensor:
        return torch.tensor([
            self.grammar.field2id[hyp.frontier_info[fid].field]
                for hyp, fid in zip(self.hyps, self.frontier_ids)], dtype=torch.long, device=self.device)


    def get_current_depth_ids(self) -> torch.LongTensor:
        return torch.tensor([
            hyp.frontier_info[fid].parent_node.depth + 1
                for hyp, fid in zip(self.hyps, self.frontier_ids)], dtype=torch.long, device=self.device)


    def get_parent_timesteps(self) -> torch.LongTensor:
        return torch.tensor([
            hyp.frontier_info[fid].parent_node.created_time
                for hyp, fid in zip(self.hyps, self.frontier_ids)], dtype=torch.long, device=self.device)


    def get_previous_actions(self) -> List[Action]:
        return [hyp.actions[-1] for hyp in self.hyps]


    def get_previous_hyp_ids(self, offset: int = 0) -> List[int]: # offset of other samples
        return [idx + offset for idx in self.live_hyp_ids]


    def advance(self, ar_scores: torch.FloatTensor, ss_scores: torch.FloatTensor, gt_scores: torch.FloatTensor):
        """ Scores for apply rule, select schema and generate tokens.
        """
        prev_hyp_ids, action_ids, hyp_scores = [], [], []
        for idx in range(len(self.hyps)):
            hyp, fid = self.hyps[idx], self.frontier_ids[idx]
            frontier_field = hyp.frontier_info[fid]
            act_type = self.tranx.field_to_action(frontier_field)
            if act_type == ApplyRuleAction:
                frontier_type = frontier_field.type if frontier_field else self.grammar.root_type
                valid_productions = self.grammar[frontier_type]
                prod_ids = torch.tensor([self.grammar.prod2id[prod] for prod in valid_productions], dtype=torch.long, device=self.device)
                prev_hyp_ids.append(prod_ids.new_full(prod_ids.size(), idx))
                action_ids.append(prod_ids)
                prod_scores = torch.index_select(ar_scores[idx], 0, prod_ids)
                hyp_scores.append(hyp.score + prod_scores)
            elif act_type == SelectTableAction:
                table_ids = torch.arange(self.table_num, device=self.device)
                prev_hyp_ids.append(table_ids.new_full(table_ids.size(), idx))
                action_ids.append(table_ids)
                table_scores = ss_scores[idx, :self.table_num]
                hyp_scores.append(hyp.score + table_scores)
            elif act_type == SelectColumnAction:
                column_ids = torch.arange(self.column_num, device=self.device)
                prev_hyp_ids.append(column_ids.new_full(column_ids.size(), idx))
                action_ids.append(column_ids)
                column_scores = ss_scores[idx, self.table_num: self.table_num + self.column_num]
                hyp_scores.append(hyp.score + column_scores)
            else:
                if len(hyp._value_buffer) == 0: # at least one non-EOS token
                    gt_scores[idx, GenerateTokenAction.EOV_ID] -= 1e20
                token_ids = torch.arange(self.token_num, device=self.device)
                prev_hyp_ids.append(token_ids.new_full(token_ids.size(), idx))
                action_ids.append(token_ids)
                hyp_scores.append(hyp.score + gt_scores[idx])

        prev_hyp_ids, action_ids, hyp_scores = torch.cat(prev_hyp_ids, dim=0), torch.cat(action_ids, dim=0), torch.cat(hyp_scores, dim=0)
        topk_hyp_scores, meta_ids = hyp_scores.topk(min(self.beam_size, hyp_scores.size(0)), -1, True, True)
        prev_hyp_ids = torch.index_select(prev_hyp_ids, dim=0, index=meta_ids).tolist()
        action_ids = torch.index_select(action_ids, dim=0, index=meta_ids).tolist()

        # update the hypothesis and fields list
        new_hyps, new_frontier_ids, live_hyp_ids = [], [], []
        for hyp_score, hyp_id, act_id in zip(topk_hyp_scores, prev_hyp_ids, action_ids):
            hyp, fid = self.hyps[hyp_id], self.frontier_ids[hyp_id]
            refield = hyp.frontier_info[fid]
            action = self.tranx.field_to_action(refield)(act_id)
            new_hyp = hyp.clone_and_apply_action(action, fid)
            new_hyp.score = hyp_score

            if new_hyp.completed:
                self.completed_hyps.append(new_hyp)
                continue

            next_frontier_ids = new_hyp.get_next_frontier_ids()
            new_frontier_ids.extend(next_frontier_ids)
            new_hyps.extend([new_hyp] * len(next_frontier_ids))
            live_hyp_ids.extend([hyp_id] * len(next_frontier_ids))

        if new_hyps: self.hyps, self.frontier_ids, self.live_hyp_ids = new_hyps, new_frontier_ids, live_hyp_ids
        else: self.hyps, self.frontier_ids, self.live_hyp_ids = [], [], []


    @property
    def done(self):
        return len(self.completed_hyps) >= self.n_best or self.hyps == []


    def sort_finished(self):
        if len(self.completed_hyps) > 0:
            size = min([self.n_best, len(self.completed_hyps)])
            completed_hyps = sorted(self.completed_hyps, key=lambda hyp: - hyp.score)[:size] # / hyp.tree.size
        else:
            completed_hyps = [Hypothesis(self.tranx, self.decode_order)]
        return completed_hyps
