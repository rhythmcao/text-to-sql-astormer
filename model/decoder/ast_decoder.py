#coding=utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from model.decoder.ast_beam import ASTBeam
from model.decoder.decoder_cell import DecoupledAstormer, Astormer, LSTM, ONLSTM
from model.model_utils import Registrable, MultiHeadAttention, PointerNetwork, TiedLinearClassifier
from asdl.transition_system import ApplyRuleAction, SelectTableAction, SelectColumnAction, GenerateTokenAction, TransitionSystem


@Registrable.register('ast')
class ASTDecoder(nn.Module):

    def __init__(self, args, tranx: TransitionSystem):
        super(ASTDecoder, self).__init__()
        self.args, self.tranx = args, tranx
        self.grammar, self.relation = self.tranx.grammar, self.tranx.ast_relation
        self.max_depth = self.relation.MAX_ABSOLUTE_DEPTH - 1

        embed_size = args.embed_size # the same size as word embeddings, defined in the encoder
        # embedding matrices for ASDL production rules and fields~(types), plus 1 due to the root node
        self.production_embed = nn.Embedding(len(self.grammar.prod2id) + 1, args.encoder_hidden_size, padding_idx=len(self.grammar.prod2id))
        self.field_embed = nn.Embedding(len(self.grammar.field2id) + 1, args.encoder_hidden_size, padding_idx=len(self.grammar.field2id))
        self.depth_embed = nn.Embedding(self.relation.MAX_ABSOLUTE_DEPTH, args.encoder_hidden_size, padding_idx=0)
        # input of decoder lstm: sum of previous_action + parent_production_rule + current_node_type + tree_depth
        self.input_layer_norm = nn.LayerNorm(args.encoder_hidden_size)
        self.word_embed_affine = nn.Linear(embed_size, args.encoder_hidden_size) if embed_size != args.encoder_hidden_size else lambda x: x

        if args.decoder_cell == 'transformer':
            # assert args.encoder_hidden_size == args.decoder_hidden_size, f'For Astormer architecture, hidden size of encoder({args.encoder_hidden_size:d})/decoder({args.decoder_hidden_size:d}) must be the same.'
            args.decoder_hidden_size = args.encoder_hidden_size
            self.decoder_network = Astormer(args.decoder_hidden_size, args.decoder_num_layers, args.num_heads, args.dropout)
            # self.decoder_network = DecoupledAstormer(args.decoder_hidden_size, args.decoder_num_layers, args.num_heads, args.dropout)
        else:
            self.hidden2action = nn.Linear(args.decoder_hidden_size, args.encoder_hidden_size) if args.decoder_hidden_size != args.encoder_hidden_size else lambda x: x
            cell_constructor = ONLSTM if args.decoder_cell == 'onlstm' else LSTM
            self.decoder_network = cell_constructor(args.encoder_hidden_size, args.decoder_hidden_size, num_layers=args.decoder_num_layers, dropout=args.dropout, dropconnect=args.dropout)
            self.context_attn = MultiHeadAttention(args.decoder_hidden_size, args.encoder_hidden_size, output_size=args.encoder_hidden_size, num_heads=args.num_heads, dropout=args.dropout)
            self.attention_linear = nn.Sequential(nn.Linear(args.decoder_hidden_size + args.encoder_hidden_size, args.decoder_hidden_size), nn.Tanh())

        # output space, ApplyRule, SelectTable, SelectColumn, GenerateToken or copy raw tokens
        self.apply_rule = TiedLinearClassifier(args.decoder_hidden_size, args.encoder_hidden_size)
        self.selector = PointerNetwork(args.decoder_hidden_size, args.encoder_hidden_size, num_heads=args.num_heads, dropout=args.dropout)
        self.generator = TiedLinearClassifier(args.decoder_hidden_size, embed_size)
        # generate token from vocabulary or copy tokens from input
        self.switcher = nn.Sequential(nn.Linear(args.decoder_hidden_size + args.encoder_hidden_size, 1), nn.Sigmoid())


    def score(self, memories, batch, return_attention_weights=False, **kwargs):
        """ Training function for grammar-based AST decoder.
        @args:
            memories: dict stores (key, value) pairs, including `encodings` states (bs x src_len x dim), `schema` memory (bs x (table_num + column_num) x dim),
                `copy` memory (bs x copy_len x dim), `copy_ids` (map input to target vocab position, bs x copy_len), `generator` memory for token generation (vocab_size x dim),
                and masks `mask`, `schema_mask`, `copy_mask` for each memory.
            batch: we use ast_actions, production_ids, field_ids, depth_ids, decoder_relations (if Astormer) and get_parent_state_ids (if LSTM-series)
        @return: sum of training loss
        """
        args = self.args
        encodings, generator_memory, mask = memories['encodings'], memories['generator'], memories['mask']
        schema_memory, schema_mask, max_schema_num = memories['schema'], memories['schema_mask'], memories['schema'].size(1)
        copy_memory, copy_ids, copy_mask = memories['copy'], memories['copy_ids'], memories['copy_mask']

        # previous action embedding, depending on which action_type
        vocab_size, grammar_size = self.tranx.tokenizer.vocab_size, len(self.grammar.prod2id) + 1
        prev_actions, init_actions = batch.ast_actions[:, :-1], encodings.new_zeros((batch.ast_actions.size(0), 1, encodings.size(-1)))
        decoder_token_mask = prev_actions < vocab_size
        decoder_rule_mask = (prev_actions >= vocab_size) & (prev_actions < vocab_size + grammar_size)
        decoder_schema_mask = prev_actions >= vocab_size + grammar_size

        prev_inputs = encodings.new_zeros((prev_actions.size(0), prev_actions.size(1), encodings.size(-1)))
        if torch.any(decoder_token_mask).item(): # SQL value is optional in the minibatch
            token_input = F.embedding(prev_actions.masked_select(decoder_token_mask), generator_memory)
            token_input = self.word_embed_affine(token_input)
            prev_inputs.masked_scatter_(decoder_token_mask.unsqueeze(-1), token_input)
        rule_input = F.embedding((prev_actions - vocab_size).masked_select(decoder_rule_mask), self.production_embed.weight)
        prev_inputs.masked_scatter_(decoder_rule_mask.unsqueeze(-1), rule_input)
        shift_schema_ids = prev_actions - vocab_size - grammar_size + max_schema_num * torch.arange(len(batch), device=encodings.device).unsqueeze(1)
        schema_input = schema_memory.contiguous().view(-1, schema_memory.size(-1))[shift_schema_ids.masked_select(decoder_schema_mask)]
        prev_inputs.masked_scatter_(decoder_schema_mask.unsqueeze(-1), schema_input)
        prev_inputs = torch.cat([init_actions, prev_inputs], dim=1) # right shift one

        # parent production rule embedding and current field embedding
        parent_prods = self.production_embed(batch.production_ids)
        current_fields = self.field_embed(batch.field_ids)

        if args.decoder_cell == 'transformer':
            current_depth = self.depth_embed(torch.clamp(batch.depth_ids, 0, self.max_depth))
            # inputs = self.input_layer_norm(current_fields + current_depth)
            # inputs = self.input_layer_norm(parent_prods + current_fields + current_depth)
            inputs = self.input_layer_norm(prev_inputs + parent_prods + current_fields + current_depth)

            # forward into Astormer
            outputs = self.decoder_network(inputs, encodings, rel_ids=batch.decoder_relations, enc_mask=mask, return_attention_weights=return_attention_weights)
            # outputs = self.decoder_network(inputs, prev_inputs, encodings, rel_ids=batch.decoder_relations,
                # shift_rel_ids=batch.shift_decoder_relations, enc_mask=mask, return_attention_weights=return_attention_weights)
            if return_attention_weights:
                outputs, attention_weights = outputs

            # action logprobs calculation
            apply_rule_logprob = self.apply_rule(outputs, self.production_embed.weight)
            select_schema_logprob = torch.log(self.selector(outputs, schema_memory, mask=schema_mask) + 1e-32)
            gate = self.switcher(torch.cat([outputs, inputs], dim=-1))
            copy_token_prob = self.selector(outputs, copy_memory, mask=copy_mask) * gate
            gen_token_prob = self.generator(outputs, generator_memory, log=False) * (1 - gate)
            copy_ids = copy_ids.unsqueeze(1).expand(-1, copy_token_prob.size(1), -1)
            generate_token_prob = gen_token_prob.scatter_add_(-1, copy_ids, copy_token_prob)
            generate_token_logprob = torch.log(generate_token_prob + 1e-32)
            # token vocabulary, production rules, tables, columns
            logprobs = torch.cat([generate_token_logprob, apply_rule_logprob, select_schema_logprob], dim=-1)

            # loss aggregation
            logprobs = torch.gather(logprobs, dim=-1, index=batch.ast_actions.unsqueeze(-1)).squeeze(-1)
        else:
            inputs = prev_inputs + parent_prods + current_fields
            h_c, history_states, logprobs = None, [], encodings.new_zeros((len(batch), 0)) # parent hidden states feeding

            for t in range(batch.ast_actions.size(1)): # need to extract parent hidden states (parent feeding)
                cur_inputs = inputs[:, t]
                if t > 0:
                    parent_state_ids = batch.get_parent_state_ids(t) # parent hidden state timestep
                    parent_state = torch.stack([history_states[t][eid] for eid, t in enumerate(parent_state_ids)])
                    parent_state = self.hidden2action(parent_state)
                else: parent_state = encodings.new_zeros((len(batch), encodings.size(-1)))
                cur_inputs = self.input_layer_norm(cur_inputs + parent_state)

                # advance decoder lstm and attention calculation
                lstm_outputs, h_c = self.decoder_network(cur_inputs.unsqueeze(1), h_c, start=(t==0))
                history_states.append(h_c[0][-1])
                lstm_outputs = lstm_outputs.squeeze(1)
                context = self.context_attn(lstm_outputs, encodings, mask)
                outputs = self.attention_linear(torch.cat([lstm_outputs, context], dim=-1))

                # action logprobs calculation
                gate = self.switcher(torch.cat([outputs, cur_inputs], dim=-1))
                copy_token_prob = self.selector(outputs, copy_memory, mask=copy_mask) * gate
                gen_token_prob = self.generator(outputs, generator_memory, log=False) * (1 - gate)
                generate_token_prob = gen_token_prob.scatter_add_(-1, copy_ids, copy_token_prob)
                generate_token_logprob = torch.log(generate_token_prob + 1e-32)
                apply_rule_logprob = self.apply_rule(outputs, self.production_embed.weight)
                select_schema_logprob = torch.log(self.selector(outputs, schema_memory, mask=schema_mask) + 1e-32)
                cur_logprobs = torch.cat([generate_token_logprob, apply_rule_logprob, select_schema_logprob], dim=-1)

                # loss aggregation
                cur_logprobs = torch.gather(cur_logprobs, dim=-1, index=batch.ast_actions[:, t].unsqueeze(-1))
                logprobs = torch.cat([logprobs, cur_logprobs], dim=1)

        loss = - logprobs.masked_select(batch.tgt_mask).sum()
        if return_attention_weights:
            return loss, attention_weights
        return loss


    def parse(self, memories, batch, beam_size=5, n_best=5, decode_order='dfs+l2r', **kwargs):
        """ Decoding with beam search for grammar-based AST decoder.
        @args:
            memories: dict stores (key, value) pairs, including `encodings` states (bs x src_len x dim), `schema` memory (bs x (table_num + column_num) x dim),
                `copy` memory (bs x copy_len x dim), `copy_ids` (map input to target vocab position, bs x copy_len), `generator` memory for token generation (vocab_size x dim),
                and masks `mask`, `schema_mask`, `copy_mask` for each memory.
            batch: we use max_action_num and database info (table_number and column_number)
        @return: hypotheses(list): list of hypotheses, bs x List[Hypothesis()]
        """
        args, device, num_examples = self.args, batch.device, len(batch)
        encodings, generator_memory, mask = memories['encodings'], memories['generator'], memories['mask']
        schema_memory, schema_mask = memories['schema'], memories['schema_mask']
        copy_memory, copy_ids, copy_mask = memories['copy'], memories['copy_ids'], memories['copy_mask']

        # prepare data structure to record each sample predictions
        active_idx = list(range(num_examples))
        beams = [ASTBeam(self.tranx, db, beam_size=beam_size, n_best=n_best, decode_order=decode_order, device=device) for db in batch.database]
        if args.decoder_cell == 'transformer':
            history_action_embeds = encodings.new_zeros((num_examples, 0, args.decoder_hidden_size)) 
            prev_inputs = encodings.new_zeros((num_examples, 0, args.decoder_hidden_size))
        else:
            prev_idx = torch.arange(num_examples, dtype=torch.long, device=device)
            h_c, history_states = None, encodings.new_zeros(num_examples, 0, args.decoder_hidden_size)

        for t in range(batch.max_action_num):
            # notice that, different samples may have different number of forward hyps (not necessarily beam size) depending on the number of frontier fields
            select_index = [bid for bid in active_idx for _ in range(len(beams[bid].hyps))]
            cur_encodings, cur_schema_memory, cur_copy_memory = encodings[select_index], schema_memory[select_index], copy_memory[select_index]
            cur_mask, cur_schema_mask, cur_copy_mask, cur_copy_ids = mask[select_index], schema_mask[select_index], copy_mask[select_index], copy_ids[select_index]

            # previous action embedding, parent production, current field, depth or parent hidden states
            if t == 0:
                action_embeds = encodings.new_zeros((num_examples, encodings.size(-1)))
                prod_ids = torch.full((num_examples,), len(self.grammar.prod2id), dtype=torch.long, device=device)
                prod_embeds = self.production_embed(prod_ids)
                field_ids = torch.full((num_examples,), len(self.grammar.field2id), dtype=torch.long, device=device)
                field_embeds = self.field_embed(field_ids)
            else:
                action_embeds = []
                prev_actions = [beams[bid].get_previous_actions() for bid in active_idx]
                for bid, actions in zip(active_idx, prev_actions):
                    for action in actions:
                        if isinstance(action, ApplyRuleAction):
                            action_embeds.append(self.production_embed.weight[action.token])
                        elif isinstance(action, SelectTableAction):
                            action_embeds.append(schema_memory[bid, action.token])
                        elif isinstance(action, SelectColumnAction): # notice the bias of table number for columns
                            action_embeds.append(schema_memory[bid, batch.database[bid]['table'] + action.token])
                        elif isinstance(action, GenerateTokenAction):
                            action_embeds.append(self.word_embed_affine(generator_memory[action.token]))
                        else: raise ValueError('[ERROR]: Unrecognized action type!')
                action_embeds = torch.stack(action_embeds, dim=0)
                prod_ids = torch.cat([beams[bid].get_parent_prod_ids() for bid in active_idx])
                prod_embeds = self.production_embed(prod_ids)
                field_ids = torch.cat([beams[bid].get_current_field_ids() for bid in active_idx])
                field_embeds = self.field_embed(field_ids)

            if args.decoder_cell == 'transformer':
                if t == 0: depth_ids = prod_ids.new_zeros((num_examples,))
                else: depth_ids = torch.cat([beams[bid].get_current_depth_ids() for bid in active_idx])
                depth_embeds = self.depth_embed(torch.clamp(depth_ids, min=0, max=self.max_depth))
                # cur_inputs = self.input_layer_norm(field_embeds + depth_embeds)
                # cur_inputs = self.input_layer_norm(prod_embeds + field_embeds + depth_embeds)
                cur_inputs = self.input_layer_norm(action_embeds + prod_embeds + field_embeds + depth_embeds)
                prev_inputs = torch.cat([prev_inputs, cur_inputs.unsqueeze(1)], dim=1)
                cur_decoder_relations = torch.stack([beams[bid].hyps[hid].get_relation(fid, device) for bid in active_idx for hid, fid in enumerate(beams[bid].frontier_ids)], dim=0)
                root_relations = torch.tensor([[self.tranx.ast_relation.child_relation_mappings[rel_id] for rel_id in rels[:, 0].tolist()] for rels in cur_decoder_relations], dtype=torch.long).to(device)
                shift_decoder_relations = F.pad(cur_decoder_relations[:, :-1, :-1], (1, 0, 1, 0), value=self.tranx.ast_relation.relation2id['padding-padding'])
                shift_decoder_relations[:, 1:root_relations.size(0), 0] = root_relations[:, :-1]
                shift_decoder_relations[:, 0, 0] = self.tranx.ast_relation.relation2id['0-0']
                outputs = self.decoder_network(prev_inputs, cur_encodings, rel_ids=cur_shift_decoder_relations, enc_mask=cur_mask)[:, -1]
                # outputs = self.decoder_network(prev_inputs, cur_encodings, rel_ids=cur_decoder_relations, enc_mask=cur_mask)[:, -1]
                # prev_action_embeds = torch.cat([history_action_embeds, action_embeds.unsqueeze(1)], dim=1)
                # root_relations = torch.tensor([[self.tranx.ast_relation.child_relation_mappings[rel_id] for rel_id in rels[:, 0].tolist()] for rels in cur_decoder_relations], dtype=torch.long).to(device).unsqueeze(-1)
                # cur_node_cross_action_relations = torch.cat([root_relations, cur_decoder_relations[:, :, :-1]], dim=-1)
                # outputs = self.decoder_network(prev_inputs, prev_action_embeds, cur_encodings, rel_ids=cur_decoder_relations,
                    # shift_rel_ids=cur_node_cross_action_relations, enc_mask=cur_mask)[:, -1]
            else:
                cur_inputs = action_embeds + prod_embeds + field_embeds
                if t == 0: parent_states = encodings.new_zeros((num_examples, encodings.size(-1)))
                else:
                    parent_ts = torch.cat([beams[bid].get_parent_timesteps() for bid in active_idx])
                    parent_states = torch.gather(history_states, 1, parent_ts.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, history_states.size(-1))).squeeze(1)
                    parent_states = self.hidden2action(parent_states)
                cur_inputs = self.input_layer_norm(cur_inputs + parent_states)
                lstm_outputs, (h_t, c_t) = self.decoder_network(cur_inputs.unsqueeze(1), h_c, start=(t==0), prev_idx=prev_idx)
                lstm_outputs = lstm_outputs.squeeze(1)
                context = self.context_attn(lstm_outputs, cur_encodings, cur_mask)
                outputs = self.attention_linear(torch.cat([lstm_outputs, context], dim=-1))

            gate = self.switcher(torch.cat([outputs, cur_inputs], dim=-1))
            copy_token_prob = self.selector(outputs, cur_copy_memory, mask=cur_copy_mask) * gate
            gen_token_prob = self.generator(outputs, generator_memory, log=False) * (1 - gate)
            generate_token_prob = gen_token_prob.scatter_add_(1, cur_copy_ids, copy_token_prob)
            generate_token_logprob = torch.log(generate_token_prob + 1e-32)
            apply_rule_logprob = self.apply_rule(outputs, self.production_embed.weight)
            select_schema_logprob = torch.log(self.selector(outputs, cur_schema_memory, mask=cur_schema_mask) + 1e-32)

            # rank and select based on AST type constraints
            num_hyps = [len(beams[bid].hyps) for bid in active_idx]
            ar_scores, ss_scores, gt_scores = torch.split(apply_rule_logprob, num_hyps), torch.split(select_schema_logprob, num_hyps), torch.split(generate_token_logprob, num_hyps)
            new_active_idx, cum_num_hyps, live_hyp_ids = [], np.cumsum([0] + num_hyps), []
            for idx, bid in enumerate(active_idx):
                beams[bid].advance(ar_scores[idx], ss_scores[idx], gt_scores[idx])
                if not beams[bid].done:
                    new_active_idx.append(bid)
                    live_hyp_ids.extend(beams[bid].get_previous_hyp_ids(cum_num_hyps[idx]))

            if not new_active_idx: # all beams are finished
                break

            # update each unfinished beam and record active history infos
            active_idx = new_active_idx
            if args.decoder_cell == 'transformer':
                prev_inputs = prev_inputs[live_hyp_ids]
                # history_action_embeds = prev_action_embeds[live_hyp_ids]
            else: 
                h_c = (h_t[:, live_hyp_ids], c_t[:, live_hyp_ids])
                history_states = torch.cat([history_states[live_hyp_ids], h_c[0][-1].unsqueeze(1)], dim=1)
                prev_idx = prev_idx[live_hyp_ids]

        completed_hyps = [b.sort_finished() for b in beams]
        return completed_hyps
