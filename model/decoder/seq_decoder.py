#coding=utf8
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.model_utils import Registrable, PointerNetwork, TiedLinearClassifier, MultiHeadAttention, PositionalEncoding, tile
from model.decoder.decoder_cell import ONLSTM, LSTM
from model.decoder.seq_beam import SEQBeam


@Registrable.register('seq')
class SEQDecoder(nn.Module):

    def __init__(self, args, tranx):
        super(SEQDecoder, self).__init__()
        self.args, self.tranx = args, tranx
        if args.decoder_cell == 'transformer':
            self.pe = PositionalEncoding(args.embed_size)
            # assert args.decoder_hidden_size == args.encoder_hidden_size, "Transformer model should has the same hidden size for encoder and decoder."
            args.decoder_hidden_size = args.encoder_hidden_size
            decoder_layer = nn.TransformerDecoderLayer(args.decoder_hidden_size, args.num_heads, dim_feedforward=args.decoder_hidden_size * 4, dropout=args.dropout)
            self.decoder_network = nn.TransformerDecoder(decoder_layer, args.decoder_num_layers)
            self.input_affine = nn.Linear(args.embed_size, args.decoder_hidden_size) if args.embed_size != args.decoder_hidden_size else lambda x: x
        else:
            cell_constructor = ONLSTM if args.decoder_cell == 'onlstm' else LSTM
            self.decoder_network = cell_constructor(args.embed_size, args.decoder_hidden_size, num_layers=args.decoder_num_layers,
                dropout=args.dropout, dropconnect=args.dropout)
            self.context_attn = MultiHeadAttention(args.decoder_hidden_size, args.encoder_hidden_size, output_size=args.encoder_hidden_size,
                num_heads=args.num_heads, dropout=args.dropout)
            self.attention_linear = nn.Sequential(nn.Linear(args.decoder_hidden_size + args.encoder_hidden_size, args.decoder_hidden_size), nn.Tanh())
        # map schema items to the word embedding space as decoder input
        self.schema_affine = nn.Linear(args.encoder_hidden_size, args.embed_size)
        # generate tokens from vocabulary, copy raw tokens from input, or select schema items
        self.generator = TiedLinearClassifier(args.decoder_hidden_size, args.embed_size)
        if (not args.no_select_schema) or (not args.no_copy_mechanism):
            self.selector = PointerNetwork(args.decoder_hidden_size, args.encoder_hidden_size, num_heads=args.num_heads, dropout=args.dropout)
        self.gate_num = 3 - int(args.no_select_schema) - int(args.no_copy_mechanism)
        if self.gate_num > 1: self.switcher = nn.Linear(args.decoder_hidden_size + args.embed_size, 2 * self.gate_num - 3)
        self.loss_function = nn.NLLLoss(reduction='sum', ignore_index=self.tranx.tokenizer.pad_token_id)


    def score(self, memories, batch):
        """ Training function for token-based Sequence decoder.
        memories: dict stores (key, value) pairs, including `encodings` states (bs x src_len x dim), `schema` memory (bs x (table_num + column_num) x dim),
            `copy` memory (bs x copy_len x dim), `copy_ids` (map input to target vocab position, bs x copy_len), `generator` memory for token generation (vocab_size x dim),
            and masks `mask`, `schema_mask`, `copy_mask` for each memory.
        """
        args = self.args
        encodings, generator_memory, mask = memories['encodings'], memories['generator'], memories['mask']
        if not args.no_select_schema:
            max_schema_num = memories['schema'].size(1)
            schema_memory, schema_mask = memories['schema'], memories['schema_mask']
            schema_embed = self.schema_affine(schema_memory)
        if not args.no_copy_mechanism:
            copy_memory, copy_ids, copy_mask = memories['copy'], memories['copy_ids'], memories['copy_mask']

        # construct input matrices, bs x tgt_len x dim
        inp_actions, vocab_size = batch.seq_actions[:, :-1], self.tranx.tokenizer.vocab_size
        if args.no_select_schema: word_inputs = F.embedding(inp_actions, generator_memory)
        else:
            inp_schema_mask = inp_actions >= vocab_size # bs x tgt_len
            word_inputs = F.embedding(inp_actions.masked_fill(inp_schema_mask, 0), generator_memory) # bs x tgt_len x dim
            shift_schema_actions = inp_actions - vocab_size + max_schema_num * torch.arange(len(batch), device=inp_actions.device).unsqueeze(1)
            inp_schema_actions = shift_schema_actions.masked_select(inp_schema_mask)
            schema_inputs = schema_embed.contiguous().view(len(batch) * max_schema_num, args.embed_size)[inp_schema_actions]
            word_inputs = word_inputs.masked_scatter_(inp_schema_mask.unsqueeze(-1), schema_inputs)

        if args.decoder_cell == 'transformer':
            decoder_inputs = self.input_affine(self.pe(word_inputs)) # input is shifted
            subsequent_mask = torch.tril(mask.new_ones((inp_actions.size(1), inp_actions.size(1))))
            outputs = self.decoder_network(decoder_inputs.transpose(0, 1), encodings.transpose(0, 1), tgt_mask=~ subsequent_mask,
                tgt_key_padding_mask=~ batch.tgt_mask, memory_key_padding_mask=~ mask).transpose(0, 1) # bs x tgt_len x dim
        else:
            outputs, _ = self.decoder_network(word_inputs, start=True)
            context = self.context_attn(outputs, encodings, mask)
            outputs = self.attention_linear(torch.cat([outputs, context], dim=-1)) # bs x tgt_len x dim

        if self.gate_num > 1: # use copy mechanism or select schema items
            gate = self.switcher(torch.cat([outputs, word_inputs], dim=-1))
            gate = torch.sigmoid(gate) if self.gate_num == 2 else torch.softmax(gate, dim=-1)
            gen_token_prob = self.generator(outputs, generator_memory, log=False) * gate[:, :, 0:1] # bs x tgt_len x vocab_size
            if not args.no_copy_mechanism:
                copy_gate = 1 - gate if self.gate_num == 2 else gate[:, :, 1:2]
                copy_token_prob = self.selector(outputs, copy_memory, mask=copy_mask) * copy_gate # bs x tgt_len x max_copy_len
                copy_ids = copy_ids.unsqueeze(1).expand(-1, copy_token_prob.size(1), -1)
                gen_token_prob = gen_token_prob.scatter_add_(-1, copy_ids, copy_token_prob)
            if not args.no_select_schema:
                select_gate = 1 - gate if self.gate_num == 2 else gate[:, :, 2:3]
                select_schema_prob = self.selector(outputs, schema_memory, mask=schema_mask) * select_gate # bs x tgt_len x max_schema_num
                gen_token_prob = torch.cat([gen_token_prob, select_schema_prob], dim=-1) # bs x tgt_len x (vocab_size + max_schema_num)
            logprob = torch.log(gen_token_prob + 1e-32)
        else: logprob = self.generator(outputs, generator_memory)

        out_actions = batch.seq_actions[:, 1:].contiguous().view(-1)
        loss = self.loss_function(logprob.contiguous().view(out_actions.size(0), -1), out_actions)
        return loss


    def parse(self, memories, batch, beam_size=5, n_best=5, **kwargs):
        """ Decoding function for token-based Sequence decoder.
        memories: dict stores (key, value) pairs, including `encodings` states (bs x src_len x dim), `schema` memory (bs x (table_num + column_num) x dim),
            `copy` memory (bs x copy_len x dim), `copy_ids` (map input to target vocab position, bs x copy_len), `generator` memory for token generation (vocab_size x dim),
            and masks `mask`, `schema_mask`, `copy_mask` for each memory.
        """
        args, vocab_size = self.args, self.tranx.tokenizer.vocab_size
        # repeat input beam_size times
        encodings, mask = tile([memories['encodings'], memories['mask']], beam_size)
        generator_memory = memories['generator']
        if not args.no_select_schema:
            max_schema_num = memories['schema'].size(1)
            schema_memory, schema_mask = tile([memories['schema'], memories['schema_mask']], beam_size)
        if not args.no_copy_mechanism:
            copy_memory, copy_ids, copy_mask = tile([memories['copy'], memories['copy_ids'], memories['copy_mask']], beam_size)

        num_samples, batch_idx = len(batch), list(range(len(batch)))
        beams = [SEQBeam(self.tranx, db, beam_size, n_best, encodings.device) for db in batch.database]
        if args.decoder_cell == 'transformer': prev_inputs = encodings.new_zeros((num_samples * beam_size, 0, args.decoder_hidden_size))
        else: h_c, prev_idx = None, torch.arange(num_samples * beam_size, dtype=torch.long, device=encodings.device)

        for t in range(batch.max_action_num):
            # (a) construct inputs from remaining samples
            ys = torch.cat([b.get_current_state() for b in beams if not b.done], dim=0) # num_samples * beam_size
            ys_schema_mask = ys >= vocab_size # num_samples * beam_size
            if not args.no_select_schema and torch.any(ys_schema_mask).item():
                inputs = F.embedding(ys.masked_fill(ys_schema_mask, 0), generator_memory) # num_samples * beam_size x dim
                shift_schema_actions = ys - vocab_size + max_schema_num * torch.arange(num_samples * beam_size, device=encodings.device)
                inp_schema_actions = shift_schema_actions.masked_select(ys_schema_mask)
                schema_embed = self.schema_affine(schema_memory)
                schema_inputs = schema_embed.contiguous().view(-1, args.embed_size)[inp_schema_actions]
                inputs = inputs.masked_scatter_(ys_schema_mask.unsqueeze(-1), schema_inputs)
            else: inputs = F.embedding(ys, generator_memory) # num_samples * beam_size x dim

            # (b) calculate attention vectors over each hyp
            if args.decoder_cell == 'transformer':
                decoder_inputs = self.input_affine(self.pe(inputs.unsqueeze(1), timestep=t))
                prev_inputs = torch.cat([prev_inputs, decoder_inputs], dim=1) # num_hyps x tgt_len x embed_size
                subsequent_mask = torch.tril(mask.new_ones((t + 1, t + 1)))
                outputs = self.decoder_network(prev_inputs.transpose(0, 1), encodings.transpose(0, 1), tgt_mask=~ subsequent_mask,
                    memory_key_padding_mask=~ mask).transpose(0, 1)[:, -1] # num_hyps x hs
            else:
                outputs, (h_t, c_t) = self.decoder_network(inputs.unsqueeze(1), h_c, start=(t==0), prev_idx=prev_idx)
                outputs = outputs.squeeze(1)
                context = self.context_attn(outputs, encodings, mask)
                outputs = self.attention_linear(torch.cat([outputs, context], dim=-1)) # num_hyps x hs

            # (c) caluclate probability distribution
            if self.gate_num > 1:
                gate = self.switcher(torch.cat([outputs, inputs], dim=-1))
                gate = torch.sigmoid(gate) if self.gate_num == 2 else torch.softmax(gate, dim=-1)
                gen_token_prob = self.generator(outputs, generator_memory, log=False) * gate[:, 0:1] # num_hyps x vocab_size
                if not args.no_copy_mechanism:
                    copy_gate = 1 - gate if self.gate_num == 2 else gate[:, 1:2]
                    copy_token_prob = self.selector(outputs, copy_memory, mask=copy_mask) * copy_gate # num_hyps x copy_len
                    gen_token_prob = gen_token_prob.scatter_add_(-1, copy_ids, copy_token_prob)
                if not args.no_select_schema:
                    select_gate = 1 - gate if self.gate_num == 2 else gate[:, 2:3]
                    select_schema_prob = self.selector(outputs, schema_memory, mask=schema_mask) * select_gate # num_hyps x max_schema_num
                    gen_token_prob = torch.cat([gen_token_prob, select_schema_prob], dim=-1) # num_hyps x (vocab_size + max_schema_num)
                logprob = torch.log(gen_token_prob + 1e-32)
            else: logprob = self.generator(outputs, generator_memory)
            logprob = logprob.contiguous().view(num_samples, beam_size, -1)

            # (c) advance each beam
            active, select_indexes = [], []
            # Loop over the remaining_batch number of beam
            for b in range(num_samples):
                idx = batch_idx[b] # idx represent the original order in minibatch_size
                beams[idx].advance(logprob[b])
                if not beams[idx].done:
                    active.append((idx, b))
                select_indexes.append(beams[idx].get_current_origin() + b * beam_size)

            if not active:
                break

            # (d) update hidden_states history
            select_indexes = torch.cat(select_indexes, dim=0)
            if args.decoder_cell == 'transformer': prev_inputs = prev_inputs[select_indexes]
            else: h_c, prev_idx = (h_t[:, select_indexes], c_t[:, select_indexes]), prev_idx[select_indexes]

            # (e) reserve un-finished batches
            active_idx = torch.tensor([item[1] for item in active], dtype=torch.long, device=encodings.device) # original order in remaining batch
            batch_idx = { idx: item[0] for idx, item in enumerate(active) } # order for next remaining batch

            if len(active) < num_samples: # some samples are finished
                def update_active(inp, dim=0):
                    if dim != 0: inp = inp.transpose(0, dim)
                    inp_reshape = inp.contiguous().view(num_samples, beam_size, -1)[active_idx]
                    new_size = list(inp.size())
                    new_size[0] = -1
                    inp_reshape = inp_reshape.contiguous().view(*new_size)
                    if dim != 0: inp_reshape = inp_reshape.transpose(0, dim)
                    return inp_reshape

                encodings, mask = update_active(encodings), update_active(mask)
                if not args.no_copy_mechanism:
                    copy_memory, copy_mask, copy_ids = update_active(copy_memory), update_active(copy_mask), update_active(copy_ids)
                if not args.no_select_schema:
                    schema_memory, schema_mask = update_active(schema_memory), update_active(schema_mask)

                if args.decoder_cell == 'transformer': prev_inputs = update_active(prev_inputs)
                else: h_c, prev_idx = (update_active(h_c[0], dim=1), update_active(h_c[1], dim=1)), update_active(prev_idx)

            num_samples = len(active)

        completed_hyps = [b.sort_finished() for b in beams]
        return completed_hyps