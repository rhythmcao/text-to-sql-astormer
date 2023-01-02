#coding=utf8
import torch.nn as nn
from model.model_utils import PoolingFunction


class EncoderOutputLayer(nn.Module):

    def __init__(self, args, tranx):
        super(EncoderOutputLayer, self).__init__()
        self.encode_method = args.encode_method
        if self.encode_method == 'none':
            self.schema_aggregation = PoolingFunction(args.encoder_hidden_size, args.encoder_hidden_size)


    def forward(self, inputs, batch, word_embed):
        """ Construct memories and masks for the decoder, including copy_memory and schema_memory.
        @args:
            inputs~(torch.FloatTensor): bs x max_len x hs
            batch: we use the following fields, e.g.,
                select_schema_mask, to extract the position of schema items from encoder output
                schema_token_mask, to aggregate multi-token schema items into one vector
                schema_mask, mask of schema items indicating the number of schema items for each training sample
                select_copy_mask, to extract the position of copy tokens (mainly question tokens)
                copy_mask, to re-allocate space for copy tokens
                copy_ids, to map the copy position to word vocab id
            word_embed~(torch.FloatTensor): encoder word embedding module, vocab_size x embedding_size
        @return:
            memories~(Dict[key, torch.Tensor])
        """
        memories = {'encodings': inputs, 'mask': batch.mask}
        schema_inputs = inputs.masked_select(batch.select_schema_mask.unsqueeze(-1))
        if self.encode_method == 'none':
            schema_inputs = inputs.new_zeros((batch.schema_token_mask.size(0), batch.schema_token_mask.size(1), inputs.size(-1))).masked_scatter_(batch.schema_token_mask.unsqueeze(-1), schema_inputs)
            schema_inputs = self.schema_aggregation(schema_inputs, batch.schema_token_mask)
        schema_memory = inputs.new_zeros((batch.schema_mask.size(0), batch.schema_mask.size(1), inputs.size(-1))).masked_scatter_(batch.schema_mask.unsqueeze(-1), schema_inputs)

        copy_inputs = inputs.masked_select(batch.select_copy_mask.unsqueeze(-1))
        copy_memory = inputs.new_zeros((batch.copy_mask.size(0), batch.copy_mask.size(1), inputs.size(-1))).masked_scatter_(batch.copy_mask.unsqueeze(-1), copy_inputs)

        memories['schema'], memories['copy'], memories['generator'] = schema_memory, copy_memory, word_embed
        memories['schema_mask'], memories['copy_mask'], memories['copy_ids'] = batch.schema_mask, batch.copy_mask, batch.copy_ids
        return memories