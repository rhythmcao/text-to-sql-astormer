#coding=utf8
import os
import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
from model.model_utils import rnn_wrapper
from nsts.transition_system import CONFIG_PATHS


class EncoderSWVInputLayer(nn.Module):

    def __init__(self, args, tranx):
        super(EncoderSWVInputLayer, self).__init__()
        plm = os.path.join(CONFIG_PATHS['plm_dir'], args.plm)
        config = AutoConfig.from_pretrained(plm)
        args.embed_size = config.embedding_size if hasattr(config, 'embedding_size') else config.hidden_size
        self.swv = AutoModel.from_config(config).embeddings.word_embeddings if getattr(args, 'lazy_load', False) else AutoModel.from_pretrained(plm).embeddings.word_embeddings
        self.question_rnn = nn.LSTM(args.embed_size, args.encoder_hidden_size // 2, num_layers=1, bidirectional=True, batch_first=True)
        self.schema_rnn = nn.LSTM(args.embed_size, args.encoder_hidden_size // 2, num_layers=1, bidirectional=True, batch_first=True)
        self.dropout_layer = nn.Dropout(p=args.dropout)


    def forward(self, batch):
        questions, schemas = self.swv(batch.inputs["question_ids"]), self.swv(batch.input_ids["schema_ids"])
        questions, _ = rnn_wrapper(self.question_rnn, self.dropout_layer(questions), batch.question_lens)
        questions = questions.view(-1, questions.size(-1))[batch.question_mask.view(-1)]
        _, hiddens = rnn_wrapper(self.schema_rnn, self.dropout_layer(schemas), batch.schema_token_lens)
        schemas = hiddens[0].transpose(0, 1).contiguous().view(-1, questions.size(-1))
        outputs = []
        for q, s in zip(questions.split(batch.question_lens.tolist(), 0), schemas.split(batch.schema_lens.tolist(), 0)):
            padding = q.new_zeros((batch.mask.size(1) - q.size(0) - s.size(0), q.size(-1)))
            outputs.append(torch.cat([q, s, padding], dim=0))
        outputs = torch.stack(outputs, dim=0) # bs x max_len x hs
        return outputs


class EncoderInputLayer(nn.Module):

    def __init__(self, args, tranx):
        super(EncoderInputLayer, self).__init__()
        plm = os.path.join(CONFIG_PATHS['plm_dir'], args.plm)
        config = AutoConfig.from_pretrained(plm)
        args.embed_size = config.embedding_size if hasattr(config, 'embedding_size') else config.hidden_size
        self.plm = AutoModel.from_config(config) if getattr(args, 'lazy_load', False) else AutoModel.from_pretrained(plm)
        self.encode_method = args.encode_method
        if self.encode_method != 'none':
            # use RNN to encode context information and compress schema tokens into a single node
            self.question_rnn = nn.LSTM(config.hidden_size, args.encoder_hidden_size // 2, num_layers=1, bidirectional=True, batch_first=True)
            self.schema_rnn = nn.LSTM(config.hidden_size, args.encoder_hidden_size // 2, num_layers=1, bidirectional=True, batch_first=True)
            self.dropout_layer = nn.Dropout(p=args.dropout)
        else: args.encoder_hidden_size = config.hidden_size


    def forward(self, batch):
        """
        @args:
            batch: we use the following field, e.g.,
                plm_question_mask and plm_schema_mask, to extract representation from PLM outputs
                question_mask and question_lens, to further encode question via RNN
                schema_token_mask and schema_token_lens, to further encode each schema item via RNN
                question_lens and schema_lens, to re-organize the entire inputs, including question tokens and schema items
        @retrun:
            outputs: torch.FloatTensor, encoded representation of question tokens and schema items
        """
        outputs = self.plm(**batch.inputs)[0]

        if self.encode_method != 'none':
            # further encode PLM outputs with RNN, question tokens obtain contextual information, schema items are each compressed into one single node
            plm_question_outputs = outputs.masked_select(batch.plm_question_mask.unsqueeze(-1))
            question_outputs = outputs.new_zeros((batch.question_mask.size(0), batch.question_mask.size(1), outputs.size(-1))).masked_scatter_(batch.question_mask.unsqueeze(-1), plm_question_outputs)
            question_outputs, _ = rnn_wrapper(self.question_rnn, self.dropout_layer(question_outputs), batch.question_lens)
            question_outputs = question_outputs.view(-1, question_outputs.size(-1))[batch.question_mask.view(-1)]

            plm_schema_outputs = outputs.masked_select(batch.plm_schema_mask.unsqueeze(-1))
            schema_outputs = outputs.new_zeros((batch.schema_token_mask.size(0), batch.schema_token_mask.size(1), outputs.size(-1))).masked_scatter_(batch.schema_token_mask.unsqueeze(-1), plm_schema_outputs)
            _, hiddens = rnn_wrapper(self.schema_rnn, self.dropout_layer(schema_outputs), batch.schema_token_lens)
            schema_outputs = hiddens[0].transpose(0, 1).contiguous().view(-1, question_outputs.size(-1))

            outputs = []
            for q, s in zip(question_outputs.split(batch.question_lens.tolist(), 0), schema_outputs.split(batch.schema_lens.tolist(), 0)):
                padding = q.new_zeros((batch.mask.size(1) - q.size(0) - s.size(0), q.size(-1)))
                outputs.append(torch.cat([q, s, padding], dim=0))
            outputs = torch.stack(outputs, dim=0) # bs x max_len x hs
        return outputs
