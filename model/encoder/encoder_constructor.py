#coding=utf8
import torch.nn as nn
from model.model_utils import Registrable
from model.encoder.input_layer import EncoderInputLayer
from model.encoder.hidden_layer import RGATHiddenLayer, IRNetHiddenLayer, NoneHiddenLayer
from model.encoder.output_layer import EncoderOutputLayer


class Encoder(nn.Module):

    def __init__(self, args, tranx):
        super(Encoder, self).__init__()
        self.input_layer = EncoderInputLayer(args, tranx)
        self.hidden_layer = Registrable.by_name(args.encode_method)(args, tranx)
        self.output_layer = EncoderOutputLayer(args, tranx)


    def forward(self, batch):
        outputs = self.input_layer(batch)
        outputs = self.hidden_layer(outputs, batch)
        if hasattr(self.input_layer, 'plm'):
            word_embed = self.input_layer.plm.embeddings.word_embeddings.weight
        else:
            word_embed = self.input_layer.swv.weight
        return self.output_layer(outputs, batch, word_embed)