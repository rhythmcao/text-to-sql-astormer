#coding=utf8
import torch.nn as nn
from model.model_utils import Registrable
from model.encoder.encoder_constructor import Encoder
from model.decoder.decoder_constructor import Decoder


@Registrable.register('text2sql')
class Text2SQL(nn.Module):

    def __init__(self, args, tranx):
        super(Text2SQL, self).__init__()
        self.encoder = Encoder(args, tranx)
        self.decoder = Decoder(args, tranx)


    def forward(self, batch):
        """ This function is used during training, which returns the training loss
        """
        return self.decoder(self.encoder(batch), batch)


    def parse(self, batch, beam_size=5, n_best=5, decode_order='dfs+l2r', **kwargs):
        """ This function is used for decoding, which returns a batch of hypothesis
        """
        return self.decoder.parse(self.encoder(batch), batch, beam_size=beam_size, n_best=n_best, decode_order=decode_order)