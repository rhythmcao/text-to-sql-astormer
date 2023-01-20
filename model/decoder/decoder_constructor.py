#coding=utf8
import torch.nn as nn
from model.model_utils import Registrable
from model.decoder.ast_decoder import ASTDecoder
from model.decoder.seq_decoder import SEQDecoder


class Decoder(nn.Module):

    def __init__(self, args, tranx):
        super(Decoder, self).__init__()
        self.decoder = Registrable.by_name(args.decode_method)(args, tranx)


    def forward(self, memories, batch, **kwargs):
        return self.decoder.score(memories, batch, **kwargs)


    def parse(self, memories, batch, beam_size=5, n_best=5, decode_order='dfs+l2r', **kwargs):
        return self.decoder.parse(memories, batch, beam_size, n_best, decode_order=decode_order, **kwargs)