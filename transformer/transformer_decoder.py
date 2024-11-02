import torch
from torch import nn
import math
import torch.nn.functional as F
from multi_head_attention import MultiheadAttention
from transformer_encoder import *


# DecoderLayer
class DecoderLayer(nn.Module):
    def __init__(self, model_dim, hidden, num_head, drop_prob):
        super(DecoderLayer, self).__init__()
        self.attention1 = MultiheadAttention(model_dim, num_head)
        self.norm1 = LayerNorm(model_dim)
        self.drop1 = nn.Dropout(drop_prob)

        self.cross_attention = MultiheadAttention(model_dim, num_head)
        self.norm2 = LayerNorm(model_dim)
        self.drop2 = nn.Dropout(drop_prob)

        self.ffn = PositionwiseFeedForward(model_dim, hidden, drop_prob)
        self.norm3 = LayerNorm(model_dim)
        self.drop3 = nn.Dropout(drop_prob)

    def forward(self, dec, enc, t_mask, s_mask):
        _x = dec
        x = self.attention1(dec, dec, dec, t_mask)  # 下三角掩码

        x = self.drop1(x)
        x = self.norm1(x + _x)

        if enc is not None:
            _x = x
            x = self.cross_attention(x, enc, enc, s_mask)

            x = self.drop2(x)
            x = self.norm2(x + _x)
        
        _x = x
        x = self.ffn(x)

        x = self.drop3(x)
        x = self.norm3(x + _x)
        return x


# 集成
class Encoder(nn.Module):
    def __init__(self, model_dim, enc_voc_size, maxlen, hidden, num_head, n_layer, drop_prob, device):
        super(Encoder, self).__init__()

        self.embedding = TransformerEmbedding(model_dim, enc_voc_size, maxlen, drop_prob, device=device)

        self.layers = nn.ModuleList(
            [EncoderLayer(model_dim, num_head, hidden, drop_prob) for _ in range(n_layer)]
        )

    def forward(self, x, s_mask):
        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x, s_mask)
        return x


class Decoder(nn.Module):
    def __init__(self, model_dim, dec_voc_size, maxlen, hidden, num_head, n_layer, drop_prob, device):
        super(Decoder, self).__init__()

        self.embedding = TransformerEmbedding(model_dim, dec_voc_size, maxlen, drop_prob, device=device)

        self.layers = nn.ModuleList(
            [DecoderLayer(model_dim, hidden, num_head, drop_prob) for _ in range(n_layer)]
        )

        self.fc = nn.Linear(model_dim, dec_voc_size)

    def forward(self, dec, enc, t_mask, s_mask):
        dec = self.embedding(dec)
        for layer in self.layers:
            dec = layer(dec, enc, t_mask, s_mask)
        
        dec = self.fc(dec)
        return dec