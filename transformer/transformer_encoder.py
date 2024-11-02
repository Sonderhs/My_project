import torch
from torch import nn
import math
import torch.nn.functional as F
from multi_head_attention import MultiheadAttention

# Embedding
# Token Embedding
class TokenEmbedding(nn.Embedding):
    def __init__(self, vocab_size, model_dim):
        super(TokenEmbedding, self).__init__(vocab_size, model_dim, padding_idx=1)


# Position Embedding
class PositionEmbedding(nn.Module):
    def __init__(self, model_dim, maxlen, device):
        super(PositionEmbedding, self).__init__()
        self.encodeing = torch.zeros(maxlen, model_dim, device=device)
        self.encodeing.requires_grad_(False)

        pos = torch.arange(0, maxlen, device=device)
        pos = pos.float().unsqueeze(1)
        _2i = torch.arange(0, model_dim, 2, device=device)

        # 偶数位置
        self.encodeing[:, 0::2] = torch.sin(pos / (10000 ** (_2i / model_dim)))
        # 奇数位置
        self.encodeing[:, 1::2] = torch.cos(pos / (10000 ** (_2i / model_dim)))

    def forward(self, x):
        seq_len = x.shape[1]
        return self.encodeing[:seq_len, :]


# Layer Norm
# 过程：求出均值mu和方差sigma，
# 然后out=(x-mean)/sqrt(sigma+eps)之后进行偏移y=gamma*x+beta
class LayerNorm(nn.Module):
    def __init__(self, model_dim, eps=1e-10):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(model_dim))
        self.beta = nn.Parameter(torch.zeros(model_dim))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, unbiased=False, keepdim=True)
        out = (x - mean) / torch.sqrt(var + self.eps)
        out = self.gamma * out + self.beta
        return out


# FFN:Position-Wise Fully Connected Feed-Forward Network
class PositionwiseFeedForward(nn.Module):
    def __init__(self, model_dim, hidden, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(model_dim, hidden)
        self.fc2 = nn.Linear(hidden, model_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# TransformerEmbedding
class TransformerEmbedding(nn.Module):
    def __init__(self, model_dim, vocab_size, maxlen, drop_prob, device):
        super(TransformerEmbedding, self).__init__()
        self.tok_emb = TokenEmbedding(vocab_size, model_dim)
        self.pos_emb = PositionEmbedding(model_dim, maxlen, device=device)
        self.drop_out = nn.Dropout(p=drop_prob)

    def forward(self, x):
        tok_emb = self.tok_emb(x)
        pos_emb = self.pos_emb(x)
        out = self.drop_out(tok_emb + pos_emb)
        return out


# EncoderLayer
class EncoderLayer(nn.Module):
    def __init__(self, model_dim, num_head, hidden, drop_prob):
        super(EncoderLayer, self).__init__()
        self.attention = MultiheadAttention(model_dim, num_head)
        self.norm1 = LayerNorm(model_dim)
        self.drop1 = nn.Dropout(drop_prob)

        self.ffn = PositionwiseFeedForward(model_dim, hidden, drop_prob)
        self.norm2 = LayerNorm(model_dim)
        self.drop2 = nn.Dropout(drop_prob)

    def forward(self, x, mask=None):
        _x = x
        x = self.attention(x, x, x, mask)
        
        x = self.drop1(x)
        x = self.norm1(x + _x)

        _x = x
        x = self.ffn(x)
        x = self.drop2(x)
        x = self.norm2(x + _x)
        return x
        
