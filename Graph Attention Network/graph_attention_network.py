import torch
from torch import nn
import math

class GraphAttentionLayer(nn.Module):
    def __init__(self, c_in, c_out, n_heads, is_concat, dropout, leaky_relu_negative_slope=0.2):
        super(GraphAttentionLayer, self).__init__()

        self.is_mut_head = is_concat
        self.n_heads = n_heads

        if self.is_mut_head:
            assert c_out % self.n_heads == 0
            self.n_hidden = c_out // self.n_heads
        else:
            self.n_hidden = c_out

        self.w = nn.Linear(c_in, self.n_heads * self.n_hidden, bias=False)
        self.attn = nn.Linear(2 * self.n_hidden, 1, bias=False)
        self.activation = nn.LeakyReLU(leaky_relu_negative_slope)
        self.softmax = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h, adj_mat):
        n_nodes = h.shape[0]

        g = self.w(h)
        # g=[g1,g2,...,gn]
        # g.shape=[n_nodes, n_heads, n_hidden]
        g = g.view(n_nodes, self.n_heads, self.n_hidden)

        # g_repeat=[g1,g2,...,gn,g1,g2,...,gn,...,g1,g2,...gn]
        # g_repeat.shape=[n_nodes * n_nodes, n_heads, n_hidden] 
        g_repeat = g.repeat(n_nodes, 1, 1)
        # g_repeat_interleave=[g1,g1,...g1,g2,g2,...,g2,...,gn,gn,...,gn]
        # g_repeat_interleave.shape=[n_nodes * n_nodes, n_heads, n_hidden] 
        g_repeat_interleave = g.repeat_interleave(n_nodes, dim=0)

        # g_concat=[g1||g1,g1||g2,...,g1||gn, 
        #           g2||g1,g2||g2,...,g2||gn,
        #           ...
        #           gn||g1,gn||g2,...,gn||gn]
        # g_concat.shape=[n_nodes * n_nodes, n_head, n_hidden * 2]
        # g_concat.view=[n_nodes, n_nodes, n_head, n_hidden * 2]
        g_concat = torch.concat([g_repeat_interleave, g_repeat], dim=-1)
        g_concat = g_concat.view(n_nodes, n_nodes, self.n_heads, self.n_hidden*2)

        # e.shape=[n_nodes, n_nodes, n_head, 1]
        # squeeze->[n_nodes, n_nodes, n_head]
        e = self.attn(g_concat)
        e = e.squeeze(-1)

        # 邻接矩阵adj_mat的shape为[n_nodes, n_nodes, n_heads]或[n_nodes, n_nodes, 1]
        assert adj_mat.shape[0] == 1 or adj_mat.shape[0] == n_nodes
        assert adj_mat.shape[1] == 1 or adj_mat.shape[1] == n_nodes
        assert adj_mat.shape[2] == 1 or adj_mat.shape[2] == self.n_heads

        # 使用邻接矩阵adj_mat做mask
        e = e.masked_fill(adj_mat == 0, float('-inf'))

        a = self.softmax(e)
        a = self.dropout(a)

        attn_res = torch.einsum('ijh,jhf->ihf', a, g)

        # 连接头部
        if self.is_mut_head:
            return attn_res.reshape(n_nodes, self.n_heads * self.n_hidden)
        else:
            return attn_res.mean(dim=1)