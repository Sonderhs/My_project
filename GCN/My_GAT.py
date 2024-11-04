import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_dense_adj

# GATLayer
class GATConv(nn.Module):
    def __init__(self, in_features, out_features, dropout=0.2, alpha=0.2, concat=True):
        super(GATConv, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        # dropout rate
        self.dropout = dropout
        # LeakReLu层参数
        self.alpha = alpha
        # 是否拼接
        self.concat = concat

        # 初始化W权重矩阵
        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        # 初始化a权重矩阵
        self.a = nn.Parameter(torch.zeros(size=(2* out_features, 1)))
        # xavier 初始化方法中服从均匀分布 U(−a,a) ，分布的参数 a = gain * sqrt(6/fan_in+fan_out)，
        # 其中 gain 是默认为 1 的增益因子，fan_in 是权重张量的输入单元数，fan_out 是权重张量的输出单元数。
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        nn.init.xavier_uniform_(self.a.data, gain= 1.414)

        # 定义LeakyReLU激活函数
        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, edge_index):
        # 使用to_dense_adj函数将edge_indx转化为邻接矩阵adj
        adj = to_dense_adj(edge_index).squeeze(0)

        N = h.size()[0]  # N 图的节点数
        Wh = torch.mm(h, self.W)
        # wh = torch.cat([wh.repeat(1, N).view(N * N, -1), wh.repeat(N, 1)], dim=1).view(N, -1, 2 * self.out_features)

        e = self._prepare_attentional_mechanism_input(Wh)
        # e = torch.matmul(wh, self.a).squeeze(2)
        # e = F.leaky_relu(e, self.alpha)

        # mask
        e = e.masked_fill(adj == 0, float('-inf'))

        # softmax
        a = F.softmax(e, dim=1)
        # dropout
        a = F.dropout(a, self.dropout, training=self.training)

        output = torch.matmul(a, Wh)
        return output

    def _prepare_attentional_mechanism_input(self, Wh):
        # 这个函数负责计算注意力系数
        # 首先通过与a的前半部分做矩阵乘法计算得到每个节点的影响力分数Wh1
        Wh1 = torch.matmul(Wh, self.a[:self.out_features, :])
        # 通过与a的后半部分做矩阵乘法计算得到每个节点被影响的分数Wh2
        Wh2 = torch.matmul(Wh, self.a[self.out_features:, :])
        # 将Wh1加上Wh2的转置，得到每一对节点的非归一化注意力分数e

        # Wh1: [N, 1], 
        # Wh2.T: [1, N]
        # e: [N, N]
        e = Wh1 + Wh2.T
        i = 0
        if i == 0:
            print(Wh1, Wh2.T, e)
            i = 1
        # 使用LeakyReLU激活函数处理e，增加非线性
        return self.leakyrelu(e)