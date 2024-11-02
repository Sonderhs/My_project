import torch
from torch import nn
import math
import torch.nn.functional as F

# # 定义参数
# x = torch.rand(128, 64, 512)  # batch, seq_len, dimension
# print(x.shape)
# model_dim = 512
# num_head = 8


class MultiheadAttention(nn.Module):
    def __init__(self, model_dim, num_head) -> None:
        super(MultiheadAttention, self).__init__()
        
        self.model_dim = model_dim  # 初始化模型维数
        self.num_head = num_head  # 初始化头个数
        # 初始化w_q,W_k,w_v矩阵
        self.w_q = nn.Linear(model_dim, model_dim)
        self.w_k = nn.Linear(model_dim, model_dim)
        self.w_v = nn.Linear(model_dim, model_dim)
        # 初始化输出矩阵
        self.o = nn.Linear(model_dim, model_dim)
        # 定义softmax层
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v, mask=None):
        # q的形状为[batch_size,seq_len,dimension]
        # batch_size：批量大小
        # seq_len：序列长度
        # dimension：每个序列元素的嵌入维度
        batch, seq_len, dimension = q.shape
        head_dim = self.model_dim // self.num_head  # 计算头的dimension

        # 首先让q,k,v经过线性层投影
        Q = self.w_q(q)
        K = self.w_k(k)
        V = self.w_v(v)

        # 分割q,k,v为多头
        Q = Q.view(batch, seq_len, self.num_head, head_dim).permute(0, 2, 1, 3)
        K = K.view(batch, seq_len, self.num_head, head_dim).permute(0, 2, 1, 3)
        V = V.view(batch, seq_len, self.num_head, head_dim).permute(0, 2, 1, 3)

        # 计算注意力得分score=Q@K^T/sqrt(head_d)
        scores = Q @ K.transpose(2, 3) / math.sqrt(head_dim)
        if mask is not None:
            # mask = torch.tril(torch.ones(seq_len, seq_len, dtype=bool))
            scores = scores.masked_fill(mask == 0, -1e9)
        # 再通过softmax
        scores = self.softmax(scores) @ V

        # 拼接所有的头
        # scores形状：[batch_size, num_head, seq_len, seq_len]
        scores = scores.permute(0, 2, 1, 3).contiguous().view(batch, seq_len, dimension)

        # 通过最后的线性层得到输出
        output = self.o(scores)
        return output

# attention = MultiheadAttention(model_dim, num_head)
# output = attention(x, x, x)
# print(output, output.shape)
