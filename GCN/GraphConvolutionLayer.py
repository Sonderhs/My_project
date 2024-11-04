import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parameter as Parameter

# GCNLayer
class GCNConv(nn.Module):
    def __init__(self, in_features, out_features):
        super(GCNConv, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        # Parameter: 将一个不可训练的类型Tensor转换成可以训练的类型parameter
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.bias = nn.Parameter(torch.FloatTensor(out_features))
        self.reset_parameter()

    def reset_parameter(self):
        # 使用均匀分布初始化权重
        nn.init.xavier_uniform_(self.weight)
        # 使用0初始化偏置
        nn.init.zeros_(self.bias)

    def forward(self, x, edge_index):
        # x：节点特征，大小为[num_nodes, in_features]
        # edge_index：边的索引，大小为[2, num_edges]
        
        # 计算WX，x: 输入特征
        x = torch.matmul(x, self.weight)
        
        # 计算邻接矩阵
        # 初始化邻接矩阵形状的全0张量
        num_nodes = x.size(0)
        adj = torch.zeros(num_nodes, num_nodes).cuda()
        # 根据edge_index填充邻接矩阵
        adj[edge_index[0], edge_index[1]] = 1
        adj[edge_index[1], edge_index[0]] = 1

        # 计算度矩阵D
        deg = torch.sum(adj, dim=1)
        deg_inv_sqrt = 1.0 / torch.sqrt(deg)

        # 计算D^-0.5AD^-0.5
        norm_adj = deg_inv_sqrt.unsqueeze(1)
        norm_adj = norm_adj * deg_inv_sqrt.unsqueeze(0)
        print(norm_adj.shape)
        print(deg_inv_sqrt.shape)
        print(deg_inv_sqrt.unsqueeze(1).shape)
        print(deg_inv_sqrt.unsqueeze(0).shape)
        # 计算最终输出
        output = torch.matmul(norm_adj, x) + self.bias
        return output