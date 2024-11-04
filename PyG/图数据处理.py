import torch
from torch_geometric import datasets
from torch_geometric.data import Data

# 数据属性
# 图用于对对象（节点）之间的关系（边）进行建模。PyG中的单个图形由torch_geometric.data.Data对象表示，该对象包含以下属性：
# data.x：图节点的属性信息，节点特征矩阵，维度为[num_nodes, num_node_features]
# data.edge_index： COO格式的图的边信息，类型为torch.long，维度为[2, num_edges]，具体包含两个列表，每个列表对应位置上的数字表示相应节点之间存在边连接
#          COO格式：COO格式是一种常用的稀疏矩阵存储格式，其主要包含两个列表，分别存储行索引和列索引，表示矩阵中非零元素的位置
#                  例如，对于一个3*3的矩阵[[1, 0, 2], 
#                                        [0, 3, 0], 
#                                        [4, 0, 5]]，
#                  其COO格式为：row = [0, 0, 1, 2, 2]
#                              col = [0, 2, 1, 0, 2]
#                  其中，row和col分别表示非零元素的行索引和列索引
# data.edge_attr：图的边属性信息，维度为[num_edges, num_edge_features]
# data.y：图的标签信息，维度为[num_nodes, num_node_labels]，用于节点分类任务，如果在整个图上的分类任务，则为[1, num_graph_labels]
# data.pos：节点的位置信息，维度为[num_nodes, num_dimensions]

# 图的数据处理
# 我们使用的数据是一个简单的无向图，有三个节点0,1,2，两条边(0,1)和(1,2)。每个节点有一个特征，特征维度为1，每条边有一个特征，特征维度为1。
edge_index = torch.tensor([
                            [0, 1, 1, 2],
                            [1, 0, 2, 1]], dtype=torch.long)  # 边信息
x = torch.tensor([[-1, 0], 
                  [0, 0], 
                  [1, 0]], dtype=torch.float)  # 节点特征
edge_attr = torch.tensor([[1], [1], [2], [2]])  # 边特征

data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

data.validate(raise_on_error=True)  # 验证数据是否有效
print(data)

print(data.keys)  # 获取数据的属性名称

print(data['x'])  # 获取节点特征

print(data.x)  # 获取节点特征

for key, item in data:
    print(f'{key} found in data:{item}')

print(data.num_nodes)  # 获取节点数量
print(data.num_edges)  # 获取边数量
print(data.num_node_features)  # 获取节点特征的维度
print(data.has_isolated_nodes())  # 判断是否有孤立节点
print(data.has_self_loops())  # 判断是否有自环
print(data.is_directed())  # 判断是否是有向图

