import torch
from torch_geometric.datasets import TUDataset

# TUDataset小行星数据集
# TUDataset是PyG中用于加载图分类数据集的类，其主要参数有：
# name：数据集名称
# root：数据集存储路径
# use_node_attr：是否使用节点属性
# use_edge_attr：是否使用边属性
# use_node_label：是否使用节点标签
# split：数据集划分方式
# pre_transform：数据集预处理方法
# transform：数据集处理方法
# pre_filter：数据集过滤方法
# filter：数据集过滤方法

# # 加载 MUTAG 数据集（可以换成其他支持的图数据集名称）
# dataset = TUDataset(root="/tmp/ENZYMES", name="ENZYMES")

# print(dataset)

# print(len(dataset))  # 数据集大小
# print(dataset.num_classes)  # 类别数量
# print(dataset.num_node_features)  # 节点特征维度
# print(dataset._data)  # 数据集中的图数据
# print(dataset[0])  # 获取第一个图
# print(dataset.edge_index.shape)  # 获取第一个图的边信息

# print(dataset[0].is_undirected())  # 判断图是否是无向图

# train_dataset = dataset[:540]  # 划分训练集
# print(train_dataset)

# test_dataset = dataset[540:]  # 划分测试集
# print(test_dataset)

# dataset = dataset.shuffle()  # 打乱数据集
# print(dataset[0])

# 半监督简单示例
from torch_geometric.datasets import Planetoid
dataset = Planetoid(root="/tmp/Cora", name="Cora")
print(dataset)

print(len(dataset))  # 数据集大小
print(dataset.num_classes)  # 类别数量
print(dataset.num_node_features)  # 节点特征维度
print(dataset._data)  # 数据集中的图数据
print(dataset[0])  # 获取第一个图
