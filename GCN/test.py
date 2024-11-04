import torch

a = torch.tensor([[1], 
                 [2], 
                 [3]])   
b = torch.tensor([[4, 5, 6]])

print(a.shape)
print(b.shape)
c = a + b
print(c)