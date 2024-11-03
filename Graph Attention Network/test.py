import torch

# a = torch.tensor([[[1, 3, 5],
#                    [7, 9, 11],
#                    [13, 15, 17]],
#                    [[0, 2, 4],
#                    [6, 8, 10],
#                    [12, 14, 16]]])
# print(a)
# print(a.shape)

# a = a.repeat(3, 1, 1)
# print(a)
# print(a.shape)

# b = a.repeat_interleave(3, dim=0)
# print(b)
# print(b.shape)

# c = a.repeat(3, 1, 1)
# print(c)
# print(c.shape)

# a_concat = torch.concat([b, c], dim=-1)
# print(a_concat)
# print(a_concat.shape)

a = torch.tensor([[1, 2, 3],
                 [4, 5, 6],
                 [7, 8, 9]])
print(a.shape)
a = a.masked_fill(a == 2, 0)

print(a)

print(torch.cuda.is_available())