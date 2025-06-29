import torch

a = torch.tensor([[[1, 3, 4], [2, 6, 7]], [[1, 7, 9], [3, 8, 1]], [[11, 17, 19], [13, 81, 111]]])
# print(a)
# print(a.shape)
index = torch.tensor([])
# index = torch.tensor([[[0, 1, 2], [2, 1, 0]], [[1, 0, 2], [2, 0, 1]]])
# print(index)
# print(index.shape)
src = torch.tensor([[[-1, -2, -3], [-4, -5, -6]], [[-7, -8, -9], [-10, -11, -12]]])
# print(src)
# print(src.shape)
print(a.scatter_(0, index, src))

#tensor([[[ -1,  -8,   4],
#         [  2, -11,  -6]],
#
#        [[ -7,  -2,   9],
#         [  3,  -5, -12]],
#
#        [[ 11,  17,  -9],
#         [-10,  81, 111]]])

# print(a)

# print(torch.range(1, 1, 0))

# a = torch.Tensor([[0, 0], [1, 1]])

# print(a.sum(0, keepdims = True))

# vector x vector
# tensor1 = torch.randn(3)
# tensor2 = torch.randn(3)
# print(torch.matmul(tensor1, tensor2).size())
# print(tensor1)
# print(tensor2)
# print(torch.matmul(tensor1, tensor2))
# # matrix x vector
# tensor1 = torch.randn(3, 4)
# tensor2 = torch.randn(4)
# print(torch.matmul(tensor1, tensor2).size())
# # batched matrix x broadcasted vector
# tensor1 = torch.randn(10, 3, 4)
# tensor2 = torch.randn(4)
# print(torch.matmul(tensor1, tensor2).size())
# # batched matrix x batched matrix
# tensor1 = torch.randn(10, 3, 4)
# tensor2 = torch.randn(10, 4, 5)
# print(torch.matmul(tensor1, tensor2).size())
# # batched matrix x broadcasted matrix
# tensor1 = torch.randn(10, 3, 4)
# tensor2 = torch.randn(4, 5)
# print(torch.matmul(tensor1, tensor2).size())
# # test non

# tensor1 = torch.randn(1, 1)
# # tensor2 = torch.tensor([])
# print(tensor1)
# tensor2 = torch.tensor([])
# print(tensor2)
# print(torch.matmul(tensor1, tensor2))
# print(torch.matmul(tensor1, tensor2).size())