import torch

a = torch.randn(4, 4)

print(a)

print(torch.argmax(a, dim = 1))

print(torch.argmax(a, dim = 0))

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