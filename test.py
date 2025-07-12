import clownpiece as CP
from clownpiece import Tensor
from clownpiece.nn import Module, Linear, Tanh, Sigmoid, ReLU, LeakyReLU, Sequential
from clownpiece.nn import MSELoss, CrossEntropyLoss
from clownpiece.autograd import no_grad
import math
# Define a custom network by subclassing nn.Module
class SimpleNet(Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        # Define layers as attributes. They are automatically registered.
        self.layer1 = Linear(input_size, hidden_size)
        self.activation = ReLU()
        self.layer2 = Linear(hidden_size, output_size)

    # Define the forward pass
    def forward(self, x):
        x = self.layer1(x)
        x = self.activation(x)
        x = self.layer2(x)
        return x

# Instantiate the model
model = SimpleNet(input_size=784, hidden_size=128, output_size=10)
print(model)

# The module system makes it easy to inspect all parameters
print("\nNamed Parameters:")
for name, param in model.named_parameters():
    print(f"{name}: {param.shape}")

# a = 24
# b = [2, 5]
# print(list([a]), list(b))

# import torch

# a = torch.tensor([[2.0, 3.0], [3.0, 4.0]], requires_grad=True)
# # # a = torch.tensor([[[1, 3, 4], [2, 6, 7]], [[1, 7, 9], [3, 8, 1]], [[11, 17, 19], [13, 81, 111]]])

# b = a.sum(dim = None, keepdims = True)
# # b.backward(torch.ones_like(b))
# print(b)
# print(a.grad)

# print(a)
# print(a.shape)
# index = torch.tensor([])
# # index = torch.tensor([[[0, 1, 2], [2, 1, 0]], [[1, 0, 2], [2, 0, 1]]])
# # print(index)
# # print(index.shape)
# src = torch.tensor([[[-1, -2, -3], [-4, -5, -6]], [[-7, -8, -9], [-10, -11, -12]]])
# # print(src)
# # print(src.shape)
# print(a.scatter_(0, index, src))

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