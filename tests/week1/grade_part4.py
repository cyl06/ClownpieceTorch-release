import sys
import os


import torch
from graderlib import self_path
from graderlib import set_debug_mode, testcase, grader_summary



@testcase(name="basic_calc1: Binary Operations", score=10)

def basic_calc1(impl = torch):
    a = impl.Tensor([[1, -2, 3], [4, 5, -6]])
    b = impl.Tensor([[7, 8, -9], [10, -11, -12]])
    c = impl.Tensor([1, 2, 3, 4, 5, 6])
    d = impl.Tensor([-1, -2, -3, -4, -5, -6])
    
    return (a + b, a - b, a * b, a / b, impl.dot(c, d))
    
@testcase(name="basic_calc2: Unary Operations1", score=10)

def basic_calc2(impl = torch):
    a = impl.Tensor([[1, -2, 3], [4, 5, -6]])
    b = impl.Tensor([[7, 8, -9], [10, -11, -12]])
    
    return (-a, a.abs(), abs(a), 
            a.sin(), impl.sin(a), 
            a.cos(), impl.cos(a),
            a.tanh(), impl.tanh(a))
    
@testcase(name="basic_calc3: Unary Operations2", score=10)

def basic_calc3(impl = torch):
    base = 2
    a = impl.Tensor([[1, -2, 3], [4, 5, -6]])
    b = impl.Tensor([[7, 8, -9], [10, -11, -12]])
    
    return (a.exp(), impl.exp(a), a.log(), impl.log(a), 
            a.sqrt(), impl.sqrt(a),
            a.pow(2), impl.pow(a, 2),
            a.clamp(2, 5), impl.clamp(a, 2, 5))

@testcase(name="basic_calc4: Comparison Operations", score=10)

def basic_calc4(impl = torch):
    a = impl.Tensor([[1, -2, 3], [4, 5, -6]])
    b = impl.Tensor([[7, -2, -9], [10, 6, -12]])
    c = 3.0

    if impl.__name__ == "torch":
        return ((a == b).to(torch.int), (a != b).to(torch.int),
                (a < b).to(torch.int), (a <= b).to(torch.int),
                (a > b).to(torch.int), (a >= b).to(torch.int),
                (a > c).to(torch.int), (a < c).to(torch.int),
                (a >= c).to(torch.int), (a <= c).to(torch.int))
    else:
        return (a == b, a != b, a < b, a <= b, a > b, a >= b,
                a > c, a < c, a >= c, a <= c)
    

@testcase(name="broadcast1: Scalar Boardcasting", score=10)

def broadcast1(impl = torch):
    a = impl.Tensor([[1, -2, 3], [4, 5, -6]])
    if impl.__name__ == "torch":
        b = impl.tensor(2)
    else:
        b = impl.Tensor(2)

    return (a + 2, a - 2, a * 2, a / 2,
            2 + a, 2 - a, 2 * a, 2 / a,
            a + b, a - b, 
            a * b, a / b)
    
@testcase(name="broadcast2: Vector Boardcasting", score=10)

def broadcast2(impl = torch):
    a = impl.Tensor([[1, -2, 3], [4, 5, -6]])
    b = impl.Tensor([[7, 8, -9], [10, -11, -12]])
    
    c = impl.zeros((3, 4, 6))
    d = impl.Tensor([1, 1, 4, 5, 1, 4])
    
    e = impl.zeros((2, 3, 2))
    f = impl.zeros((2, 1, 2))
    f[0][0][0] = 1
    f[0][0][1] = 2
    f[1][0][0] = 3
    f[1][0][1] = 4
    
    g = impl.zeros((3, 5, 7))
    h = impl.zeros((1, 7))
    
    i = impl.zeros((4, 1, 6))
    j = impl.zeros((2, 4, 5, 6))
    
    return (a + b[0], a - b[1], a * b[0], a / b[1], 
            c + d, e + f, g + h, i + j)

@testcase(name="broadcast3: Invailid Vector Boardcasting", score=10)

def broadcast3(impl = torch):
    a = impl.zeros((2, 3))
    b = impl.zeros((3, 3))
    
    if impl.__name__ == "torch":
        return True
    
    else:
        try:
            return a + b
        except RuntimeError as e:
            pass
        
        try:
            c = impl.zeros((2, 5, 4))
            d = impl.zeros((2, 4))
            return c + d
        except RuntimeError as e:
            return True
                

def testsets_part4():
    basic_calc1()
    basic_calc2()
    basic_calc3()
    basic_calc4()
    
    broadcast1()
    broadcast2()
    broadcast3()

if __name__ == "__main__":
  print("Beginning grading part 4")
#   set_debug_mode(True)
  testsets_part4()
  grader_summary("part4")