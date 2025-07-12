"""
  +-*/
"""

import torch
from graderlib import set_debug_mode, testcase, grader_summary, print_separate_line


lhs_hard = [[ 0.4043,  0.6582, -0.1347], \
            [-1.5471, -0.4674, -1.5250]]
rhs_hard = [[-0.8357, -0.0665, -0.6086], \
            [-0.6420,  3.3154, -0.7132]]

@testcase(name="add1", score=10)
def add1(impl = torch):
  a = impl.ones((2, 3), requires_grad=True)
  b = impl.ones((2, 3), requires_grad=True)
  c = a + b
  c.backward(impl.ones_like(c))
  
  d = impl.Tensor(lhs_hard)
  d.requires_grad_()
  e = impl.Tensor(rhs_hard)
  e.requires_grad_()
  f = d + e
  f.backward(impl.ones_like(c))
  
  return a.grad, b.grad, c, d.grad, e.grad, f

@testcase(name="add2", score=10)
def add2(impl = torch):
  a = 1
  b = impl.ones((2, 3), requires_grad=True)
  c = a + b
  c.backward(impl.ones_like(c))
  
  d = impl.ones((2, 3), requires_grad=True)
  e = 1
  f = d + e
  f.backward(impl.ones_like(f))
  
  h = 0.6
  i = impl.Tensor(rhs_hard)
  i.requires_grad_()
  j = h + i
  j.backward(impl.ones_like(j))
  
  k = impl.Tensor(lhs_hard)
  k.requires_grad_()
  l = 0.3
  m = k + l
  m.backward(impl.ones_like(m))
  
  return b.grad, c, d.grad, f, i.grad, j, k.grad, m

@testcase(name="sub1", score=10)
def sub1(impl = torch):
  a = impl.ones((2, 3)) * 2
  a.requires_grad_()
  b = impl.ones((2, 3), requires_grad=True)
  c = a - b
  c.backward(impl.ones_like(c))
  
  d = impl.Tensor(lhs_hard)
  d.requires_grad_()
  e = impl.Tensor(rhs_hard)
  e.requires_grad_()
  f = d - e
  f.backward(impl.ones_like(f))
    
  return a.grad, b.grad, c, d.grad, e.grad, f

@testcase(name="sub2", score=10)
def sub2(impl = torch):
  a = 2.0
  b = impl.ones((2, 3), requires_grad=True)
  c = a - b
  c.backward(impl.ones_like(c))
  
  d = impl.ones((2, 3), requires_grad=True)
  e = 3.0
  f = d - e
  f.backward(impl.ones_like(f))
  
  h = 0.5
  i = impl.Tensor(rhs_hard)
  i.requires_grad_()
  j = h - i
  j.backward(impl.ones_like(j))
  
  k = impl.Tensor(lhs_hard)
  k.requires_grad_()
  l = 0.2
  m = k - l
  m.backward(impl.ones_like(m))
  
  return b.grad, c, d.grad, f, i.grad, j, k.grad, m

@testcase(name="mul1", score=10)
def mul1(impl = torch):
  a = impl.ones((2, 3)) * 3
  a.requires_grad_()
  b = impl.ones((2, 3)) * 2
  b.requires_grad_()
  c = a * b
  c.backward(impl.ones_like(c))
  
  d = impl.Tensor(lhs_hard)
  d.requires_grad_()
  e = impl.Tensor(rhs_hard)
  e.requires_grad_()
  f = d * e
  f.backward(impl.ones_like(f))
  
  return a.grad, b.grad, c, d.grad, e.grad, f

@testcase(name="mul2", score=10)
def mul2(impl = torch):
  a = 2.5
  b = impl.ones((2, 3), requires_grad=True)
  c = a * b
  c.backward(impl.ones_like(c))
  
  d = impl.ones((2, 3), requires_grad=True)
  e = 3.5
  f = d * e
  f.backward(impl.ones_like(f))
  
  h = 0.4
  i = impl.Tensor(rhs_hard)
  i.requires_grad_()
  j = h * i
  j.backward(impl.ones_like(j))
  
  k = impl.Tensor(lhs_hard)
  k.requires_grad_()
  l = 1.2
  m = k * l
  m.backward(impl.ones_like(m))
  
  return b.grad, c, d.grad, f, i.grad, j, k.grad, m

@testcase(name="div1", score=10)
def div1(impl = torch):
  a = impl.ones((2, 3)) * 6
  a.requires_grad_()
  b = impl.ones((2, 3)) * 2
  b.requires_grad_()
  c = a / b
  c.backward(impl.ones_like(c))
  
  d = impl.Tensor(lhs_hard) 
  d.requires_grad_()
  e = impl.Tensor(rhs_hard) 
  e.requires_grad_()
  f = d / e
  f.backward(impl.ones_like(f))
  
  return a.grad, b.grad, c, d.grad, e.grad, f

@testcase(name="div2", score=10)
def div2(impl = torch):
  a = 2.0
  b = impl.ones((2, 3), requires_grad=True)
  c = a / b
  c.backward(impl.ones_like(c))
  
  d = impl.ones((2, 3), requires_grad=True)
  e = 4.0
  f = d / e
  f.backward(impl.ones_like(f))
  
  h = 0.5
  i = impl.Tensor(rhs_hard)
  i.requires_grad_()
  j = h / i
  j.backward(impl.ones_like(j))
  
  k = impl.Tensor(lhs_hard)
  k.requires_grad_()
  l = 1.5
  m = k / l
  m.backward(impl.ones_like(m))
  
  return b.grad, c, d.grad, f, i.grad, j, k.grad, m

@testcase(name="extra", score=10)
def extra(impl = torch):
  a = impl.ones((2, 3)) * 6
  a.requires_grad_()
  b = impl.ones((2, 3)) * 2
  b.requires_grad_()
  x = a - b
  y = x * x
  y.backward(impl.ones_like(y))
  
  return a.grad, b.grad, x, y

def testsets_part3():
  print_separate_line()  
  print("Testing Part3 +-*/...")
  add1()
  add2()
  sub1()
  sub2()
  mul1()
  mul2()
  div1()
  div2()
  extra()
  
if __name__ == "__main__":
  testsets_part3()
    
  grader_summary("part3")