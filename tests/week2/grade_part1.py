"""
Clone/Contiguous/Subscriptor
"""
import torch
from graderlib import set_debug_mode, testcase, grader_summary, print_separate_line

@testcase(name="clone1", score=10)
def clone1(impl = torch):  
  a = impl.ones((2, 3), requires_grad=True)
  b = a.clone()
  b.backward(impl.ones_like(b))
  return b, a.grad

@testcase(name="clone2", score=10)
def clone2(impl = torch):  
  a = impl.ones((2, 3), requires_grad=True)
  b = a.clone()
  b.backward(impl.ones_like(b))
  
  return b, a.grad

@testcase(name="contiguous1", score=10)
def contiguous1(impl = torch):
  a = impl.ones((2, 3), requires_grad=True)
  b = a.contiguous()
  b.backward(impl.ones_like(b))
  return b, a.grad

@testcase(name="contiguous2", score=10)
def contiguous2(impl = torch):
  a = impl.ones((2, 3), requires_grad=False)
  
  aT = impl.Tensor(a.transpose(-1, -2))
  aT.requires_grad_()
  b = aT.contiguous()
  b.backward(impl.ones_like(b))
  return b, aT.grad

@testcase(name="subscriptor1", score=10)
def subscriptor1(impl = torch):
  a = impl.ones((2, 3), requires_grad=True)
  b = a[0, :]
  b.backward(impl.ones_like(b))
  
  return b, a.grad

@testcase(name="subscriptor2", score=10)
def subscriptor2(impl = torch):
  a = impl.ones((2, 3), requires_grad=True)
  b = a[0, 1:2]
  b.backward(impl.ones_like(b))
  
  return b, a.grad

def testsets_part1():
  print_separate_line()
  print("Testing Part1 Clone/Contiguous/Subscriptor...")
  clone1()
  clone2()
  contiguous1()
  contiguous2()
  subscriptor1()
  subscriptor2()
  
if __name__ == "__main__":
  set_debug_mode(True)
  testsets_part1()
  grader_summary("part1")