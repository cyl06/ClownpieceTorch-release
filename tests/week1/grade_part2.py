import sys
import os


import torch
from graderlib import self_path
from graderlib import set_debug_mode, testcase, grader_summary



@testcase(name="utils1: from scalar", score=10)

def utils1(impl = torch):
  scalar = 42
  if impl.__name__ == "torch":
    a = impl.tensor(scalar)
  else:
    a = impl.Tensor(scalar)
  return (impl.numel(a), a.dim(), tuple(a.size()))

@testcase(name="utils2: from 1-D tensor", score=10)

def utils2(impl = torch):
  lst = [1, 2, 3, 4, 5]
  a = impl.Tensor(lst)
  return (impl.numel(a), a.dim(), tuple(a.size()), a.size(0))

@testcase(name="utils3: from n-D tensor", score=10)

def utils3(impl = torch):
  a = impl.zeros((1, 2, 3, 4, 5))
  return (impl.numel(a), a.dim(), tuple(a.size()), a.size(0), a.size(1), a.size(2), a.size(3), a.size(4))

@testcase(name="clone1: correctness of clone", score=10)

def clone1(impl = torch):
  lst = [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]]
  a = impl.Tensor(lst)
  b = a.clone()
  return b

@testcase(name="clone2: deepness of clone", score=10)

def clone2(impl = torch):
  lst = [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]]
  a = impl.Tensor(lst)
  b = a.clone()
  if impl.__name__ == "torch":
    b[0][0] = -1
  else:
    b.change_data_at(0, -1)
  return a

@testcase(name="copyfrom1: correctness of copyfrom", score=10)

def copyfrom1(impl = torch):
  lst = [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]]
  a = impl.Tensor(lst)
  print("A = ", a)
  b = impl.zeros((2, 5))
  print("B = ", b)
  b.copy_(a)
  return b

@testcase(name="copyfrom2: throwing error", score=10)

def copyfrom2(impl = torch):
  lst = [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]]
  a = impl.Tensor(lst)
  b = impl.zeros((2, 4))
  if impl.__name__ == "torch":
    return True
  else:
    try:
      b.copy_(a)
    except RuntimeError as e:
      return True
    else:
      return False


def testsets_part2():
  utils1()
  utils2()
  utils3()
  clone1()
  clone2()
  copyfrom1()
  copyfrom2()

if __name__ == "__main__":
  print("Beginning grading part 2")
  set_debug_mode(True)
  testsets_part2()
  grader_summary("part2")