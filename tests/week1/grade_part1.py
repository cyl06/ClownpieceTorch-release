import sys
import os


import torch
from graderlib import self_path
from graderlib import set_debug_mode, testcase, grader_summary


@testcase(name="init1: from list 1D", score=10)

def init1(impl = torch):
  lst = [1, 2, 3, 4, 5]
  a = impl.Tensor(lst)
  return a

@testcase(name="init2: from list 2D", score=10)

def init2(impl = torch):
  lst = [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]]
  a = impl.Tensor(lst)
  return a

@testcase(name="init3: from list 4D", score=10)

def init3(impl = torch):
  lst = [[[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]]
  a = impl.Tensor(lst)
  return a

@testcase(name="init4: scalar", score=10)

def init4(impl = torch):
  scalar = 42
  if impl.__name__ == "torch":
    a = impl.tensor(scalar)
  else:
    a = impl.Tensor(scalar)
  return a

@testcase(name="copy1: correctness of copy", score=10)

def copy1(impl = torch):
  lst = [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]]
  a = impl.Tensor(lst)
  b = a
  return b

@testcase(name="copy2: shallowness of copy", score=10)

def copy2(impl = torch):
  lst = [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]]
  a = impl.Tensor(lst)
  b = a
  if impl.__name__ == "torch":
    b[0][0] = -1
  else:
    b.change_data_at(0, -1)
  return a

@testcase(name="item: singleton tensor", score=10)

def item(impl = torch):
  lst = [[1]]
  a = impl.Tensor(lst)
  return a.item()

def testsets_part1():
  init1()
  init2()
  init3()
  init4()
  copy1()
  copy2()
  item()

if __name__ == "__main__":
  print("Beginning grading part 1")
  set_debug_mode(True)
  testsets_part1()
  grader_summary("part1")