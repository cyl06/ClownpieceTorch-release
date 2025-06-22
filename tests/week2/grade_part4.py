"""
  Matmul
"""
import torch
from graderlib import set_debug_mode, testcase, grader_summary, print_separate_line

lmat_data_simpl = [[1, 2, 3],
                   [1, 1, 1]] # 2 * 3
rmat_data_simpl = [[0, 1, 0],
                   [1, 0, 1],
                   [-1, -1, -1]] # 3 * 3

lmat_data = \
       [[-0.1881,  0.4301, -0.2369, -0.4883,  0.3123],
        [ 1.4300,  0.0990, -0.5364, -0.4863,  0.2455],
        [ 0.8592, -0.4500,  0.6266,  0.7882,  1.4241]] # 3*5
rmat_data = \
       [[ 0.6283,  1.8698, -0.4764,  0.1726],
        [ 1.1265, -2.0787,  0.7971,  0.3262],
        [ 0.8133, -2.3465,  0.8429,  0.3183],
        [ 1.2807, -0.3375,  1.3699,  1.0875],
        [ 0.8404, -0.2846,  1.0160,  0.4048]] # 5*4

lvec_data_simpl = [1, -1, 1] # 1*3
rvec_data_simpl = [1, 2, 3] # 1*3

lvec_data = [0.4988, -0.7680, -1.5290, -1.1482,  1.1839] # 1*5
rvec_data = [-1.5693,  1.6855, -1.4498, -0.4059, -0.8464] # 1*5

lbmat_data = \
       [[[[ 0.4454,  0.1129,  0.0439, -1.3556, -0.3421],
          [-0.1757,  1.3784,  1.0133,  0.2585, -0.2453],
          [-1.4564,  0.3136, -1.0170,  0.4356, -0.6302]]],


        [[[ 0.9641,  2.6973,  1.6681,  0.5993, -0.7435],
          [ 1.3640,  0.5214, -0.0495,  0.1118, -0.7354],
          [ 0.8125, -0.1413,  0.0901,  0.2137, -0.0084]]]] # 2*1*3*5
       
rbmat_data = \
       [[[ 0.4776, -2.9350, -0.9743, -0.1935],
         [ 0.7217, -0.8606, -0.5964,  1.1155],
         [ 0.1282,  0.1577, -0.4656, -1.2169],
         [ 0.2632, -0.0030,  0.2982, -0.8387],
         [ 1.0646, -0.8062, -2.9018, -2.5361]],

        [[ 0.2271, -1.8711, -1.0631, -1.1534],
         [ 0.6070, -1.9680,  0.0459,  1.2907],
         [ 0.5063, -0.3761,  0.2917,  1.4551],
         [-0.6584, -1.6214,  0.6585, -0.4686],
         [ 0.5424, -2.2991,  0.0118,  1.3848]]] # 2*5*4

@testcase(name="matmul_lmat_rmat", score=10, timeout=1000)
def matmul_lmat_rmat(impl=torch):
  a = impl.Tensor(lmat_data_simpl)
  a.requires_grad_()
  b = impl.Tensor(rmat_data_simpl)
  b.requires_grad_()
  c = a.matmul(b)
  c.backward(impl.ones_like(c))
  
  d = impl.Tensor(lmat_data)
  d.requires_grad_()
  e = impl.Tensor(rmat_data)
  e.requires_grad_()
  f = d.matmul(e)
  f.backward(impl.ones_like(f))

@testcase(name="matmul_lvec_rvec", score=10)
def matmul_lvec_rvec(impl=torch):
  a = impl.Tensor(lvec_data_simpl)
  a.requires_grad_()
  b = impl.Tensor(rvec_data_simpl)
  b.requires_grad_()
  c = a.matmul(b)
  c.backward(impl.ones_like(c))

  d = impl.Tensor(lvec_data)
  d.requires_grad_()
  e = impl.Tensor(rvec_data)
  e.requires_grad_()
  f = d.matmul(e)
  f.backward(impl.ones_like(f))
  
  return a.grad, b.grad, c, d.grad, e.grad, f

@testcase(name="matmul_lvec_rmat", score=10)
def matmul_lvec_rmat(impl=torch):
  a = impl.Tensor(lvec_data_simpl)
  a.requires_grad_()
  b = impl.Tensor(rmat_data_simpl)
  b.requires_grad_()
  c = a.matmul(b)
  c.backward(impl.ones_like(c))

  d = impl.Tensor(lvec_data)
  d.requires_grad_()
  e = impl.Tensor(rmat_data)
  e.requires_grad_()
  f = d.matmul(e)
  f.backward(impl.ones_like(f))
  
  return a.grad, b.grad, c, d.grad, e.grad, f

@testcase(name="matmul_lmat_rvec", score=10)
def matmul_lmat_rvec(impl=torch):
  a = impl.Tensor(lmat_data_simpl)
  a.requires_grad_()
  b = impl.Tensor(rvec_data_simpl)
  b.requires_grad_()
  c = a.matmul(b)
  c.backward(impl.ones_like(c))

  d = impl.Tensor(lmat_data)
  d.requires_grad_()
  e = impl.Tensor(rvec_data)
  e.requires_grad_()
  f = d.matmul(e)
  f.backward(impl.ones_like(f))
  
  return a.grad, b.grad, c, d.grad, e.grad, f

@testcase(name="matmul_lbmat_rmat", score=10)
def matmul_lbmat_rmat(impl=torch):
  a = impl.Tensor(lbmat_data)
  a.requires_grad_()
  b = impl.Tensor(rmat_data)
  b.requires_grad_()
  c = a.matmul(b)
  c.backward(impl.ones_like(c))
  
  return a.grad, b.grad, c

@testcase(name="matmul_lbmat_rvec", score=10)
def matmul_lbmat_rvec(impl=torch):
  a = impl.Tensor(lbmat_data)
  a.requires_grad_()
  b = impl.Tensor(rvec_data)
  b.requires_grad_()
  c = a.matmul(b)
  c.backward(impl.ones_like(c))
  
  return a.grad, b.grad, c

@testcase(name="matmul_lmat_rbmat", score=10)
def matmul_lmat_rbmat(impl=torch):
  a = impl.Tensor(lmat_data)
  a.requires_grad_()
  b = impl.Tensor(rbmat_data)
  b.requires_grad_()
  c = a.matmul(b)
  c.backward(impl.ones_like(c))
  
  return a.grad, b.grad, c

@testcase(name="matmul_lvec_rbmat", score=10)
def matmul_lvec_rbmat(impl=torch):
  a = impl.Tensor(lvec_data)
  a.requires_grad_()
  b = impl.Tensor(rbmat_data)
  b.requires_grad_()
  c = a.matmul(b)
  c.backward(impl.ones_like(c))
  
  return a.grad, b.grad, c

@testcase(name="matmul_lbmat_rbmat", score=10)
def matmul_lbmat_rbmat(impl=torch):
  a = impl.Tensor(lbmat_data)
  a.requires_grad_()
  b = impl.Tensor(rbmat_data)
  b.requires_grad_()
  c = a.matmul(b)
  c.backward(impl.ones_like(c))
  
  return a.grad, b.grad, c 

def testsets_part4():
  print_separate_line()  
  print("Testing Part4 Matmul...")
  matmul_lmat_rmat()
  matmul_lvec_rvec()
  matmul_lvec_rmat()
  matmul_lmat_rvec()
  
  matmul_lbmat_rmat()
  matmul_lbmat_rvec()
  matmul_lmat_rbmat()
  matmul_lvec_rbmat()
  matmul_lbmat_rbmat()
  
if __name__ == "__main__":
  testsets_part4()
  
  grader_summary("Matmul")