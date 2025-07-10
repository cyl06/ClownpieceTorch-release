# Sigmoid, ReLU, Tanh, LeakyReLU

from clownpiece.tensor import Tensor
from clownpiece.nn.module import Module

class Sigmoid(Module):
  def __init__(self):
    super().__init__()
  
  def forward(self, x: Tensor) -> Tensor:
    return 1 / (1 + (-x).exp())
  
class ReLU(Module):
  def __init__(self):
    super().__init__()
    
  def forward(self, x: Tensor) -> Tensor:
    return x * (x > 0)
    
class Tanh(Module):
  def __init__(self):
    super().__init__()

  def forward(self, x: Tensor) -> Tensor:
    return x.tanh()
      
class LeakyReLU(Module):
  def __init__(self, negative_slope: float = 0.01):
    super().__init__()
    self.negative_slope = negative_slope

  def forward(self, x: Tensor) -> Tensor:
    return x * (x > 0) + x * (x <= 0) * self.negative_slope