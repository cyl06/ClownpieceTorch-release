# Linear, Embedding, LayerNorm, BatchNorm, MultiheadAttention

from typing import Optional
from clownpiece.tensor import Tensor
from clownpiece.nn.module import Module, Parameter, Buffer
from . import init
import math



class Linear(Module):

  def __init__(self, in_features: int, out_features: int, bias: bool=True):
    # remember to wrap W and b in Parameter class, otherwise they won't be registered.
    # for now, init W, b with empty
    super().__init__()
    self.in_features = in_features
    self.out_features = out_features
    self.weight = Parameter(Tensor.empty((out_features, in_features)))
    self.bias = Parameter(Tensor.empty((out_features,))) if bias else None

  def forward(self, x: Tensor) -> Tensor:
    y = x @ self.weight.transpose()
    if self.bias is not None:
      y += self.bias
    return y

  def extra_repr(self):
    return f"in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}"


class Embedding(Module):
  def __init__(self, num_embd: int, embd_dim: int):
    pass

  def forward(self, x: Tensor) -> Tensor:
    pass
    
class LayerNorm(Module):
  def __init__(self, num_features: int, eps: float = 1e-5, affine: bool = True):
    # input is reshaped to (-1, num_features) for normalziation.
    # for example:
    #   to normalize last two dimensions of tensor (batch_size, height, width)
    #   then num_features should be height x width
    # this interface differs from pytorch
    pass

  def forward(self, x: Tensor) -> Tensor:
    pass

class BatchNorm(Module):
  def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1, affine: bool = True):
    pass

  def forward(self, x: Tensor) -> Tensor:
    pass
    
class MultiheadAttention(Module):
  def __init__(self, hidden_dim: int, num_heads: int, bias: bool = True):
    pass

  def forward(self, hidden_states: Tensor, attn_mask: Optional[Tensor] = None):
    pass