# Linear, Embedding, LayerNorm, BatchNorm, MultiheadAttention

from typing import Optional, List, Union
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
    if bias:
      self.bias = Parameter(Tensor.empty((out_features,)))
    else:
      self.register_parameter('bias', None)
    self.reset_parameters()
  
  def reset_parameters(self) -> None:
    b = math.sqrt(1 / self.in_features)
    init.uniform_(self.weight, -b, b)
    if self.bias is not None:
      init.uniform_(self.bias, -b, b)

  def forward(self, x: Tensor) -> Tensor:
    y = x @ self.weight.transpose()
    if self.bias is not None:
      y += self.bias
    return y

  def extra_repr(self):
    return f"in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}"


class Embedding(Module):
  def __init__(self, num_embd: int, embd_dim: int):
    super().__init__()
    self.num_embd = num_embd
    self.embd_dim = embd_dim
    self.weight = Parameter(Tensor.empty((num_embd, embd_dim)))
    init.normal_(self.weight)

  def forward(self, x: Tensor) -> Tensor:
    return self.weight[x]
  
  def extra_repr(self):
    return f"num_embd={self.num_embd}, embd_dim={self.embd_dim}"
  
class LayerNorm(Module):
  def __init__(self, num_features: int, eps: float = 1e-5, affine: bool = True):
    # input is reshaped to (-1, num_features) for normalziation.
    # for example:
    #   to normalize last two dimensions of tensor (batch_size, height, width)
    #   then num_features should be height x width
    # this interface differs from pytorch
    super().__init__()
    self.num_features = num_features
    self.eps = eps
    self.affine = affine
    
    if self.affine:
      self.weight = Parameter(Tensor.ones((num_features,)))
      self.bias = Parameter(Tensor.zeros((num_features,)))
    else:
      self.register_parameter('weight', None)
      self.register_parameter('bias', None)

  def forward(self, x: Tensor) -> Tensor:
    lastd = x.reshape([-1, self.num_features])
    mean = lastd.mean(dim=-1, keepdims=True)
    var = lastd.var(dim=-1, keepdims=True)
    hat_x = (x - mean) / (var + self.eps).sqrt()
    if self.affine:
      hat_x = hat_x * self.weight + self.bias
    return hat_x.reshape(list(x.shape))
  
  def extra_repr(self):
    return f"num_features={self.num_features}, eps={self.eps}, affine={self.affine}"

class BatchNorm(Module):
  def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1, affine: bool = True):
    super().__init__()
    self.num_features = num_features
    self.eps = eps
    self.momentum = momentum
    self.affine = affine
    
    if self.affine:
      self.weight = Parameter(Tensor.ones((num_features,)))
      self.bias = Parameter(Tensor.zeros((num_features,)))
    else:
      self.register_parameter('weight', None)
      self.register_parameter('bias', None)
    
    self.running_mean = Buffer(Tensor.zeros((num_features,)))
    self.running_var = Buffer(Tensor.ones((num_features,)))

  def forward(self, x: Tensor) -> Tensor:
    if self.training:
      current_mean = x.mean(dim=0)
      current_var = x.var(dim=0, unbiased=False)
      self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * current_mean
      self.running_var = (1 - self.momentum) * self.running_var + self.momentum * current_var
    else:
      current_mean = self.running_mean
      current_var = self.running_var
    
    hat_x = (x - current_mean) / (current_var + self.eps).sqrt()
    if self.affine:
      hat_x = hat_x * self.weight + self.bias
    return hat_x
  
  def extra_repr(self):
    return f"num_features={self.num_features}, self.eps={self.eps}, self.momentum={self.momentum}, self.affine={self.affine}"
    
class MultiheadAttention(Module):
  def __init__(self, hidden_dim: int, num_heads: int, bias: bool = True):
    super().__init__()
    if hidden_dim % num_heads != 0:
      raise ValueError(f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads})")
    
    self.hidden_dim = hidden_dim
    self.num_heads = num_heads
    self.bias = bias
    self.head_dim = hidden_dim // num_heads

    # Linear layers for Query, Key, Value projections and final output projection
    self.q_proj = Linear(hidden_dim, hidden_dim, bias)
    self.k_proj = Linear(hidden_dim, hidden_dim, bias)
    self.v_proj = Linear(hidden_dim, hidden_dim, bias)
    self.out_proj = Linear(hidden_dim, hidden_dim, bias)

  def forward(self, hidden_states: Tensor, attn_mask: Optional[Tensor] = None):
    q = self.q_proj(hidden_states)
    k = self.k_proj(hidden_states)
    v = self.v_proj(hidden_states)

    batch_size, seq_len, _ = hidden_states.shape

    q = q.reshape([batch_size, seq_len, self.num_heads, self.head_dim]).transpose(1, 2)
    k = k.reshape([batch_size, seq_len, self.num_heads, self.head_dim]).transpose(1, 2)
    v = v.reshape([batch_size, seq_len, self.num_heads, self.head_dim]).transpose(1, 2)

    attn_scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)

    if attn_mask is not None:
      attn_scores = attn_scores + attn_mask * -1e9

    attn_probs = attn_scores.softmax(dim=-1)

    attn_output = attn_probs @ v
    attn_output = attn_output.transpose(1, 2).reshape([batch_size, seq_len, self.hidden_dim])
    
    output = self.out_proj(attn_output)
    return output
  
  def extra_repr(self):
    return f"hidden_dim={self.hidden_dim}, num_heads={self.num_heads}, bias={self.bias}"

class Conv2D(Module):
  def __init__(self, in_channels: int, out_channels: int, kernel_size: Union[int, List[int]], bias: bool = True):
    super().__init__()
    self.in_channels = in_channels
    self.out_channels = out_channels
    self.kernel_size = kernel_size if isinstance(kernel_size, (list, tuple)) else (kernel_size, kernel_size)
    
    self.weight = Parameter(Tensor.empty((out_channels, in_channels, *self.kernel_size)))
    
    if bias:
      self.bias = Parameter(Tensor.empty((out_channels,)))
    else:
      self.register_parameter('bias', None)
    
    self.reset_parameters()

  def reset_parameters(self) -> None:
    fan_in = self.in_channels * self.kernel_size[0] * self.kernel_size[1]
    gain = init.calcuate_gain("relu")
    b = gain * math.sqrt(3.0 / fan_in)
    init.uniform_(self.weight, -b, b)
    if self.bias:
      init.uniform_(self.bias, -b, b)

  def forward(self, x: Tensor) -> Tensor:
    unfolded = x.unfold(self.kernel_size) # (N, C_in * kh * kw, L)
    kernel_weight = self.weight.reshape([self.out_channels, -1]) # (C_out, C_in * kh * kw)
    
    output = kernel_weight @ unfolded # (N, C_out, L)
    
    batch_size, _, height, width = x.shape
    out_height = height - self.kernel_size[0] + 1
    out_width = width - self.kernel_size[1] + 1
    
    output = output.reshape([batch_size, self.out_channels, out_height, out_width]) # (N, C_out, H_out, W_out)
    
    if self.bias:
      output += self.bias.view([1, -1, 1, 1])
    
    return output

  def extra_repr(self):
    return f"in_channels={self.in_channels}, out_channels={self.out_channels}, kernel_size={self.kernel_size}, bias={True if self.bias else False}"