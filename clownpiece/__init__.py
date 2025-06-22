from . import tensor_impl as cp
from .tensor import Tensor, TensorBase
from .utils import wrap_tuple
from .tensor_impl import TensorBaseImpl
from .autograd import backward


__all__ = ['TensorBaseImpl', 'TensorBase', 'Tensor']

for name in ['ones', 'zeros', 'empty', 'empty_like', 'ones_like', 'zeros_like', 'randn', 'randn_like', 'stack', 'cat', 'broadcast']:
    globals()[name] = getattr(Tensor, name)