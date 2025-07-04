"""
    Implement Various Functions
"""

from typing import List, Union
import copy

from clownpiece.tensor import Tensor, zeros, zeros_like
from clownpiece.autograd.autograd import Node, Edge
from clownpiece.autograd.no_grad import no_grad
from clownpiece.utils_ import wrap_tuple


class Context():
    def __init__(self):
        self.saved_tensors = []
        
    def save_for_backward(self, *args) -> None:
        self.saved_tensors.extend(
            [self.repack_tensor(tensor) for tensor in args if isinstance(tensor, Tensor)]
        )
        
    def get_saved_tensors(self) -> List[Tensor]:
        return self.saved_tensors
    
    @staticmethod
    def repack_tensor(tensor: Tensor):
        # avoid cyclic reference
        if isinstance(tensor, Tensor):
            return copy.copy(tensor) # shallow copy
        else:
            return tensor
    

class Function(Node):
    """
    Base class for all functions.
    """
    ctx: Context
    
    def __init__(self):
        super().__init__()
        self.ctx = None
        
    @staticmethod
    def forward(ctx: Context, *args):
        raise NotImplementedError("Forward method not implemented")

    @staticmethod
    def backward(ctx: Context, *args):
        raise NotImplementedError("Backward method not implemented")    
    
    # run forward pass
    def apply(self, *args, **kwargs):
        # your implement here

        # step 1. initialize self.ctx and populate self.next_edges
        self.ctx = Context()
        self.next_edges = [Edge.gradient_edge(arg) for arg in args]

        # step 2. outputs = self.forward(...) with no_grad
        with no_grad():
            outputs = self.forward(self.ctx, *args, **kwargs)

        # step 3. set grad_fn for outputs to self
        outputs = wrap_tuple(outputs)
        for i, out in enumerate(outputs):
            if isinstance(out, Tensor):
                out.requires_grad = True
                out.grad_fn, out.output_nr = self, i

        # step 4. return outputs
        return outputs[0] if len(outputs) == 1 else outputs

    
    # run backward pass
    def run(self, *args):
        # your implement here

        # step 1. grad_inputs = self.backward(...) with no_grad
        with no_grad():
            grad_inputs = self.backward(self.ctx, *args)

        # step 2. return grad_inputs
        return wrap_tuple(grad_inputs)

class AccumulateGrad(Function):
    """
    Accumulate gradient to .grad field
    
    grad_fn for leaf tensors
    """
    def __init__(self, input: Tensor):
        # your implement here
        super().__init__()
        self.ctx = Context()
        self.ctx.input = input
    
    # this forward should never be called
    @staticmethod
    def forward(ctx: Context):
        return None
    
    @staticmethod
    def backward(ctx: Context, output_grad: Tensor):
        # your implement here
        if ctx.input.requires_grad:
            if ctx.input.grad is None:
                ctx.input.grad = zeros(ctx.input.shape)
            ctx.input.grad += output_grad
        return None

"""
    Clone Contiguous
"""

class Clone(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor):
        return input.clone()
    
    @staticmethod
    def backward(ctx: Context, grad_output: Tensor):
        return grad_output

class Contiguous(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor):
        return input.contiguous()
    
    @staticmethod
    def backward(ctx: Context, grad_output: Tensor):
        return grad_output
    
"""
    Subscriptor
"""

class Subscriptor(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor, index_or_slice: Union[int, slice, List[int], List[slice]]):
        ctx.input_shape = input.shape
        ctx.index_or_slice = index_or_slice
        return input[index_or_slice]
    
    @staticmethod
    def backward(ctx: Context, grad_output: Tensor):
        grad_input = zeros(ctx.input_shape)
        sub_grad_input = grad_input[ctx.index_or_slice]
        sub_grad_input.copy_(grad_output)
        return grad_input, None
"""
    Element-wise Binary and Unary Operators
"""

class Neg(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor):
        return -input
    
    @staticmethod
    def backward(ctx: Context, grad_output: Tensor):
        return -grad_output

# backward method for broadcast
def reduce_broadcast(grad_output: Tensor, input_shape: List[int], output_shape: List[int], end_dim: int = 0) -> Tensor:
    # end_dim argument is for matmul, which only broadcasts dim <= dim() - 2
    extra_dim = len(output_shape) - len(input_shape)
    for i in range(extra_dim):
        grad_output = grad_output.sum(0, keepdims=False)
    for i in range(len(input_shape) + end_dim):
        if input_shape[i] == 1 and output_shape[extra_dim + i] > 1:
            grad_output = grad_output.sum(i, keepdims=True)
    return grad_output

# binary op forward decorator
def binary_op_forward_wrapper(forward_impl):
    def wrapper(ctx: Context, input1: Tensor, input2: Tensor):
        # save input shapes into ctx
        ctx.input1_shape = input1.shape
        ctx.input2_shape = input2.shape
        # call forward_impl
        return forward_impl(ctx, input1, input2)
    return wrapper

# binary op backward decorator
def binary_op_backward_wrapper(backward_impl):
    def wrapper(ctx: Context, grad_output: Tensor):
        # call backward_impl to get grad_inputs_broadcasted
        grad_input1_broadcasted, grad_input2_broadcasted = backward_impl(ctx, grad_output)
        # call reduce_broadcast to get grad_inputs
        grad_input1 = reduce_broadcast(grad_input1_broadcasted, ctx.input1_shape, grad_output.shape)
        grad_input2 = reduce_broadcast(grad_input2_broadcasted, ctx.input2_shape, grad_output.shape)
        return grad_input1, grad_input2
    return wrapper

class Add(Function):
    @staticmethod
    @binary_op_forward_wrapper
    def forward(ctx: Context, input1: Tensor, input2: Tensor):
        return input1 + input2
    
    @staticmethod
    @binary_op_backward_wrapper
    def backward(ctx: Context, grad_output: Tensor):
        return grad_output, grad_output
    
class Sub(Function):
    @staticmethod
    @binary_op_forward_wrapper
    def forward(ctx: Context, input1: Tensor, input2: Tensor):
        return input1 - input2
    
    @staticmethod
    @binary_op_backward_wrapper
    def backward(ctx: Context, grad_output: Tensor):
        return grad_output, -grad_output
    
class Mul(Function):
    @staticmethod
    @binary_op_forward_wrapper
    def forward(ctx: Context, input1: Tensor, input2: Tensor):
        ctx.save_for_backward(input1, input2)
        return input1 * input2
    
    @staticmethod
    @binary_op_backward_wrapper
    def backward(ctx, grad_output):
        input1, input2 = ctx.get_saved_tensors()[0:2]
        return grad_output * input2, grad_output * input1
    
class Div(Function):
    @staticmethod
    @binary_op_forward_wrapper
    def forward(ctx: Context, input1: Tensor, input2: Tensor):
        ctx.save_for_backward(input1, input2)
        return input1 / input2
    
    @staticmethod
    @binary_op_backward_wrapper
    def backward(ctx, grad_output):
        input1, input2 = ctx.get_saved_tensors()[0:2]
        return grad_output / input2, grad_output * input1 / -(input2 * input2)
    
class Sign(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor):
        ctx.input_shape = input.shape
        return input.sign()
    
    @staticmethod
    def backward(ctx: Context, grad_output: Tensor):
        return zeros(ctx.input_shape)
    
class Abs(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor):
        ctx.save_for_backward(input)
        return input.abs()
    
    @staticmethod
    def backward(ctx: Context, grad_output: Tensor):
        input = ctx.get_saved_tensors()[0]
        return grad_output * input.sign()
    
class Sin(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor):
        ctx.save_for_backward(input)
        return input.sin()
        
    @staticmethod
    def backward(ctx: Context, grad_output: Tensor):
        input = ctx.get_saved_tensors()[0]
        return grad_output * input.cos()

class Cos(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor):
        ctx.save_for_backward(input)
        return input.cos()
    
    @staticmethod
    def backward(ctx: Context, grad_output: Tensor):
        input = ctx.get_saved_tensors()[0]
        return grad_output * -(input.sin())

class Tanh(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor):
        output = input.tanh()
        ctx.save_for_backward(output)
        return output
    
    @staticmethod
    def backward(ctx: Context, grad_output: Tensor):
        output = ctx.get_saved_tensors()[0]
        return grad_output * (1 - output * output) # broadcast has been perfectly completed

class Clamp(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor, min_val: float, max_val: float):
        ctx.save_for_backward(input)
        ctx.min_val = min_val
        ctx.max_val = max_val
        return input.clamp(min_val, max_val)
    
    @staticmethod
    def backward(ctx: Context, grad_output: Tensor):
        input = ctx.get_saved_tensors()[0]
        mask = (ctx.min_val <= input) * (input <= ctx.max_val)
        return grad_output * mask, None, None

class Log(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor):
        ctx.save_for_backward(input)
        return input.log()
    
    @staticmethod
    def backward(ctx: Context, grad_output: Tensor):
        input = ctx.get_saved_tensors()[0]
        return grad_output / input

class Exp(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor):
        output = input.exp()
        ctx.save_for_backward(output)
        return output
    
    @staticmethod
    def backward(ctx: Context, grad_output: Tensor):
        output = ctx.get_saved_tensors()[0]
        return grad_output * output

class Pow(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor, exponent: float): 
        output = input.pow(exponent)
        ctx.save_for_backward(input, output)
        ctx.exponent = exponent
        return output
    
    @staticmethod
    def backward(ctx: Context, grad_output: Tensor):
        input, output = ctx.get_saved_tensors()[0:2]
        return grad_output * ctx.exponent * output / input
    
class Sqrt(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor):
        output = input.sqrt()
        ctx.save_for_backward(output)
        return output
    
    @staticmethod
    def backward(ctx: Context, grad_output: Tensor):
        output = ctx.get_saved_tensors()[0]
        return grad_output / (output * 2)
    
"""
    Matrix Multiplication
"""

class MatMul(Function):
    @staticmethod
    def forward(ctx: Context, input1: Tensor, input2: Tensor):
        ctx.save_for_backward(input1, input2)
        return input1.matmul(input2)
    
    @staticmethod
    def backward(ctx: Context, grad_output: Tensor):
        input1, input2 = ctx.get_saved_tensors()[0:2]
        ipt1_1D, ipt2_1D = (input1.dim() == 1), (input2.dim() == 1)
        
        if ipt1_1D == 1:
            input1 = input1.unsqueeze(-2)
        if ipt2_1D == 1:
            input2 = input2.unsqueeze(-1)
        
        if ipt1_1D == 1 and ipt2_1D == 1:
            grad_output = grad_output.reshape([1, 1])
        elif ipt1_1D == 1:
            grad_output = grad_output.unsqueeze(-2)
        elif ipt2_1D == 1:
            grad_output = grad_output.unsqueeze(-1)
        
        grad_input1_broadcasted = grad_output.matmul(input2.transpose())
        grad_input2_broadcasted = input1.transpose().matmul(grad_output)
        
        grad_input1 = reduce_broadcast(grad_input1_broadcasted, input1.shape, grad_input1_broadcasted.shape, end_dim=-2)
        grad_input2 = reduce_broadcast(grad_input2_broadcasted, input2.shape, grad_input2_broadcasted.shape, end_dim=-2)
        
        if ipt1_1D == 1:
            grad_input1 = grad_input1.squeeze(-2)
        if ipt2_1D == 1:
            grad_input2 = grad_input2.squeeze(-1)
        
        return grad_input1, grad_input2

"""
    Reduction and Normalization Operations
"""

def reduce_forward_wrapper(forward_impl):
    pass

class Sum(Function):
    @staticmethod
    @reduce_forward_wrapper
    def forward(ctx: Context, input: Tensor, dim: Union[int, List[int], None], keepdims: bool = False):
        pass
    
    @staticmethod
    def backward(ctx: Context, grad_output: Tensor):
        pass
    
class Max(Function):
    @staticmethod
    @reduce_forward_wrapper
    def forward(ctx: Context, input: Tensor, dim: int, keepdims: bool = False):
        pass
    
    @staticmethod
    def backward(ctx: Context, grad_output: Tensor, grad_indices: Tensor = None):
        pass
    
class Softmax(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor, dim: int):
        pass
    
    @staticmethod
    def backward(ctx: Context, grad_output: Tensor):
        pass
    
"""
    Shape Manipulation
"""

class Permute(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor, perm: List[int]):
        pass

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor):
        pass
    
class Transpose(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor, dim0: int, dim1: int):
        pass
    
    @staticmethod
    def backward(ctx: Context, grad_output: Tensor):
        pass

class Reshape(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor, shape: List[int]):
        pass
    
    @staticmethod
    def backward(ctx: Context, grad_output: Tensor):
        pass
    
class View(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor, shape: List[int]):
        pass
    
    @staticmethod
    def backward(ctx: Context, grad_output: Tensor):
        pass
    
class Narrow(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor, dim: int, start: int, length: int):
        pass
    
    @staticmethod
    def backward(ctx: Context, grad_output: Tensor):
        pass
    
class Chunk(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor, chunks: int, dim: int = 0):
        pass
        
    @staticmethod
    def backward(ctx: Context, *grad_outputs: Tensor):
        pass
    
class Split(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor, split: Union[int, List[int]], dim: int = 0):
        pass

    @staticmethod
    def backward(ctx: Context, *grad_outputs: Tensor):
        pass
    
class Stack(Function):
    @staticmethod
    def forward(ctx: Context, *inputs: Tensor, dim: int = 0):
        pass
    
    @staticmethod
    def backward(ctx: Context, grad_output: Tensor):
        pass
    
class Cat(Function):
    @staticmethod
    def forward(ctx: Context, *inputs: Tensor, dim: int = 0):
        pass
    
    @staticmethod
    def backward(ctx, grad_output: Tensor):
        pass
        
class Squeeze(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor, dim: int = 0):
        pass
    
    @staticmethod
    def backward(ctx: Context, grad_output: Tensor):    
        pass
    
class Unsqueeze(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor, dim: int = 0):
        pass
    
    @staticmethod
    def backward(ctx: Context, grad_output: Tensor):
        pass
    
class BroadcastTo(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor, shape: List[int]):
        pass
    
    @staticmethod
    def backward(ctx: Context, grad_output: Tensor):
        pass
    
class Broadcast(Function):
    @staticmethod
    def forward(ctx: Context, *inputs: Tensor):
        pass
    
    @staticmethod
    def backward(ctx: Context, *grad_outputs: Tensor):
        pass