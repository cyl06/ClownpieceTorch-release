"""
    Unary Operations
"""
import torch
from graderlib import set_debug_mode, testcase, grader_summary, print_separate_line

# Test data
unary_test_data_basic = [[-2., 0., 3.5], [-1.5, 2.5, -0.7]] # For neg, sign, abs
unary_test_data_angles = [[0., 0.5 * 3.14159, -3.14159 / 2], [3.14159, -0.25 * 3.14159, 0.75 * 3.14159]] # For sin, cos, tanh
unary_test_data_positive = [[1e-4, 1.0, 4.0], [0.25, 2.25, 9.0]] # For log, sqrt (strictly positive for log), pow
unary_test_data_exp = [[-2.0, 0.0, 1.5], [-1.0, 0.5, 2.0]] # For exp


@testcase(name="neg_op", score=10)
def neg_op(impl=torch):
    a_data = unary_test_data_basic
    a = impl.Tensor(a_data)
    a.requires_grad_()
    b = -a
    b.backward(impl.ones_like(b))
    return b, a.grad

@testcase(name="sign_op", score=10)
def sign_op(impl=torch):
    a_data = unary_test_data_basic
    a = impl.Tensor(a_data)
    a.requires_grad_()
    b = a.sign()
    b.backward(impl.ones_like(b))
    return b, a.grad

@testcase(name="abs_op", score=10)
def abs_op(impl=torch):
    a_data = unary_test_data_basic
    a = impl.Tensor(a_data)
    a.requires_grad_()
    b = a.abs()
    b.backward(impl.ones_like(b))
    return b, a.grad

@testcase(name="sin_op", score=10)
def sin_op(impl=torch):
    a_data = unary_test_data_angles
    a = impl.Tensor(a_data)
    a.requires_grad_()
    b = a.sin()
    b.backward(impl.ones_like(b))
    return b, a.grad

@testcase(name="cos_op", score=10)
def cos_op(impl=torch):
    a_data = unary_test_data_angles
    a = impl.Tensor(a_data)
    a.requires_grad_()
    b = a.cos()
    b.backward(impl.ones_like(b))
    return b, a.grad

@testcase(name="tanh_op", score=10)
def tanh_op(impl=torch):
    a_data = unary_test_data_angles
    a = impl.Tensor(a_data)
    a.requires_grad_()
    b = a.tanh()
    b.backward(impl.ones_like(b))
    return b, a.grad

@testcase(name="clamp_op", score=10)
def clamp_op(impl=torch):
    a_data = [[-2.5, 0.5, 3.0], [1.8, -0.7, 2.2]]
    min_val = -1.0
    max_val = 2.0
    a = impl.Tensor(a_data)
    a.requires_grad_()
    b = a.clamp(min_val, max_val)
    b.backward(impl.ones_like(b))
    return b, a.grad

@testcase(name="log_op", score=10)
def log_op(impl=torch):
    a_data = unary_test_data_positive # Ensure positive inputs
    a = impl.Tensor(a_data)
    a.requires_grad_()
    b = a.log()
    b.backward(impl.ones_like(b))
    return b, a.grad

@testcase(name="exp_op", score=10)
def exp_op(impl=torch):
    a_data = unary_test_data_exp
    a = impl.Tensor(a_data)
    a.requires_grad_()
    b = a.exp()
    b.backward(impl.ones_like(b))
    return b, a.grad

@testcase(name="pow_op", score=10)
def pow_op(impl=torch):
    a_data = unary_test_data_positive
    exponent = 1.5
    a = impl.Tensor(a_data)
    a.requires_grad_()
    b = a.pow(exponent)
    b.backward(impl.ones_like(b))
    return b, a.grad


@testcase(name="sqrt_op", score=10)
def sqrt_op(impl=torch):
    a_data = unary_test_data_positive # Ensure non-negative inputs
    a = impl.Tensor(a_data)
    a.requires_grad_()
    b = a.sqrt()
    b.backward(impl.ones_like(b))
    return b, a.grad


def testsets_part2():
    print_separate_line()
    print("Testing Part2 Unary Operations...")
    neg_op()
    sign_op()
    abs_op()
    sin_op()
    cos_op()
    tanh_op()
    clamp_op()
    log_op()
    exp_op()
    pow_op()
    sqrt_op()

if __name__ == "__main__":

    testsets_part2()
    
    grader_summary("Part2")