import torch
from torch.autograd import Variable
from collections import Iterable


def iter_gradients(x):
    if isinstance(x, Variable):
        if x.requires_grad:
            yield x.grad.data if x.grad is not None else None
    elif isinstance(x, Iterable):
        for elem in x:
            for result in iter_gradients(elem):
                yield result


def zero_gradients(x):
    if isinstance(x, Variable):
        if x.grad is not None:
            x.grad.detach_()
            x.grad.data.zero_()
    elif isinstance(x, Iterable):
        for elem in x:
            zero_gradients(elem)


def make_jacobian(input, num_out):
    if isinstance(input, Variable) and not input.requires_grad:
        return None
    elif torch.is_tensor(input) or isinstance(input, Variable):
        return torch.zeros(input.nelement(), num_out)
    elif isinstance(input, Iterable):
        jacobians = list(filter(
            lambda x: x is not None, (make_jacobian(elem, num_out) for elem in input)))
        if not jacobians:
            return None
        return type(input)(jacobians)
    else:
        return None


def iter_tensors(x, only_requiring_grad=False):
    if torch.is_tensor(x):
        yield x
    elif isinstance(x, Variable):
        if x.requires_grad or not only_requiring_grad:
            yield x.data
    elif isinstance(x, Iterable):
        for elem in x:
            for result in iter_tensors(elem, only_requiring_grad):
                yield result


def contiguous(input):
    if torch.is_tensor(input):
        return input.contiguous()
    elif isinstance(input, Variable):
        return input.contiguous()
    elif isinstance(input, Iterable):
        return type(input)(contiguous(e) for e in input)
    return input


def get_numerical_jacobian(fn, input, target, eps=1e-3):
    # To be able to use .view(-1) input must be contiguous
    input = contiguous(input)
    output_size = fn(input).numel()
    jacobian = make_jacobian(target, output_size)

    # It's much easier to iterate over flattened lists of tensors.
    # These are reference to the same objects in jacobian, so any changes
    # will be reflected in it as well.
    x_tensors = [t for t in iter_tensors(target, True)]
    j_tensors = [t for t in iter_tensors(jacobian)]

    outa = torch.DoubleTensor(output_size)
    outb = torch.DoubleTensor(output_size)

    # TODO: compare structure
    for x_tensor, d_tensor in zip(x_tensors, j_tensors):
        flat_tensor = x_tensor.view(-1)
        for i in range(flat_tensor.nelement()):
            orig = flat_tensor[i]
            flat_tensor[i] = orig - eps
            outa.copy_(fn(input), broadcast=False)
            flat_tensor[i] = orig + eps
            outb.copy_(fn(input), broadcast=False)
            flat_tensor[i] = orig

            outb.add_(-1, outa).div_(2 * eps)
            d_tensor[i] = outb

    return jacobian


def get_analytical_jacobian(input, output):
    jacobian = make_jacobian(input, output.numel())
    grad_output = output.data.clone().zero_()
    flat_grad_output = grad_output.view(-1)

    for i in range(flat_grad_output.numel()):
        flat_grad_output.zero_()
        flat_grad_output[i] = 1
        zero_gradients(input)
        output.backward(grad_output, retain_graph=True)
        for jacobian_x, d_x in zip(jacobian, iter_gradients(input)):
            if d_x is None:
                jacobian_x[:, i].zero_()
            else:
                jacobian_x[:, i] = d_x.to_dense() if d_x.is_sparse else d_x

    return jacobian


def _as_tuple(x):
    if isinstance(x, tuple):
        return x
    elif isinstance(x, list):
        return tuple(x)
    else:
        return x,


def gradcheck(func, inputs, eps=1e-6, atol=1e-5, rtol=1e-3):
    """Check gradients computed via small finite differences
       against analytical gradients

    The check between numerical and analytical has the same behaviour as
    numpy.allclose https://docs.scipy.org/doc/numpy/reference/generated/numpy.allclose.html
    meaning it check that
        absolute(a - n) <= (atol + rtol * absolute(n))
    is true for all elements of analytical jacobian a and numerical jacobian n.

    Args:
        func: Python function that takes Variable inputs and returns
            a tuple of Variables
        inputs: tuple of Variables
        eps: perturbation for finite differences
        atol: absolute tolerance
        rtol: relative tolerance

    Returns:
        True if all differences satisfy allclose condition
    """
    output = func(*inputs)
    output = _as_tuple(output)

    for i, o in enumerate(output):
        if not o.requires_grad:
            continue

        def fn(input):
            return _as_tuple(func(*input))[i].data

        analytical = get_analytical_jacobian(_as_tuple(inputs), o)
        numerical = get_numerical_jacobian(fn, inputs, inputs, eps)

        for a, n in zip(analytical, numerical):
            if not ((a - n).abs() <= (atol + rtol * n.abs())).all():
                return False

    # check if the backward multiplies by grad_output
    zero_gradients(inputs)
    output = _as_tuple(func(*inputs))
    torch.autograd.backward(output, [o.data.new(o.size()).zero_() for o in output])
    var_inputs = list(filter(lambda i: isinstance(i, Variable), inputs))
    if not var_inputs:
        raise RuntimeError("no Variables found in input")
    for i in var_inputs:
        if i.grad is None:
            continue
        if not i.grad.data.eq(0).all():
            return False

    return True

def gradgradcheck(func, inputs, grad_outputs, eps=1e-6, atol=1e-5, rtol=1e-3):
    def new_func(*inputs):
        outputs = func(*inputs)
        outputs = _as_tuple(outputs)
        grad_inputs = torch.autograd.grad(outputs, inputs, grad_outputs, only_inputs=True)
        return grad_inputs

    return gradcheck(new_func, inputs, eps, atol, rtol)
