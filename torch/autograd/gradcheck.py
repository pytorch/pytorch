import torch
from torch.autograd import Variable


def iter_gradients(x):
    if isinstance(x, Variable):
        if x.requires_grad:
            yield x.grad.data
    else:
        for elem in x:
            for result in iter_gradients(elem):
                yield result


def zero_gradients(i):
    for t in iter_gradients(i):
        t.zero_()


def make_jacobian(input, num_out):
    if isinstance(input, Variable) and not input.requires_grad:
        return None
    if torch.is_tensor(input) or isinstance(input, Variable):
        return torch.zeros(input.nelement(), num_out)
    else:
        return type(input)(filter(lambda x: x is not None,
                                  (make_jacobian(elem, num_out) for elem in input)))


def iter_tensors(x, only_requiring_grad=False):
    if torch.is_tensor(x):
        yield x
    elif isinstance(x, Variable):
        if x.requires_grad or not only_requiring_grad:
            yield x.data
    else:
        for elem in x:
            for result in iter_tensors(elem, only_requiring_grad):
                yield result


def contiguous(input):
    if torch.is_tensor(input):
        return input.contiguous()
    elif isinstance(input, Variable):
        return input.contiguous()
    else:
        return type(input)(contiguous(e) for e in input)


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
            outa.copy_(fn(input))
            flat_tensor[i] = orig + eps
            outb.copy_(fn(input))
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
        output.backward(grad_output, retain_variables=True)
        for jacobian_x, d_x in zip(jacobian, iter_gradients(input)):
            jacobian_x[:, i] = d_x

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

        numerical = get_numerical_jacobian(fn, inputs, inputs, eps)
        analytical = get_analytical_jacobian(_as_tuple(inputs), o)

        for a, n in zip(analytical, numerical):
            if not ((a - n).abs() <= (atol + rtol * n.abs())).all():
                return False
    return True
