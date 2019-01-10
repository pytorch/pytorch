import torch
from collections import Iterable
import torch.testing
import sys


def zero_gradients(x):
    if isinstance(x, torch.Tensor):
        if x.grad is not None:
            x.grad.detach_()
            x.grad.data.zero_()
    elif isinstance(x, Iterable):
        for elem in x:
            zero_gradients(elem)


def make_jacobian(input, num_out):
    if isinstance(input, torch.Tensor):
        if not input.is_floating_point():
            return None
        if not input.requires_grad:
            return None
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
    if isinstance(x, torch.Tensor):
        if x.requires_grad or not only_requiring_grad:
            yield x.data
    elif isinstance(x, Iterable):
        for elem in x:
            for result in iter_tensors(elem, only_requiring_grad):
                yield result


def iter_tensors_with_grad(x):
    if isinstance(x, torch.Tensor):
        if x.requires_grad:
            yield (x.grad.data, x.data) if x.grad is not None else (None, None)
    elif isinstance(x, Iterable):
        for elem in x:
            for result in iter_tensors_with_grad(elem):
                yield result


def contiguous(input):
    if isinstance(input, torch.Tensor):
        return input.contiguous()
    elif isinstance(input, Iterable):
        return type(input)(contiguous(e) for e in input)
    return input


def get_numerical_jacobian(fn, input, target, eps=1e-3):
    # To be able to use .view(-1) input must be contiguous
    input = contiguous(input)
    target = contiguous(target)
    output_size = fn(input).numel()
    jacobian = make_jacobian(target, output_size)

    # It's much easier to iterate over flattened lists of tensors.
    # These are reference to the same objects in jacobian, so any changes
    # will be reflected in it as well.
    x_tensors = [t for t in iter_tensors(target, True)]
    j_tensors = [t for t in iter_tensors(jacobian)]

    # TODO: compare structure
    for x_tensor, d_tensor in zip(x_tensors, j_tensors):
        flat_tensor = x_tensor.view(-1).detach()
        for i in range(flat_tensor.nelement()):
            orig = flat_tensor[i].item()
            flat_tensor[i] = orig - eps
            outa = fn(input).clone()
            flat_tensor[i] = orig + eps
            outb = fn(input).clone()
            flat_tensor[i] = orig

            r = (outb - outa) / (2 * eps)
            d_tensor[i] = r.detach().contiguous().view(-1)

    return jacobian


def get_analytical_jacobian(input, output):
    input = contiguous(input)
    jacobian = make_jacobian(input, output.numel())
    jacobian_reentrant = make_jacobian(input, output.numel())
    grad_output = torch.zeros_like(output)
    flat_grad_output = grad_output.view(-1)
    reentrant = True
    correct_grad_sizes = True

    for i in range(flat_grad_output.numel()):
        flat_grad_output.zero_()
        flat_grad_output[i] = 1
        for jacobian_c in (jacobian, jacobian_reentrant):
            zero_gradients(input)
            output.backward(grad_output, create_graph=True)
            for jacobian_x, (d_x, x) in zip(jacobian_c, iter_tensors_with_grad(input)):
                if d_x is not None and d_x.size() != x.size():
                    correct_grad_sizes = False
                elif jacobian_x.numel() != 0:
                    if d_x is None:
                        jacobian_x[:, i].zero_()
                    else:
                        d_x_dense = d_x.to_dense() if d_x.is_sparse else d_x
                        assert jacobian_x[:, i].numel() == d_x_dense.numel()
                        jacobian_x[:, i] = d_x_dense.contiguous().view(-1)

    for jacobian_x, jacobian_reentrant_x in zip(jacobian, jacobian_reentrant):
        if jacobian_x.numel() != 0 and (jacobian_x - jacobian_reentrant_x).abs().max() != 0:
            reentrant = False

    return jacobian, reentrant, correct_grad_sizes


def _as_tuple(x):
    if isinstance(x, tuple):
        return x
    elif isinstance(x, list):
        return tuple(x)
    else:
        return x,


def _differentiable_outputs(x):
    return tuple(o for o in _as_tuple(x) if o.requires_grad)


def gradcheck(func, inputs, eps=1e-6, atol=1e-5, rtol=1e-3, raise_exception=True):
    """Check gradients computed via small finite differences
       against analytical gradients

    The check between numerical and analytical has the same behaviour as
    numpy.allclose https://docs.scipy.org/doc/numpy/reference/generated/numpy.allclose.html
    meaning it check that
        absolute(a - n) <= (atol + rtol * absolute(n))
    is true for all elements of analytical jacobian a and numerical jacobian n.

    Args:
        func: Python function that takes Tensor inputs and returns
            a Tensor or a tuple of Tensors
        inputs: tuple of Tensors
        eps: perturbation for finite differences
        atol: absolute tolerance
        rtol: relative tolerance
        raise_exception: bool indicating whether to raise an exception if
            gradcheck fails. The exception gives more information about the
            exact nature of the failure. This is helpful when debugging gradchecks.
    Returns:
        True if all differences satisfy allclose condition
    """
    tupled_inputs = _as_tuple(inputs)

    # Make sure that gradients are saved for all inputs
    for inp in tupled_inputs:
        if isinstance(inp, torch.Tensor):
            inp.retain_grad()

    output = _differentiable_outputs(func(*inputs))

    def fail_test(msg):
        if raise_exception:
            raise RuntimeError(msg)
        return False

    for i, o in enumerate(output):
        if not o.requires_grad:
            continue

        def fn(input):
            return _as_tuple(func(*input))[i].data

        analytical, reentrant, correct_grad_sizes = get_analytical_jacobian(tupled_inputs, o)
        numerical = get_numerical_jacobian(fn, inputs, inputs, eps)

        if not correct_grad_sizes:
            return fail_test('Analytical gradient has incorrect size')

        for j, (a, n) in enumerate(zip(analytical, numerical)):
            if a.numel() != 0 or n.numel() != 0:
                if not ((a - n).abs() <= (atol + rtol * n.abs())).all():
                    return fail_test('Jacobian mismatch for output %d with respect to input %d,\n'
                                     'numerical:%s\nanalytical:%s\n' % (i, j, n, a))

        if not reentrant:
            return fail_test('Backward is not reentrant, i.e., running backward with same '
                             'input and grad_output multiple times gives different values, '
                             'although analytical gradient matches numerical gradient')

    # check if the backward multiplies by grad_output
    zero_gradients(inputs)
    output = _differentiable_outputs(func(*inputs))
    if any([o.requires_grad for o in output]):
        torch.autograd.backward(output, [torch.zeros_like(o) for o in output], create_graph=True)
        var_inputs = list(filter(lambda i: isinstance(i, torch.Tensor), inputs))
        if not var_inputs:
            raise RuntimeError("no Tensors found in input")
        for i in var_inputs:
            if i.grad is None:
                continue
            if not i.grad.data.eq(0).all():
                return fail_test('backward not multiplied by grad_output')
            if i.grad.type() != i.type():
                return fail_test("grad is incorrect type")
            if i.grad.size() != i.size():
                return fail_test('grad is incorrect size')

    return True


def gradgradcheck(func, inputs, grad_outputs=None, eps=1e-6, atol=1e-5, rtol=1e-3,
                  gen_non_contig_grad_outputs=False, raise_exception=True):
    """Check gradients of gradients computed via small finite differences
       against analytical gradients
    This function checks that backpropagating through the gradients computed
    to the given grad_outputs are correct.

    The check between numerical and analytical has the same behaviour as
    numpy.allclose https://docs.scipy.org/doc/numpy/reference/generated/numpy.allclose.html
    meaning it check that
        absolute(a - n) <= (atol + rtol * absolute(n))
    is true for all elements of analytical gradient a and numerical gradient n.

    Args:
        func (function): Python function that takes Tensor inputs and returns
            a Tensor or a tuple of Tensors
        inputs (tuple of Tensor): inputs to the function
        grad_outputs (tuple of Tensor, optional): The gradients with respect to
            the function's outputs.
        eps (float, optional): perturbation for finite differences
        atol (float, optional): absolute tolerance
        rtol (float, optional): relative tolerance
        gen_non_contig_grad_outputs (bool, optional): if :attr:`grad_outputs` is
            ``None`` and :attr:`gen_non_contig_grad_outputs` is ``True``, the
            randomly generated gradient outputs are made to be noncontiguous
        raise_exception: bool indicating whether to raise an exception if
            gradcheck fails. The exception gives more information about the
            exact nature of the failure. This is helpful when debugging gradchecks.

    Returns:
        True if all differences satisfy allclose condition. Raises an exception
        otherwise.
    """
    if grad_outputs is None:
        # If grad_outputs is not specified, create random Tensors of the same
        # shape, type, and device as the outputs
        def randn_like(x):
            var = torch.testing.randn_like(x if x.is_floating_point() else x.double())
            if gen_non_contig_grad_outputs:
                var = torch.testing.make_non_contiguous(var)
            var.requires_grad = True
            return var
        outputs = _as_tuple(func(*inputs))
        grad_outputs_gen = (randn_like(x) for x in outputs)
        grad_outputs = list(grad_outputs_gen) if not isinstance(inputs, tuple) else tuple(grad_outputs_gen)

    def new_func(*input_args):
        input_args = input_args[:-len(grad_outputs)]
        outputs = _differentiable_outputs(func(*input_args))
        input_args = tuple(x for x in input_args if isinstance(x, torch.Tensor) and x.requires_grad)
        grad_inputs = torch.autograd.grad(outputs, input_args, grad_outputs, create_graph=True)
        return grad_inputs

    return gradcheck(new_func, inputs + grad_outputs, eps, atol, rtol, raise_exception)
