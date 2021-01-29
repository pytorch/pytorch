import torch
from torch.types import _TensorOrTensors
from torch._six import container_abcs, istuple
import torch.testing
from torch.overrides import is_tensor_like
from itertools import product
import warnings
from typing import Callable, Union, Optional, Iterable, List
from torch._vmap_internals import vmap
import functools

def zero_gradients(x):
    if isinstance(x, torch.Tensor):
        if x.grad is not None:
            x.grad.detach_()
            x.grad.zero_()
    elif isinstance(x, container_abcs.Iterable):
        for elem in x:
            zero_gradients(elem)


def make_jacobian(input, num_out):
    if is_tensor_like(input):
        if not input.is_floating_point() and not input.is_complex():
            return None
        if not input.requires_grad:
            return None
        return input.new_zeros((input.nelement(), num_out), dtype=input.dtype, layout=torch.strided)
    elif isinstance(input, container_abcs.Iterable) and not isinstance(input, str):
        jacobians = list(filter(
            lambda x: x is not None, (make_jacobian(elem, num_out) for elem in input)))
        if not jacobians:
            return None
        return type(input)(jacobians)  # type: ignore
    else:
        return None


def iter_tensors(x: Union[torch.Tensor, Iterable[torch.Tensor]], only_requiring_grad: bool = False) -> Iterable[torch.Tensor]:
    if is_tensor_like(x):
        # mypy doesn't narrow type of `x` to torch.Tensor
        if x.requires_grad or not only_requiring_grad:  # type: ignore
            yield x  # type: ignore
    elif isinstance(x, container_abcs.Iterable) and not isinstance(x, str):
        for elem in x:
            for result in iter_tensors(elem, only_requiring_grad):
                yield result

def get_numerical_jacobian(fn, input, target=None, eps=1e-3, grad_out=1.0):
    """
    input: input to `fn`
    target: the Tensors wrt whom Jacobians are calculated (default=`input`)
    grad_out: grad output value used to calculate gradients.

    Note that `target` may not even be part of `input` to `fn`, so please be
    **very careful** in this to not clone `target`.
    """
    if target is None:
        target = input
    output_size = fn(input).numel()
    jacobian = make_jacobian(target, output_size)

    # It's much easier to iterate over flattened lists of tensors.
    # These are reference to the same objects in jacobian, so any changes
    # will be reflected in it as well.
    x_tensors = iter_tensors(target, True)
    j_tensors = iter_tensors(jacobian)

    def update_jacobians(x, idx, d, d_idx, is_mkldnn=False):

        # compute_jacobian only works for pure real
        # or pure imaginary delta
        def compute_gradient(delta):
            # we currently assume that the norm of delta equals eps
            assert(delta == eps or delta == (eps * 1j))

            def fn_out():
                if not is_mkldnn:
                    # x is a view into input and so this works
                    return fn(input).clone()
                else:
                    # convert the dense tensor back to have mkldnn layout
                    return fn([x.to_mkldnn()])

            orig = x[idx].item()
            x[idx] = orig - delta
            outa = fn_out()
            x[idx] = orig + delta
            outb = fn_out()
            x[idx] = orig
            r = (outb - outa) / (2 * eps)
            return r.detach().reshape(-1)

        # for details on the algorithm used here, refer:
        # Section 3.5.3 https://arxiv.org/pdf/1701.00392.pdf
        # s = fn(z) where z = x for real valued input
        # and z = x + yj for complex valued input
        ds_dx = compute_gradient(eps)
        if x.is_complex():  # C -> C, C -> R
            ds_dy = compute_gradient(eps * 1j)
            # conjugate wirtinger derivative
            conj_w_d = 0.5 * (ds_dx + ds_dy * 1j)
            # wirtinger derivative
            w_d = 0.5 * (ds_dx - ds_dy * 1j)
            d[d_idx] = grad_out.conjugate() * conj_w_d + grad_out * w_d.conj()
        elif ds_dx.is_complex():  # R -> C
            # w_d = conj_w_d = 0.5 * ds_dx
            # dL_dz_conj = 0.5 * [grad_out.conj() * ds_dx + grad_out * ds_dx.conj()]
            #            = 0.5 * [grad_out.conj() * ds_dx + (grad_out.conj() * ds_dx).conj()]
            #            = 0.5 * 2 * real(grad_out.conj() * ds_dx)
            #            = real(grad_out.conj() * ds_dx)
            d[d_idx] = torch.real(grad_out.conjugate() * ds_dx)
        else:   # R -> R
            d[d_idx] = ds_dx * grad_out

    # TODO: compare structure
    for x_tensor, d_tensor in zip(x_tensors, j_tensors):
        if x_tensor.is_sparse:
            def get_stride(size):
                dim = len(size)
                tmp = 1
                stride = [0] * dim
                for i in reversed(range(dim)):
                    stride[i] = tmp
                    tmp *= size[i]
                return stride

            x_nnz = x_tensor._nnz()
            x_size = list(x_tensor.size())
            x_indices = x_tensor._indices().t()
            x_values = x_tensor._values()
            x_stride = get_stride(x_size)

            # Use .data here to get around the version check
            x_values = x_values.data

            for i in range(x_nnz):
                x_value = x_values[i]
                for x_idx in product(*[range(m) for m in x_values.size()[1:]]):
                    indices = x_indices[i].tolist() + list(x_idx)
                    d_idx = sum(indices[k] * x_stride[k] for k in range(len(x_size)))
                    update_jacobians(x_value, x_idx, d_tensor, d_idx)
        elif x_tensor.layout == torch._mkldnn:  # type: ignore
            # Use .data here to get around the version check
            x_tensor = x_tensor.data
            if len(input) != 1:
                raise ValueError('gradcheck currently only supports functions with 1 input, but got: ',
                                 len(input))
            for d_idx, x_idx in enumerate(product(*[range(m) for m in x_tensor.size()])):
                # this is really inefficient, but without indexing implemented, there's
                # not really a better way than converting back and forth
                x_tensor_dense = x_tensor.to_dense()
                update_jacobians(x_tensor_dense, x_idx, d_tensor, d_idx, is_mkldnn=True)
        else:
            # Use .data here to get around the version check
            x_tensor = x_tensor.data
            for d_idx, x_idx in enumerate(product(*[range(m) for m in x_tensor.size()])):
                update_jacobians(x_tensor, x_idx, d_tensor, d_idx)

    return jacobian


def get_analytical_jacobian(input, output, nondet_tol=0.0, grad_out=1.0):
    # it is easier to call to_dense() on the sparse output than
    # to modify analytical jacobian
    if output.is_sparse:
        raise ValueError('Sparse output is not supported at gradcheck yet. '
                         'Please call to_dense() on the output of fn for gradcheck.')
    if output.layout == torch._mkldnn:  # type: ignore
        raise ValueError('MKLDNN output is not supported at gradcheck yet. '
                         'Please call to_dense() on the output of fn for gradcheck.')
    diff_input_list = list(iter_tensors(input, True))
    jacobian = make_jacobian(input, output.numel())
    jacobian_reentrant = make_jacobian(input, output.numel())
    grad_output = torch.zeros_like(output, memory_format=torch.legacy_contiguous_format)
    flat_grad_output = grad_output.view(-1)
    reentrant = True
    correct_grad_sizes = True
    correct_grad_types = True

    for i in range(flat_grad_output.numel()):
        flat_grad_output.zero_()
        flat_grad_output[i] = grad_out
        for jacobian_c in (jacobian, jacobian_reentrant):
            grads_input = torch.autograd.grad(output, diff_input_list, grad_output,
                                              retain_graph=True, allow_unused=True)
            for jacobian_x, d_x, x in zip(jacobian_c, grads_input, diff_input_list):
                if d_x is not None and d_x.size() != x.size():
                    correct_grad_sizes = False
                elif d_x is not None and d_x.dtype != x.dtype:
                    correct_grad_types = False
                elif jacobian_x.numel() != 0:
                    if d_x is None:
                        jacobian_x[:, i].zero_()
                    else:
                        d_x_dense = d_x.to_dense() if not d_x.layout == torch.strided else d_x
                        assert jacobian_x[:, i].numel() == d_x_dense.numel()
                        jacobian_x[:, i] = d_x_dense.contiguous().view(-1)

    for jacobian_x, jacobian_reentrant_x in zip(jacobian, jacobian_reentrant):
        if jacobian_x.numel() != 0 and (jacobian_x - jacobian_reentrant_x).abs().max() > nondet_tol:
            reentrant = False

    return jacobian, reentrant, correct_grad_sizes, correct_grad_types

FAILED_BATCHED_GRAD_MSG = """
gradcheck or gradgradcheck failed while testing batched gradient computation.
This could have been invoked in a number of ways (via a test that calls
gradcheck/gradgradcheck directly or via an autogenerated test).

If you are adding a new operator, please file an issue and then use one of the
workarounds. The workaround depends on how your test invokes gradcheck/gradgradcheck.
If the test
- manually invokes gradcheck/gradgradcheck, then call gradcheck/gradgradcheck
  with `check_batched_grad=False` as a keyword argument.
- is OpInfo-based (e.g., in test_ops.py), then modify the OpInfo for the test
  to have `check_batched_grad=False` and/or `check_batched_gradgrad=False`.
- is common_method_invocations-based, then add your test to the denylist
  EXCLUDE_BATCHED_GRAD_TESTS in test_autograd.py

If you're modifying an existing operator that supports batched grad computation,
or wish to make a new operator work with batched grad computation, please read
the following.

To compute batched grads (e.g., jacobians, hessians), we vmap over the backward
computation. The most common failure case is if there is a 'vmap-incompatible
operation' in the backward pass. Please see
NOTE: [How to write vmap-compatible backward formulas]
in the codebase for an explanation of how to fix this.
""".strip()

def get_failed_batched_grad_test_msg(output_idx, input_idx, res, exp):
    return f"""
For output {output_idx} and input {input_idx}:

{FAILED_BATCHED_GRAD_MSG}

Got:
{res}

Expected:
{exp}
""".strip()

def test_batched_grad(fail_test, input, output, output_idx):
    diff_input_list = list(iter_tensors(input, True))
    grad = functools.partial(torch.autograd.grad, output, diff_input_list, retain_graph=True, allow_unused=True)

    def vjp(v):
        results = grad(v)
        results = tuple(grad if grad is not None else
                        torch.zeros([], dtype=inp.dtype, device=inp.device).expand(inp.shape)
                        for grad, inp in zip(results, diff_input_list))
        return results

    grad_outputs = [torch.randn_like(output) for _ in range(2)]

    expected = [vjp(gO) for gO in grad_outputs]
    expected = [torch.stack(shards) for shards in zip(*expected)]

    # Squash warnings since these are expected to happen in most cases
    # NB: this doesn't work for CUDA tests: https://github.com/pytorch/pytorch/issues/50209
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Batching rule not implemented")
        warnings.filterwarnings("ignore", message="torch.vmap is an experimental prototype")
        try:
            result = vmap(vjp)(torch.stack(grad_outputs))
        except RuntimeError as ex:
            # It's OK that we're not raising the error at the correct callsite.
            # That's because the callsite is always going to inside the Python
            # autograd.grad instead of the C++ traceback of what line in the
            # backward formula
            return fail_test(
                f'While computing batched gradients, got: {ex}\n\n{FAILED_BATCHED_GRAD_MSG}')

    for input_idx, (res, exp) in enumerate(zip(result, expected)):
        if torch.allclose(res, exp):
            continue
        return fail_test(get_failed_batched_grad_test_msg(output_idx, input_idx, res, exp))


def _as_tuple(x):
    if istuple(x):
        return x
    elif isinstance(x, list):
        return tuple(x)
    else:
        return x,


def _differentiable_outputs(x):
    return tuple(o for o in _as_tuple(x) if o.requires_grad)


# Note [VarArg of Tensors]
# ~~~~~~~~~~~~~~~~~~~~~~~~
# 'func' accepts a vararg of tensors, which isn't expressable in the type system at the moment.
# If https://mypy.readthedocs.io/en/latest/additional_features.html?highlight=callable#extended-callable-types is accepted,
# the '...' first argument of Callable can be replaced with VarArg(Tensor).
# For now, we permit any input.
# the '...' first argument of Callable can be replaced with VarArg(Tensor).
# For now, we permit any input.

def gradcheck(
    func: Callable[..., Union[_TensorOrTensors]],  # See Note [VarArg of Tensors]
    inputs: _TensorOrTensors,
    eps: float = 1e-6,
    atol: float = 1e-5,
    rtol: float = 1e-3,
    raise_exception: bool = True,
    check_sparse_nnz: bool = False,
    nondet_tol: float = 0.0,
    check_undefined_grad: bool = True,
    check_grad_dtypes: bool = False,
    check_batched_grad: bool = False,
) -> bool:
    r"""Check gradients computed via small finite differences against analytical
    gradients w.r.t. tensors in :attr:`inputs` that are of floating point or complex type
    and with ``requires_grad=True``.

    The check between numerical and analytical gradients uses :func:`~torch.allclose`.

    For complex functions, no notion of Jacobian exists. Gradcheck verifies if the numerical and
    analytical values of Wirtinger and Conjugate Wirtinger derivative are consistent. The gradient
    computation is done under the assumption that the overall function has a real valued output.
    For functions with complex output, gradcheck compares the numerical and analytical gradients
    for two values of :attr:`grad_output`: 1 and 1j. For more details, check out
    :ref:`complex_autograd-doc`.

    .. note::
        The default values are designed for :attr:`input` of double precision.
        This check will likely fail if :attr:`input` is of less precision, e.g.,
        ``FloatTensor``.

    .. warning::
       If any checked tensor in :attr:`input` has overlapping memory, i.e.,
       different indices pointing to the same memory address (e.g., from
       :func:`torch.expand`), this check will likely fail because the numerical
       gradients computed by point perturbation at such indices will change
       values at all other indices that share the same memory address.

    Args:
        func (function): a Python function that takes Tensor inputs and returns
            a Tensor or a tuple of Tensors
        inputs (tuple of Tensor or Tensor): inputs to the function
        eps (float, optional): perturbation for finite differences
        atol (float, optional): absolute tolerance
        rtol (float, optional): relative tolerance
        raise_exception (bool, optional): indicating whether to raise an exception if
            the check fails. The exception gives more information about the
            exact nature of the failure. This is helpful when debugging gradchecks.
        check_sparse_nnz (bool, optional): if True, gradcheck allows for SparseTensor input,
            and for any SparseTensor at input, gradcheck will perform check at nnz positions only.
        nondet_tol (float, optional): tolerance for non-determinism. When running
            identical inputs through the differentiation, the results must either match
            exactly (default, 0.0) or be within this tolerance.
        check_undefined_grad (bool, optional): if True, check if undefined output grads
            are supported and treated as zeros, for ``Tensor`` outputs.
        check_batched_grad (bool, optional): if True, check if we can compute
            batched gradients using prototype vmap support. Defaults to False.

    Returns:
        True if all differences satisfy allclose condition
    """
    def fail_test(msg):
        if raise_exception:
            raise RuntimeError(msg)
        return False

    tupled_inputs = _as_tuple(inputs)
    if not check_sparse_nnz and any(t.is_sparse for t in tupled_inputs if isinstance(t, torch.Tensor)):
        return fail_test('gradcheck expects all tensor inputs are dense when check_sparse_nnz is set to False.')

    # Make sure that gradients are saved for at least one input
    any_input_requiring_grad = False
    for idx, inp in enumerate(tupled_inputs):
        if is_tensor_like(inp) and inp.requires_grad:
            if not (inp.dtype == torch.float64 or inp.dtype == torch.complex128):
                warnings.warn(
                    f'Input #{idx} requires gradient and '
                    'is not a double precision floating point or complex. '
                    'This check will likely fail if all the inputs are '
                    'not of double precision floating point or complex. ')
            content = inp._values() if inp.is_sparse else inp
            # TODO: To cover more problematic cases, replace stride = 0 check with
            # "any overlap in memory" once we have a proper function to check it.
            if content.layout is not torch._mkldnn:  # type: ignore
                if not all(st > 0 or sz <= 1 for st, sz in zip(content.stride(), content.size())):
                    raise RuntimeError(
                        'The {}th input has a dimension with stride 0. gradcheck only '
                        'supports inputs that are non-overlapping to be able to '
                        'compute the numerical gradients correctly. You should call '
                        '.contiguous on the input before passing it to gradcheck.')
            any_input_requiring_grad = True
            inp.retain_grad()
    if not any_input_requiring_grad:
        raise ValueError(
            'gradcheck expects at least one input tensor to require gradient, '
            'but none of the them have requires_grad=True.')

    func_out = func(*tupled_inputs)
    output = _differentiable_outputs(func_out)

    if not output:
        for i, o in enumerate(func_out):
            def fn(input):
                return _as_tuple(func(*input))[i]
            numerical = get_numerical_jacobian(fn, tupled_inputs, eps=eps)
            for n in numerical:
                if torch.ne(n, 0).sum() > 0:
                    return fail_test('Numerical gradient for function expected to be zero')
        return True

    for i, o in enumerate(output):
        if not o.requires_grad:
            continue

        def fn(input):
            return _as_tuple(func(*input))[i]

        analytical, reentrant, correct_grad_sizes, correct_grad_types = get_analytical_jacobian(tupled_inputs,
                                                                                                o,
                                                                                                nondet_tol=nondet_tol)
        numerical = get_numerical_jacobian(fn, tupled_inputs, eps=eps)

        out_is_complex = o.is_complex()

        if out_is_complex:
            # analytical vjp with grad_out = 1.0j
            analytical_with_imag_grad_out, reentrant_with_imag_grad_out, \
                correct_grad_sizes_with_imag_grad_out, correct_grad_types_with_imag_grad_out \
                = get_analytical_jacobian(tupled_inputs, o, nondet_tol=nondet_tol, grad_out=1j)
            numerical_with_imag_grad_out = get_numerical_jacobian(fn, tupled_inputs, eps=eps, grad_out=1j)

        if not correct_grad_types and check_grad_dtypes:
            return fail_test('Gradient has dtype mismatch')

        if out_is_complex and not correct_grad_types_with_imag_grad_out and check_grad_dtypes:
            return fail_test('Gradient (calculated using complex valued grad output) has dtype mismatch')

        if not correct_grad_sizes:
            return fail_test('Analytical gradient has incorrect size')

        if out_is_complex and not correct_grad_sizes_with_imag_grad_out:
            return fail_test('Analytical gradient (calculated using complex valued grad output) has incorrect size')

        def checkIfNumericalAnalyticAreClose(a, n, j, error_str=''):
            if not torch.allclose(a, n, rtol, atol):
                return fail_test(error_str + 'Jacobian mismatch for output %d with respect to input %d,\n'
                                 'numerical:%s\nanalytical:%s\n' % (i, j, n, a))

        inp_tensors = iter_tensors(tupled_inputs, True)

        for j, (a, n, inp) in enumerate(zip(analytical, numerical, inp_tensors)):
            if a.numel() != 0 or n.numel() != 0:
                if o.is_complex():
                    # C -> C, R -> C
                    a_with_imag_grad_out = analytical_with_imag_grad_out[j]
                    n_with_imag_grad_out = numerical_with_imag_grad_out[j]
                    checkIfNumericalAnalyticAreClose(a_with_imag_grad_out, n_with_imag_grad_out, j,
                                                     "Gradients failed to compare equal for grad output = 1j. ")
                if inp.is_complex():
                    # C -> R, C -> C
                    checkIfNumericalAnalyticAreClose(a, n, j,
                                                     "Gradients failed to compare equal for grad output = 1. ")
                else:
                    # R -> R, R -> C
                    checkIfNumericalAnalyticAreClose(a, n, j)


        def not_reentrant_error(error_str=''):
            error_msg = "Backward" + error_str + " is not reentrant, i.e., running backward with same \
                        input and grad_output multiple times gives different values, \
                        although analytical gradient matches numerical gradient. \
                        The tolerance for nondeterminism was {}.".format(nondet_tol)
            return fail_test(error_msg)

        if not reentrant:
            return not_reentrant_error()

        if out_is_complex and not reentrant_with_imag_grad_out:
            return not_reentrant_error(' (calculated using complex valued grad output)')

        if check_batched_grad:
            assert reentrant, ('Batched gradient checking makes the assumption that '
                               'backward is reentrant. This assertion should never '
                               'be triggered: we expect gradcheck to have early '
                               'exited before reaching this point if backward is '
                               'not reentrant. Please file us a bug report.')
            # NB: test_batched_grad compares two autograd.grad invocations with a single
            # vmap(autograd.grad) invocation. It's not exactly a "gradcheck" in the
            # sense that we're not comparing an analytical jacobian with a numeric one,
            # but it is morally similar (we could have computed a full analytic jac
            # via vmap, but that is potentially slow)
            test_batched_grad(fail_test, tupled_inputs, o, j)

    # check if the backward multiplies by grad_output
    output = _differentiable_outputs(func(*tupled_inputs))
    if any([o.requires_grad for o in output]):
        diff_input_list: List[torch.Tensor] = list(iter_tensors(tupled_inputs, True))
        if not diff_input_list:
            raise RuntimeError("no Tensors requiring grad found in input")
        grads_input = torch.autograd.grad(output, diff_input_list,
                                          [torch.zeros_like(o, memory_format=torch.legacy_contiguous_format) for o in output],
                                          allow_unused=True)
        for gi, di in zip(grads_input, diff_input_list):
            if gi is None:
                continue
            if isinstance(gi, torch.Tensor) and gi.layout != torch.strided:
                if gi.layout != di.layout:
                    return fail_test('grad is incorrect layout (' + str(gi.layout) + ' is not ' + str(di.layout) + ')')
                if gi.layout == torch.sparse_coo:
                    if gi.sparse_dim() != di.sparse_dim():
                        return fail_test('grad is sparse tensor, but has incorrect sparse_dim')
                    if gi.dense_dim() != di.dense_dim():
                        return fail_test('grad is sparse tensor, but has incorrect dense_dim')
                gi = gi.to_dense()
                di = di.to_dense()

            if check_sparse_nnz:
                if not torch.allclose(gi, torch.zeros_like(gi)):
                    return fail_test('backward not multiplied by grad_output')
            elif not gi.eq(0).all():
                return fail_test('backward not multiplied by grad_output')
            if gi.dtype != di.dtype or gi.device != di.device or gi.is_sparse != di.is_sparse:
                return fail_test("grad is incorrect type")
            if gi.size() != di.size():
                return fail_test('grad is incorrect size')

        if check_undefined_grad:
            def warn_bc_breaking():
                warnings.warn((
                    'Backwards compatibility: New undefined gradient support checking '
                    'feature is enabled by default, but it may break existing callers '
                    'of this function. If this is true for you, you can call this '
                    'function with "check_undefined_grad=False" to disable the feature'))

            def check_undefined_grad_support(output_to_check):
                grads_output = [torch.zeros_like(o, memory_format=torch.legacy_contiguous_format) for o in output_to_check]
                try:
                    grads_input = torch.autograd.grad(output_to_check,
                                                      diff_input_list,
                                                      grads_output,
                                                      allow_unused=True)
                except RuntimeError:
                    warn_bc_breaking()
                    return fail_test((
                        'Expected backward function to handle undefined output grads. '
                        'Please look at "Notes about undefined output gradients" in '
                        '"tools/autograd/derivatives.yaml"'))

                for gi, i in zip(grads_input, diff_input_list):
                    if (gi is not None) and (not gi.eq(0).all()):
                        warn_bc_breaking()
                        return fail_test((
                            'Expected all input grads to be undefined or zero when all output grads are undefined '
                            'or zero. Please look at "Notes about undefined output gradients" in '
                            '"tools/autograd/derivatives.yaml"'))
                return True

            # All backward functions must work properly if all output grads are undefined
            outputs_to_check = [[
                torch._C._functions.UndefinedGrad()(o) for o in _differentiable_outputs(func(*tupled_inputs))
                # This check filters out Tensor-likes that aren't instances of Tensor.
                if isinstance(o, torch.Tensor)
            ]]

            # If there are multiple output grads, we should be able to undef one at a time without error
            if len(outputs_to_check[0]) > 1:
                for undef_grad_idx in range(len(output)):
                    output_to_check = _differentiable_outputs(func(*tupled_inputs))
                    outputs_to_check.append([
                        torch._C._functions.UndefinedGrad()(o) if idx == undef_grad_idx else o
                        for idx, o in enumerate(output_to_check)])

            for output_to_check in outputs_to_check:
                if not check_undefined_grad_support(output_to_check):
                    return False

    return True


def gradgradcheck(
    func: Callable[..., _TensorOrTensors],  # See Note [VarArg of Tensors]
    inputs: _TensorOrTensors,
    grad_outputs: Optional[_TensorOrTensors] = None,
    eps: float = 1e-6,
    atol: float = 1e-5,
    rtol: float = 1e-3,
    gen_non_contig_grad_outputs: bool = False,
    raise_exception: bool = True,
    nondet_tol: float = 0.0,
    check_undefined_grad: bool = True,
    check_grad_dtypes: bool = False,
    check_batched_grad: bool = False,
) -> bool:
    r"""Check gradients of gradients computed via small finite differences
    against analytical gradients w.r.t. tensors in :attr:`inputs` and
    :attr:`grad_outputs` that are of floating point or complex type and with
    ``requires_grad=True``.

    This function checks that backpropagating through the gradients computed
    to the given :attr:`grad_outputs` are correct.

    The check between numerical and analytical gradients uses :func:`~torch.allclose`.

    .. note::
        The default values are designed for :attr:`input` and
        :attr:`grad_outputs` of double precision. This check will likely fail if
        they are of less precision, e.g., ``FloatTensor``.

    .. warning::
       If any checked tensor in :attr:`input` and :attr:`grad_outputs` has
       overlapping memory, i.e., different indices pointing to the same memory
       address (e.g., from :func:`torch.expand`), this check will likely fail
       because the numerical gradients computed by point perturbation at such
       indices will change values at all other indices that share the same
       memory address.

    Args:
        func (function): a Python function that takes Tensor inputs and returns
            a Tensor or a tuple of Tensors
        inputs (tuple of Tensor or Tensor): inputs to the function
        grad_outputs (tuple of Tensor or Tensor, optional): The gradients with
            respect to the function's outputs.
        eps (float, optional): perturbation for finite differences
        atol (float, optional): absolute tolerance
        rtol (float, optional): relative tolerance
        gen_non_contig_grad_outputs (bool, optional): if :attr:`grad_outputs` is
            ``None`` and :attr:`gen_non_contig_grad_outputs` is ``True``, the
            randomly generated gradient outputs are made to be noncontiguous
        raise_exception (bool, optional): indicating whether to raise an exception if
            the check fails. The exception gives more information about the
            exact nature of the failure. This is helpful when debugging gradchecks.
        nondet_tol (float, optional): tolerance for non-determinism. When running
            identical inputs through the differentiation, the results must either match
            exactly (default, 0.0) or be within this tolerance. Note that a small amount
            of nondeterminism in the gradient will lead to larger inaccuracies in
            the second derivative.
        check_undefined_grad (bool, optional): if True, check if undefined output grads
            are supported and treated as zeros
        check_batched_grad (bool, optional): if True, check if we can compute
            batched gradients using prototype vmap support. Defaults to False.

    Returns:
        True if all differences satisfy allclose condition
    """
    tupled_inputs = _as_tuple(inputs)

    if grad_outputs is None:
        # If grad_outputs is not specified, create random Tensors of the same
        # shape, type, and device as the outputs
        def randn_like(x):
            y = torch.testing.randn_like(
                x if (x.is_floating_point() or x.is_complex()) else x.double(), memory_format=torch.legacy_contiguous_format)
            if gen_non_contig_grad_outputs:
                y = torch.testing.make_non_contiguous(y)
            return y.requires_grad_()
        outputs = _as_tuple(func(*tupled_inputs))
        tupled_grad_outputs = tuple(randn_like(x) for x in outputs)
    else:
        tupled_grad_outputs = _as_tuple(grad_outputs)

    num_outputs = len(tupled_grad_outputs)

    def new_func(*args):
        input_args = args[:-num_outputs]
        grad_outputs = args[-num_outputs:]
        outputs = _differentiable_outputs(func(*input_args))
        input_args = tuple(x for x in input_args if isinstance(x, torch.Tensor) and x.requires_grad)
        grad_inputs = torch.autograd.grad(outputs, input_args, grad_outputs, create_graph=True)
        return grad_inputs

    return gradcheck(
        new_func, tupled_inputs + tupled_grad_outputs, eps, atol, rtol, raise_exception,
        nondet_tol=nondet_tol, check_undefined_grad=check_undefined_grad,
        check_grad_dtypes=check_grad_dtypes, check_batched_grad=check_batched_grad)
