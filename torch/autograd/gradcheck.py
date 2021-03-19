import torch
from torch.types import _TensorOrTensors
import torch.testing
from torch.overrides import is_tensor_like
import collections
from itertools import product
import warnings
from typing import Callable, Union, Optional, Iterable, List, Dict, Tuple
from torch._vmap_internals import vmap
import functools

def zero_gradients(x):
    if isinstance(x, torch.Tensor):
        if x.grad is not None:
            x.grad.detach_()
            x.grad.zero_()
    elif isinstance(x, collections.abc.Iterable):
        for elem in x:
            zero_gradients(elem)


def is_float_or_complex_tensor(obj):
    return isinstance(obj, torch.Tensor) and (obj.is_floating_point() or obj.is_complex())


def make_jacobians(tensors, tensors_is_inputs, dim=None, dtype=None):
    """make_jacobians makes zero-filled tensors from inputs/outputs to be filled row-by-row
     or col-by-col. If tensors is inputs, for each tensor, returns a new zero-filled tensor
    with height of `t.numel` and width of `dim`. Otherwise, the new tensor will have height
     of `dim` and width of `t.numel`.

    NOTE: this is actually the *tranpose* of the jacobian. There is no particular reason we
    favor the transpose. A possible todo is to "untranspose" what this fn generates.

    If dim is None, the returned zero-filled tensor will be 1-d and have size (t.numel,).
    You can think of this as representing a single row or column of the entire jacobian.
    This is to be used by the fast version of gradcheck.
    """
    out: List[torch.Tensor] = []
    assert isinstance(tensors, tuple)
    for t in tensors:
        if is_float_or_complex_tensor(t) and (t.requires_grad or not tensors_is_inputs):
            dt = t.dtype if dtype is None else dtype
            if dim is None:
                out.append(t.new_zeros((t.nelement(),), dtype=dt, layout=torch.strided))
            elif tensors_is_inputs:
                out.append(t.new_zeros((t.nelement(), dim), dtype=dt, layout=torch.strided))
            else:
                out.append(t.new_zeros((dim, t.nelement()), dtype=dt, layout=torch.strided))
    if not out:
        return None
    return tuple(out)


def iter_tensors(x: Union[torch.Tensor, Iterable[torch.Tensor]], only_requiring_grad: bool = False) -> Iterable[torch.Tensor]:
    if is_tensor_like(x):
        # mypy doesn't narrow type of `x` to torch.Tensor
        if x.requires_grad or not only_requiring_grad:  # type: ignore
            yield x  # type: ignore
    elif isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
        for elem in x:
            for result in iter_tensors(elem, only_requiring_grad):
                yield result


def iter_tensor(x_tensor):
    """Iteration over different types of tensors: sparse, dense, and mkldnn.
    Provides a dense view of the original tensor, current index into that tensor, as well as
    a corresponding "flat index" which translates to a given row/col in the jacobian matrix.
    The provided "view" is obtained from `.data` so to avoid the version counter bump.
    """
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
                yield (x_value, x_idx, d_idx)
    elif x_tensor.layout == torch._mkldnn:  # type: ignore
        # Use .data here to get around the version check
        x_tensor = x_tensor.data
        for d_idx, x_idx in enumerate(product(*[range(m) for m in x_tensor.size()])):
            # this is really inefficient, but without indexing implemented, there's
            # not really a better way than converting back and forth
            x_tensor_dense = x_tensor.to_dense()
            yield (x_tensor_dense, x_idx, d_idx)
    else:
        # Use .data here to get around the version check
        x_tensor = x_tensor.data
        for d_idx, x_idx in enumerate(product(*[range(m) for m in x_tensor.size()])):
            yield (x_tensor, x_idx, d_idx)


def get_numerical_jacobian(fn, inputs, outputs=None, target=None, eps=1e-3, grad_out=1.0):
    """Computes the numerical jacobian for a given fn and inputs. Outputs can be provided
    to avoid one extra invocation of fn. Returns M * N jacobians where M is the number of
    input tensors that require grad, and N is the number of output float/complex tensors.

    input: input to `fn`
    target: the Tensors wrt whom Jacobians are calculated (default=`input`)
    grad_out: grad output value used to calculate gradients.

    Note that `target` may not even be part of `input` to `fn`, so please be
    **very careful** in this to not clone `target`.
    """
    jacobians: List[Tuple[torch.Tensor]] = []
    if outputs is None:
        outputs = fn(inputs)
    if target is None:
        target = inputs
    for i, inp in enumerate(iter_tensors(target, True)):
        jacobians += [get_numerical_jacobian_helper(fn, inp, inputs, outputs, eps, grad_out)]
    return jacobians


def get_numerical_jacobian_helper(fn, input, inputs, outputs, eps, grad_out):
    """Computes the numerical jacobians wrt to a single input. Returns N jacobian
    tensors, where N is the number of outputs. Input must require grad.
    """
    assert input.requires_grad
    jacobians = make_jacobians(outputs, False, input.numel(), input.dtype)

    def update_jacobians(x, idx, d_idx, is_mkldnn=False):
        # compute_jacobian only works for pure real
        # or pure imaginary delta
        def compute_gradient(delta):
            # we currently assume that the norm of delta equals eps
            assert(delta == eps or delta == (eps * 1j))

            def fn_out():
                if is_mkldnn:
                    inp = [x.to_mkldnn()]
                else:
                    inp = inputs
                return tuple(a.clone() for a in _as_tuple(fn(*inp)))

            orig = x[idx].item()
            x[idx] = orig - delta
            outa = fn_out()
            x[idx] = orig + delta
            outb = fn_out()
            x[idx] = orig

            def compute(a, b):
                ret = (b - a) / (2 * eps)
                return ret.detach().reshape(-1)

            return tuple(compute(a, b) for (a, b) in zip(outa, outb))

        # for details on the algorithm used here, refer:
        # Section 3.5.3 https://arxiv.org/pdf/1701.00392.pdf
        # s = fn(z) where z = x for real valued input
        # and z = x + yj for complex valued input
        ds_dx_tup = compute_gradient(eps)
        if x.is_complex():  # C -> C, C -> R
            ds_dy_tup = compute_gradient(eps * 1j)

            for ds_dx, ds_dy, d in zip(ds_dx_tup, ds_dy_tup, jacobians):
                # conjugate wirtinger derivative
                conj_w_d = 0.5 * (ds_dx + ds_dy * 1j)
                # wirtinger derivative
                w_d = 0.5 * (ds_dx - ds_dy * 1j)
                d[d_idx] = grad_out.conjugate() * conj_w_d + grad_out * w_d.conj()
        else:
            for ds_dx, d in zip(ds_dx_tup, jacobians):
                if ds_dx.is_complex():  # R -> C
                    # w_d = conj_w_d = 0.5 * ds_dx
                    # dL_dz_conj = 0.5 * [grad_out.conj() * ds_dx + grad_out * ds_dx.conj()]
                    #            = 0.5 * [grad_out.conj() * ds_dx + (grad_out.conj() * ds_dx).conj()]
                    #            = 0.5 * 2 * real(grad_out.conj() * ds_dx)
                    #            = real(grad_out.conj() * ds_dx)
                    d[d_idx] = torch.real(grad_out.conjugate() * ds_dx)
                else:   # R -> R
                    d[d_idx] = ds_dx * grad_out

    for x_tensor, x_idx, flat_idx in iter_tensor(input):
        is_mkldnn = x_tensor.layout == torch._mkldnn  # type: ignore # mypy: torch nas no attr _mkldnn
        if is_mkldnn and len(input) != 1:
            raise ValueError('gradcheck currently only supports functions with 1 input, but got: ',
                             len(input))
        update_jacobians(x_tensor, x_idx, flat_idx, is_mkldnn=is_mkldnn)

    return jacobians


def check_jacobians_equal(j1, j2, atol):
    # Check whether the max diff betwen two jacobians are within some tolerance `atol`
    for j1_x, j2_x in zip(j1, j2):
        if j1_x.numel() != 0 and (j1_x - j2_x).abs().max() > atol:
            return False
    return True


def combine_jacobian_rows(jacobians_rows, inputs, output):
    out_jacobians = make_jacobians(inputs, True, output.numel())
    diff_input_list = list(iter_tensors(inputs, True))
    correct_grad_sizes = True
    correct_grad_types = True
    for i, rows in jacobians_rows.items():
        inp = diff_input_list[i]
        out_jacobian = out_jacobians[i]
        for j, row in enumerate(rows):
            if row is not None and row.size() != inp.size():
                correct_grad_sizes = False
            elif row is not None and row.dtype != inp.dtype:
                correct_grad_types = False
            if row is None:
                out_jacobian[:, j].zero_()
            else:
                row_dense = row.to_dense() if not row.layout == torch.strided else row
                assert out_jacobian[:, j].numel() == row_dense.numel()
                out_jacobian[:, j] = row_dense.contiguous().view(-1)
    return out_jacobians, correct_grad_sizes, correct_grad_types


def check_analytical_jacobian_attributes(inputs, output, nondet_tol, grad_out_scale, check_grad_dtypes,
                                         raise_exception, custom_backward_fn=None):
    diff_input_list = list(iter_tensors(inputs, True))

    def backward_fn(grad_output):
        return torch.autograd.grad(output, diff_input_list, grad_output,
                                   retain_graph=True, allow_unused=True)
    fn = custom_backward_fn if custom_backward_fn is not None else backward_fn
    jacobians_rows = get_analytical_jacobian(fn, output.clone(), grad_out_scale)
    jacobians_rows_reentrant = get_analytical_jacobian(fn, output.clone(), grad_out_scale)

    jacobians, correct_grad_types, correct_grad_sizes = combine_jacobian_rows(jacobians_rows, inputs, output)
    jacobians_reentrant, _, _ = combine_jacobian_rows(jacobians_rows_reentrant, inputs, output)

    reentrant = check_jacobians_equal(jacobians, jacobians_reentrant, nondet_tol)

    complex_str = '(calculated using complex valued grad output) ' \
        if isinstance(grad_out_scale, complex) else ''

    def fail_test(msg):
        if raise_exception:
            raise RuntimeError(msg)

    if not correct_grad_types and check_grad_dtypes:
        fail_test(f'Gradient{complex_str} has dtype mismatch')
    if not correct_grad_sizes:
        fail_test(f'Analytical gradient{complex_str} has incorrect size')
    if not reentrant:
        fail_test(f'Backward{complex_str} is not reentrant, i.e., running backward with '
                  'same input and grad_output multiple times gives different values, '
                  'although analytical gradient matches numerical gradient. '
                  f'The tolerance for nondeterminism was {nondet_tol}.')
    failed = not (reentrant and correct_grad_sizes and correct_grad_types)
    return jacobians, failed


def get_analytical_jacobian(fn, sample_output, grad_out_scale):
    # Computes Jacobian row-by-row using backward function `fn` = v^T J
    # NB: we can't combine the rows into a single jacobian tensor because fn(v) for
    # different v may return tensors with different number of elements
    grad_out_base = torch.zeros_like(sample_output, memory_format=torch.legacy_contiguous_format)
    flat_grad_out = grad_out_base.view(-1)
    # jacobians_rows[i][j] represents the jth row of the ith input
    jacobians_rows: Dict[int, List[Optional[torch.Tensor]]] = {}

    for j in range(flat_grad_out.numel()):
        flat_grad_out.zero_()
        flat_grad_out[j] = grad_out_scale
        grad_inputs = fn(grad_out_base)
        for i, d_x in enumerate(grad_inputs):
            jacobians_rows[i] = jacobians_rows.get(i, []) + [d_x.clone() if isinstance(d_x, torch.Tensor) else None]
    return jacobians_rows


def check_inputs(fail_test, tupled_inputs, check_sparse_nnz) -> bool:
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
                        f'The {idx}th input has a dimension with stride 0. gradcheck only '
                        'supports inputs that are non-overlapping to be able to '
                        'compute the numerical gradients correctly. You should call '
                        '.contiguous on the input before passing it to gradcheck.')
            any_input_requiring_grad = True
            inp.retain_grad()
    if not any_input_requiring_grad:
        raise ValueError(
            'gradcheck expects at least one input tensor to require gradient, '
            'but none of the them have requires_grad=True.')
    return True


def check_outputs(outputs) -> None:
    if any(t.is_sparse for t in outputs if isinstance(t, torch.Tensor)):
        # it is easier to call to_dense() on the sparse output than
        # to modify analytical jacobian
        raise ValueError('Sparse output is not supported at gradcheck yet. '
                         'Please call to_dense() on the output of fn for gradcheck.')
    if any(t.layout == torch._mkldnn for t in outputs if isinstance(t, torch.Tensor)):  # type: ignore
        raise ValueError('MKLDNN output is not supported at gradcheck yet. '
                         'Please call to_dense() on the output of fn for gradcheck.')


def check_no_differentiable_outputs(fail_test, func, inputs, func_out, eps) -> bool:
    # When there are no differentiable outputs, numerical gradient for a function is
    # expected to be zero.
    jacobians_inputs_outputs = get_numerical_jacobian(func, inputs, func_out, eps=eps)
    for jacobian_inputs in jacobians_inputs_outputs:
        for jacobian in jacobian_inputs:
            if torch.ne(jacobian, 0).sum() > 0:
                return fail_test('Numerical gradient for function expected to be zero')
    return True


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


def test_batched_grad(fail_test, input, output, output_idx) -> bool:
    # NB: test_batched_grad compares two autograd.grad invocations with a single
    # vmap(autograd.grad) invocation. It's not exactly a "gradcheck" in the
    # sense that we're not comparing an analytical jacobian with a numeric one,
    # but it is morally similar (we could have computed a full analytic jac
    # via vmap, but that is potentially slow)
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
    return True


def test_backward_mul_by_grad_output(fail_test, outputs, inputs, check_sparse_nnz) -> bool:
    # Tests that backward is multiplied by grad_output
    diff_input_list: List[torch.Tensor] = list(iter_tensors(inputs, True))
    if not diff_input_list:
        raise RuntimeError("no Tensors requiring grad found in input")
    grads_input = torch.autograd.grad(outputs, diff_input_list,
                                      [torch.zeros_like(o, memory_format=torch.legacy_contiguous_format) for o in outputs],
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
    return True


def test_undefined_grad(fail_test, func, outputs, inputs) -> bool:
    diff_input_list: List[torch.Tensor] = list(iter_tensors(inputs, True))
    if not diff_input_list:
        raise RuntimeError("no Tensors requiring grad found in input")

    def warn_bc_breaking():
        warnings.warn((
            'Backwards compatibility: New undefined gradient support checking '
            'feature is enabled by default, but it may break existing callers '
            'of this function. If this is true for you, you can call this '
            'function with "check_undefined_grad=False" to disable the feature'))

    def check_undefined_grad_support(output_to_check):
        grads_output = [torch.zeros_like(o, memory_format=torch.legacy_contiguous_format) for o in output_to_check]
        try:
            grads_input = torch.autograd.grad(output_to_check, diff_input_list,
                                              grads_output, allow_unused=True)
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
        torch._C._functions.UndefinedGrad()(o) for o in _differentiable_outputs(func(*inputs))
        # This check filters out Tensor-likes that aren't instances of Tensor.
        if isinstance(o, torch.Tensor)
    ]]

    # If there are multiple output grads, we should be able to undef one at a time without error
    if len(outputs_to_check[0]) > 1:
        for undef_grad_idx in range(len(outputs)):
            output_to_check = _differentiable_outputs(func(*inputs))
            outputs_to_check.append([
                torch._C._functions.UndefinedGrad()(o) if idx == undef_grad_idx else o
                for idx, o in enumerate(output_to_check)])

    return all(check_undefined_grad_support(output) for output in outputs_to_check)


def _as_tuple(x):
    if isinstance(x, tuple):
        return x
    elif isinstance(x, list):
        return tuple(x)
    else:
        return x,


def _differentiable_outputs(x):
    return tuple(o for o in _as_tuple(x) if o.requires_grad)


def get_notallclose_msg(analytical, numerical, output_idx, input_idx, error_str='') -> str:
    return error_str + 'Jacobian mismatch for output %d with respect to input %d,\n' \
        'numerical:%s\nanalytical:%s\n' % (output_idx, input_idx, numerical, analytical)

def transpose(m):
    out: List[List[torch.Tensor]] = []
    for j in range(len(m[0])):
        out.append([])
        for i in range(len(m)):
            out[j].append(m[i][j])
    return out


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

    if not check_inputs(fail_test, tupled_inputs, check_sparse_nnz):
        return False

    func_out = func(*tupled_inputs)
    outputs = _differentiable_outputs(func_out)

    check_outputs(outputs)

    if not outputs:
        return check_no_differentiable_outputs(fail_test, func, tupled_inputs, _as_tuple(func_out), eps)

    numerical = transpose(get_numerical_jacobian(func, tupled_inputs, outputs, eps=eps))
    numerical_from_imag_grad_out = transpose(get_numerical_jacobian(func, tupled_inputs, outputs, eps=eps, grad_out=1j))

    # recompute because get_numerical_jacobians
    outputs = _differentiable_outputs(func(*tupled_inputs))

    for i, o in enumerate(outputs):
        analytical, failed = check_analytical_jacobian_attributes(tupled_inputs, o, nondet_tol, 1.0,
                                                                  check_grad_dtypes, raise_exception)
        if failed:
            return False

        if o.is_complex():
            analytical_from_imag_grad_out, failed = check_analytical_jacobian_attributes(
                tupled_inputs, o, nondet_tol, 1j, check_grad_dtypes, raise_exception)
            if failed:
                return False

        inp_tensors = iter_tensors(tupled_inputs, True)

        for j, (a, n, inp) in enumerate(zip(analytical, numerical[i], inp_tensors)):
            if a.numel() != 0 or n.numel() != 0:
                if o.is_complex():    # C -> C, R -> C
                    if not torch.allclose(analytical_from_imag_grad_out[j], numerical_from_imag_grad_out[i][j], rtol, atol):
                        return fail_test(get_notallclose_msg(analytical_from_imag_grad_out[j],
                                                             numerical_from_imag_grad_out[i][j], i, j,
                                                             "Gradients failed to compare equal for grad output = 1j. "))
                if inp.is_complex():  # C -> R, C -> C
                    if not torch.allclose(a, n, atol, rtol):
                        return fail_test(get_notallclose_msg(a, n, i, j,
                                                             "Gradients failed to compare equal for grad output = 1. "))
                else:                 # R -> R, R -> C
                    if not torch.allclose(a, n, atol, rtol):
                        return fail_test(get_notallclose_msg(a, n, i, j))

        if check_batched_grad:
            if not test_batched_grad(fail_test, tupled_inputs, o, i):
                return False

    if not test_backward_mul_by_grad_output(fail_test, outputs, tupled_inputs, check_sparse_nnz):
        return False

    if check_undefined_grad:
        if not test_undefined_grad(fail_test, func, outputs, tupled_inputs):
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
