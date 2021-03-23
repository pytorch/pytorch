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
    return is_tensor_like(obj) and (obj.is_floating_point() or obj.is_complex())


def make_jacobians_with_inputs(input_tensors: Tuple, dim=None):
    """makes zero-filled tensors from inputs. If `dim` is not None, for each tensor in
    `input_tensors`, returns a new zero-filled tensor with height of `t.numel` and width
    of `dim`. Otherwise, for each tensor, returns a 1-d tensor with size `(t.numel,)`.
    Each new tensor will be strided and have the same dtype and device as those of the
    corresponding input"""
    out: List[torch.Tensor] = []
    for t in input_tensors:
        if is_float_or_complex_tensor(t) and t.requires_grad:
            if dim is None:
                out.append(t.new_zeros((t.nelement(),), layout=torch.strided))
            else:
                out.append(t.new_zeros((t.nelement(), dim), layout=torch.strided))
    return tuple(out)


def make_jacobians_with_outputs(output_tensors: Tuple, dtype=None, device=None, dim=None):
    """makes zero-filled tensors from outputs. If `dim` is not None, for each tensor in
    `output_tensors`, returns a new zero-filled tensor with height of `dim` and width of
    `t.numel`. Otherwise, for each tensor, returns a 1-d tensor with size (t.numel,).
    """
    out: List[torch.Tensor] = []
    options = {"dtype": dtype, "device": device, "layout": torch.strided}
    for t in output_tensors:
        if is_float_or_complex_tensor(t):
            if dim is None:
                out.append(t.new_zeros((t.nelement(),), **options))
            else:
                out.append(t.new_zeros((dim, t.nelement()), **options))
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
    """For strided and sparse tensors, provides a "view" of the original tensor.
    Updates through the view update the original, but do not bump version count.
    For mkldnn tensors, however, the returned tensor will be a dense *copy*.
    Also provides the current index into that tensor, as well as a corresponding
    "flat index" which translates to a given row/col in the jacobian matrix.
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
                yield x_value, x_idx, d_idx
    elif x_tensor.layout == torch._mkldnn:  # type: ignore
        # Use .data here to get around the version check
        x_tensor = x_tensor.data
        for d_idx, x_idx in enumerate(product(*[range(m) for m in x_tensor.size()])):
            # this is really inefficient, but without indexing implemented, there's
            # not really a better way than converting back and forth
            x_tensor_dense = x_tensor.to_dense()
            yield x_tensor_dense, x_idx, d_idx
    else:
        # Use .data here to get around the version check
        x_tensor = x_tensor.data
        for d_idx, x_idx in enumerate(product(*[range(m) for m in x_tensor.size()])):
            yield x_tensor, x_idx, d_idx


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
        outputs = _as_tuple(fn(inputs))
    if target is None:
        target = inputs
    for i, inp in enumerate(iter_tensors(target, True)):
        if inp.layout == torch._mkldnn and len(inputs) != 1:  # type: ignore # no attr _mkldnn
            raise ValueError('gradcheck currently only supports functions with 1 input, but got: ',
                             len(inputs))
        jacobians += [get_numerical_jacobian_for_input(fn, inp, inputs, outputs, eps, eps, grad_out)]
    return jacobians


def compute_gradient(fn, entry, v, norm_v):
    """Performs finite differencing by perturbing `entry` in-place by `v` and
    returns the gradient of each of the outputs wrt to x at idx.
    """
    # we currently assume that the norm of delta equals eps
    if isinstance(v, torch.Tensor) and v.layout != torch.sparse_coo:
        v = v.reshape(entry.shape)

    orig = entry.clone()
    entry.copy_(orig - v)
    outa = fn()
    entry.copy_(orig + v)
    outb = fn()
    entry.copy_(orig)

    def compute(a, b):
        ret = (b - a) / (2 * norm_v)
        return ret.detach().reshape(-1)

    return tuple(compute(a, b) for (a, b) in zip(outa, outb))


def get_numerical_jvp(jacobians_cols, delta, jvp_fn, input_is_complex, grad_out):
    # compute gradient only works for pure real or pure imaginary delta
    # for details on the algorithm used here, refer:
    # Section 3.5.3 https://arxiv.org/pdf/1701.00392.pdf
    # s = fn(z) where z = x for real valued input
    # and z = x + yj for complex valued input
    ds_dx_tup = jvp_fn(delta)

    if input_is_complex:            # C -> C, C -> R
        ds_dy_tup = jvp_fn(delta * 1j)
        for ds_dx, ds_dy in zip(ds_dx_tup, ds_dy_tup):
            # conjugate wirtinger derivative
            conj_w_d = 0.5 * (ds_dx + ds_dy * 1j)
            # wirtinger derivative
            w_d = 0.5 * (ds_dx - ds_dy * 1j)
            jacobians_cols.append(grad_out.conjugate() * conj_w_d + grad_out * w_d.conj())
    else:
        for ds_dx in ds_dx_tup:
            if ds_dx.is_complex():  # R -> C
                # w_d = conj_w_d = 0.5 * ds_dx
                # dL_dz_conj = 0.5 * [grad_out.conj() * ds_dx + grad_out * ds_dx.conj()]
                #            = 0.5 * [grad_out.conj() * ds_dx + (grad_out.conj() * ds_dx).conj()]
                #            = 0.5 * 2 * real(grad_out.conj() * ds_dx)
                #            = real(grad_out.conj() * ds_dx)
                jacobians_cols.append(torch.real(grad_out.conjugate() * ds_dx))
            else:                   # R -> R
                # skip if grad_out is complex but output is real
                if not isinstance(grad_out, complex):
                    jacobians_cols.append(ds_dx * grad_out)
                else:
                    jacobians_cols.append(None)


def combine_jacobian_cols(jacobians_cols, outputs, input, dim=None):
    jacobians = make_jacobians_with_outputs(outputs, input.dtype, input.device, dim=dim)
    for i, jacobian in enumerate(jacobians):
        for k, v in jacobians_cols.items():
            jacobian[k] = v[i]
    return jacobians


def get_numerical_jacobian_for_input(fn, input, inputs, outputs, delta, eps, grad_out):
    """Computes the numerical jacobians wrt to a single input. Returns N jacobian
    tensors, where N is the number of outputs. Input must require grad.
    """
    assert input.requires_grad
    jacobian_cols: Dict[int, List[Optional[torch.Tensor]]] = {}
    for x, idx, d_idx in iter_tensor(input):
        def wrapped_fn():
            if input.layout == torch._mkldnn:  # type: ignore # no attr _mkldnn
                # convert the dense tensor back to have mkldnn layout
                inp = [x.to_mkldnn()]
            elif input.layout == torch.sparse_coo:
                inp = [a.clone() for a in _as_tuple(inputs)]
            else:
                # x is a view into input and so this works
                inp = _as_tuple(inputs)
            return tuple(a.clone() for a in _as_tuple(fn(*inp)))

        entry = x[idx]

        def jvp_fn(delta):
            return compute_gradient(wrapped_fn, entry, delta, eps)
        jacobian_cols[d_idx] = []
        get_numerical_jvp(jacobian_cols[d_idx], delta, jvp_fn, x.is_complex(), grad_out)
    return combine_jacobian_cols(jacobian_cols, outputs, input, dim=input.numel())

def sparse_clone(input):
    # returns a new sparse tensor that shares storage with `input`
    # this is used to avoid invoking coalesce on the original input tensor
    # assumes that input is coalesced
    if input.layout == torch.sparse_coo:
        return torch.sparse_coo_tensor(input.indices(), input.values(), input.size())
    return input

def get_fast_numerical_jacobian_for_input(fn, input_idx, input, inputs, outputs, delta, eps, grad_out):
    jacobian_cols: List[Optional[torch.Tensor]] = []

    if input.layout == torch._mkldnn:  # type: ignore # no attr _mkldnn
        # TODO do we really need this anymore for the fast case?
        entry = input.to_dense()
    elif input.layout == torch.sparse_coo:
        entry = torch.sparse_coo_tensor(input.indices(), input.values(), input.size())
    else:
        entry = input.data

    def wrapped_fn():
        if input.layout == torch._mkldnn:  # type: ignore # no attr _mkldnn
            inp = [entry.to_mkldnn()]
        elif input.layout == torch.sparse_coo:
            inp = [sparse_clone(a) if i != input_idx else entry for i, a in enumerate(_as_tuple(inputs))]
        else:
            inp = _as_tuple(inputs)
        return tuple(a.clone() for a in _as_tuple(fn(*inp)))

    def jvp_fn(delta):
        return compute_gradient(wrapped_fn, entry, delta, eps)

    get_numerical_jvp(jacobian_cols, delta, jvp_fn, input.is_complex(), grad_out)
    jacobians = make_jacobians_with_outputs(outputs, dtype=input.dtype, device=input.device)

    for i, jacobian in enumerate(jacobians):
        jacobian.copy_(jacobian_cols[i])
    return jacobians


def check_jacobians_equal(j1, j2, atol):
    # Check whether the max diff betwen two jacobians are within some tolerance `atol`
    for j1_x, j2_x in zip(j1, j2):
        if j1_x.numel() != 0 and (j1_x - j2_x).abs().max() > atol:
            return False
    return True


def combine_jacobian_rows(jacobians_rows, inputs, dim):
    out_jacobians = make_jacobians_with_inputs(inputs, dim)
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
                                         raise_exception, custom_backward_fn=None, fast_mode=False, v=None):
    diff_input_list = list(iter_tensors(inputs, True))

    def backward_fn(grad_output):
        return torch.autograd.grad(output, diff_input_list, grad_output,
                                   retain_graph=True, allow_unused=True)
    fn = custom_backward_fn if custom_backward_fn is not None else backward_fn

    if fast_mode:
        jacobians_rows = get_fast_analytic_jacobian(fn, output.clone(), v, grad_out_scale)
        jacobians_rows_reentrant = get_fast_analytic_jacobian(fn, output.clone(), v, grad_out_scale)
    else:
        jacobians_rows = get_analytical_jacobian(fn, output.clone(), grad_out_scale)
        jacobians_rows_reentrant = get_analytical_jacobian(fn, output.clone(), grad_out_scale)
    dim = output.numel() if not fast_mode else 1

    jacobians, correct_grad_types, correct_grad_sizes = combine_jacobian_rows(jacobians_rows, inputs, dim)
    jacobians_reentrant, _, _ = combine_jacobian_rows(jacobians_rows_reentrant, inputs, dim)

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


def get_fast_analytic_jacobian(fn, sample_output, v, grad_out_scale):
    # For each input, computes f(v), which is *supposed* to be v^T J
    jacobians_rows: Dict[int, List[Optional[torch.Tensor]]] = {}
    grad_inputs = fn(v.reshape(sample_output.shape) * grad_out_scale)
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
    if any(t.layout == torch.sparse_coo for t in outputs if isinstance(t, torch.Tensor)):
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


def check_no_differentiable_outputs_fast(fail_test, func, func_out, inputs, all_u, eps):
    diff_idx = 0
    for i, inp in enumerate(inputs):
        if not is_tensor_like(inp) or not inp.requires_grad:
            continue
        u = all_u[diff_idx]
        numerical = get_fast_numerical_jacobian_for_input(func, i, inp, inputs, _as_tuple(func_out), eps * u, eps, 1.0)
        for n in numerical:
            # TODO: Why do get small non-zero values here
            if not torch.allclose(n, torch.zeros_like(n), atol=1e-12):
                return fail_test('Numerical gradient for function expected to be zero')
        diff_idx += 1
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


def slow_gradcheck(fail_test, func, func_out, tupled_inputs, outputs, eps, rtol,
                   atol, raise_exception, check_grad_dtypes, nondet_tol):
    if not outputs:
        return check_no_differentiable_outputs(fail_test, func, tupled_inputs, _as_tuple(func_out), eps)

    numerical = transpose(get_numerical_jacobian(func, tupled_inputs, outputs, eps=eps))
    if any(isinstance(o, torch.Tensor) and o.is_complex() for o in _as_tuple(func_out)):
        numerical_from_imag_grad_out = transpose(get_numerical_jacobian(func, tupled_inputs, outputs, eps=eps, grad_out=1j))

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
                    if not torch.allclose(a, n, rtol, atol):
                        return fail_test(get_notallclose_msg(a, n, i, j,
                                                             "Gradients failed to compare equal for grad output = 1. "))
                else:                 # R -> R, R -> C
                    if not torch.allclose(a, n, rtol, atol):
                        return fail_test(get_notallclose_msg(a, n, i, j))
    return True


def dot(u, v):
    if v.is_complex() and not u.is_complex():
        return v.dot(u.to(dtype=v.dtype))
    elif u.is_complex() and not v.is_complex():
        return u.dot(v.to(dtype=u.dtype))
    else:
        return u.dot(v)


def fast_gradcheck(fail_test, func, func_out, tupled_inputs, outputs, eps, rtol,
                   atol, raise_exception, check_grad_dtypes, nondet_tol):
    # See https://github.com/pytorch/pytorch/issues/53876
    def vec_from_tensor(x):
        # If x is complex, we create a complex tensor with only real component
        if x.layout == torch.sparse_coo:
            # For sparse, create a random sparse vec with random values in the same
            # indices. Make sure size is set so that it isn't inferred to be smaller.
            x_values = x.values()
            values = torch.rand(x_values.nelement()).to(dtype=x.dtype, device=x.device).reshape(x_values.shape)
            values /= values.norm()
            vec = torch.sparse_coo_tensor(x.indices(), values, x.size())
        else:
            vec = torch.rand(x.nelement()).to(dtype=x.dtype, device=x.device)
            vec /= vec.norm()
        return vec

    inp_tensors = [t for t in tupled_inputs if is_tensor_like(t) and t.requires_grad]

    all_u = [vec_from_tensor(inp) for inp in inp_tensors]
    all_u_dense = [u.to_dense().reshape(-1) if u.layout == torch.sparse_coo else u for u in all_u]
    all_v = [vec_from_tensor(out) for out in outputs if is_tensor_like(out)]

    if not outputs:
        if not check_no_differentiable_outputs_fast(fail_test, func, func_out, tupled_inputs, all_u, eps):
            return False

    any_complex = any(o.is_complex() for o in outputs)
    complex_output_indices = [i for i, o in enumerate(outputs) if o.is_complex()]

    # Initialize list of lists to store jacobians for each input, output pair
    all_analytical: List[List[torch.Tensor]] = [[] for _ in outputs]
    all_numerical: List[List[torch.Tensor]] = [[] for t in inp_tensors]
    all_analytical_from_imag_grad_out: List[List[torch.Tensor]] = [[] for _ in complex_output_indices]
    all_numerical_from_imag_grad_out: List[List[torch.Tensor]] = [[] for t in inp_tensors]

    # Numerically approximate v^T (J u)
    # diff_idx represents only counts tensors that require grad, but input_idx counts
    # all tensors
    diff_idx = 0
    for input_idx, inp in enumerate(tupled_inputs):
        if not is_tensor_like(inp) or not inp.requires_grad:  # type: ignore
            continue
        u = all_u[diff_idx]
        numerical = get_fast_numerical_jacobian_for_input(func, input_idx, inp, tupled_inputs, outputs, eps * u, eps, 1.0)

        for j, (a, v) in enumerate(zip(numerical, all_v)):
            out = dot(a, v.to(device=a.device))
            all_numerical[diff_idx].append(dot(a, v.to(device=a.device)))

        if any_complex:
            numerical_from_imag_grad_out = get_fast_numerical_jacobian_for_input(
                func, input_idx, inp, tupled_inputs, outputs, eps * u, eps, 1j)
            for j in complex_output_indices:
                a, v = numerical_from_imag_grad_out[j], all_v[j]
                all_numerical_from_imag_grad_out[diff_idx].append(dot(a, v.to(device=a.device)))
        diff_idx += 1

    # Analytically calculate (v^T J) u
    for i, (out, v) in enumerate(zip(outputs, all_v)):
        analytical, failed = check_analytical_jacobian_attributes(tupled_inputs, out, nondet_tol, 1.0, check_grad_dtypes,
                                                                  raise_exception, fast_mode=True, v=v)
        if failed:
            return False

        for a, u in zip(analytical, all_u_dense):
            all_analytical[i].append(a.T.squeeze(0).dot(u))

        if out.is_complex():
            analytical_from_imag_grad_out, failed = check_analytical_jacobian_attributes(
                tupled_inputs, out, nondet_tol, 1j, check_grad_dtypes, raise_exception, fast_mode=True, v=v)
            if failed:
                return False

            for j, (a, u) in enumerate(zip(analytical_from_imag_grad_out, all_u_dense)):
                all_analytical_from_imag_grad_out[i].append(a.T.squeeze(0).dot(u))


    prefix = "Gradients failed to compare equal for grad output = 1j. "
    # Make sure analytical and numerical is same when calcaluted using grad_out = 1j
    for i, all_numerical_for_input_i in enumerate(all_numerical_from_imag_grad_out):
        for j, n in enumerate(all_numerical_for_input_i):
            # TODO: clean up cases
            a = all_analytical_from_imag_grad_out[j][i]
            if a.is_complex() and not n.is_complex():
                if not torch.allclose(a, n.to(a.dtype), rtol, atol):
                    return fail_test(get_notallclose_msg(a, n, i, j, prefix))
            elif n.is_complex() and not a.is_complex():
                if not torch.allclose(a.to(n.dtype), n, rtol, atol):
                    return fail_test(get_notallclose_msg(a, n, i, j, prefix))
            else:
                if not torch.allclose(a, n, rtol, atol):
                    return fail_test(get_notallclose_msg(a, n, i, j, prefix))

    # Make sure analytical and numerical is the same
    for i, (all_numerical_for_input_i, inp) in enumerate(zip(all_numerical, inp_tensors)):
        prefix = "" if not inp.is_complex() else \
            "Gradients failed to compare equal for grad output = 1. "
        for j, n in enumerate(all_numerical_for_input_i):
            a = all_analytical[j][i]
            if a.is_complex() and not n.is_complex():
                if not torch.allclose(a, n.to(a.dtype), rtol, atol):
                    return fail_test(get_notallclose_msg(a, n, i, j, prefix))
            elif n.is_complex() and not a.is_complex():
                if not torch.allclose(a.to(n.dtype), n, rtol, atol):
                    return fail_test(get_notallclose_msg(a, n, i, j, prefix))
            else:
                if not torch.allclose(a, n, rtol, atol):
                    return fail_test(get_notallclose_msg(a, n, i, j, prefix))

    return True


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
    nondet_tol: float = 1e-12,  # TODO WHY
    check_undefined_grad: bool = True,
    check_grad_dtypes: bool = False,
    check_batched_grad: bool = False,
    fast_mode: bool = True,
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

    # Coalesce if sparse, so we can get .values() later. We do this before we call func
    # so that input is part of the graph
    def coalesce(input):
        if is_tensor_like(input) and input.layout == torch.sparse_coo:
            return input.coalesce()
        return input

    tupled_inputs = tuple(coalesce(input) for input in _as_tuple(inputs))

    if not check_inputs(fail_test, tupled_inputs, check_sparse_nnz):
        return False

    func_out = func(*tupled_inputs)
    outputs = _differentiable_outputs(func_out)

    check_outputs(outputs)

    if fast_mode:
        if not fast_gradcheck(fail_test, func, func_out, tupled_inputs, outputs, eps, rtol,
                              atol, raise_exception, check_grad_dtypes, nondet_tol):
            return False
    else:
        if not slow_gradcheck(fail_test, func, func_out, tupled_inputs, outputs, eps, rtol,
                              atol, raise_exception, check_grad_dtypes, nondet_tol):
            return False

    for i, o in enumerate(outputs):
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
    fast_mode: bool = True,
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
        check_grad_dtypes=check_grad_dtypes, check_batched_grad=check_batched_grad, fast_mode=fast_mode)
