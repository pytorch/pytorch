import torch
from torch.types import _TensorOrTensors
import torch.testing
from torch.overrides import is_tensor_like
import collections
from itertools import product
import warnings
from typing import Callable, Union, Optional, Iterable, List, Tuple, Dict
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


def allocate_jacobians_with_inputs(input_tensors: Tuple, numel_output) -> Tuple[torch.Tensor, ...]:
    # Makes zero-filled tensors from inputs. If `numel_output` is not None, for each tensor in
    # `input_tensors`, returns a new zero-filled tensor with height of `t.numel` and width
    # of `numel_output`. Otherwise, for each tensor, returns a 1-d tensor with size `(t.numel,)`.
    # Each new tensor will be strided and have the same dtype and device as those of the
    # corresponding input
    out: List[torch.Tensor] = []
    for t in input_tensors:
        if is_float_or_complex_tensor(t) and t.requires_grad:
            out.append(t.new_zeros((t.numel(), numel_output), layout=torch.strided))
    return tuple(out)


def allocate_jacobians_with_outputs(output_tensors: Tuple, dtype=None, device=None,
                                    numel_input=None) -> Tuple[torch.Tensor, ...]:
    # Makes zero-filled tensors from outputs. If `dim` is not None, for each tensor in
    # `output_tensors`, returns a new zero-filled tensor with height of `dim` and width of
    # `t.numel`. Otherwise, for each tensor, returns a 1-d tensor with size (t.numel,).
    out: List[torch.Tensor] = []
    options = {"dtype": dtype, "device": device, "layout": torch.strided}
    for t in output_tensors:
        if is_float_or_complex_tensor(t):
            if numel_input is None:
                out.append(t.new_zeros((t.nelement(),), **options))
            else:
                out.append(t.new_zeros((numel_input, t.numel()), **options))
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
    # Enumerates over a tensor and provides a corresponding flat index that translates
    # to a given rol/col in the jacobian matrix. The order is the same as as if we flatten
    # a contiguous tensor. iter_tensor also returns a strided version of the original
    # tensor that is able to be modified inplace. If the input tensor is strided or sparse,
    # the returned tensor will share storage with the original. Otherwise, for opaque tensor
    # types like mkldnn, a copy is returned.
    #
    # Example:
    #   for a tensor t with size (2, 2), it will yield:
    #     `x, (0, 0), 0`, `x, (0, 1), 1`, `x, (1, 0), 2`, `x, (1, 1), 3`
    #
    #   where x is the t.data of the original tensor. Since input t has numel 4, the
    #   Jacobian should have 4 columns. So having a d_idx of 3 and idx of (1, 1)
    #   indicates that perturbing t[(1, 1)] allows us to updating the third (last)
    #   column of any jacobian corresponding to this particular input.
    #
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


def _get_numerical_jacobian(fn, inputs, outputs=None, target=None, eps=1e-3,
                            grad_out=1.0) -> List[Tuple[torch.Tensor, ...]]:
    """Computes the numerical jacobian for a given fn and inputs. Returns M * N jacobians
    where M is the number of input tensors that require grad, and N is the number of output
    float/complex tensors.

    Args:
        fn: the function to compute the jacobian for
        inputs: inputs to `fn`
        outputs: provide precomputed outputs to avoid one extra invocation of fn
        target: the Tensors wrt whom Jacobians are calculated (default=`inputs`)
        eps: the magnitude of the perturbation during finite differencing (default=`1e-3`)
        grad_out: grad output value used to calculate gradients.

    Returns:
        A list of M N-tuples of tensors

    Note that `target` may not even be part of `input` to `fn`, so please be
    **very careful** in this to not clone `target`.
    """
    jacobians: List[Tuple[torch.Tensor, ...]] = []
    if outputs is None:
        outputs = _as_tuple(fn(*_as_tuple(inputs)))
    if target is None:
        target = inputs
    inp_indices = [i for i, a in enumerate(target) if is_tensor_like(a) and a.requires_grad]
    for i, (inp, inp_idx) in enumerate(zip(iter_tensors(target, True), inp_indices)):
        jacobians += [get_numerical_jacobian_wrt_specific_input(fn, inp, inp_idx, inputs, outputs, eps, grad_out)]
    return jacobians


def get_numerical_jacobian(fn, inputs, target=None, eps=1e-3, grad_out=1.0):
    """Deprecated api to compute numerical jacobian for a given fn and inputs.
    Args:
        fn: the function to compute the jacobian for (must take inputs as a tuple)
        input: input to `fn`
        target: the Tensors wrt whom Jacobians are calculated (default=`input`)
        eps: the magnitude of the perturbation during finite differencing (default=`1e-3`)
        grad_out: grad output value used to calculate gradients.

    Returns:
        A list of jacobians wrt each input (or target) and the first output

    Note that `target` may not even be part of `input` to `fn`, so please be
    **very careful** in this to not clone `target`.
    """
    warnings.warn("get_numerical_jacobian was part of PyTorch's private API and not "
                  "meant to be exposed. We are deprecating it and it will be removed "
                  "in a future version of PyTorch. If you have a specific use for "
                  "this or feature request for this to be a stable API, please file "
                  "us an issue at https://github.com/pytorch/pytorch/issues/new")

    def fn_pack_inps(*inps):
        return fn(inps)
    jacobians = _get_numerical_jacobian(fn_pack_inps, inputs, None, target, eps, grad_out)

    return tuple(jacobian_for_each_output[0] for jacobian_for_each_output in jacobians)


def compute_numerical_gradient(fn, entry, v, norm_v, nbhd_checks_fn):
    # Performs finite differencing by perturbing `entry` in-place by `v` and
    # returns the gradient of each of the outputs wrt to x at idx.
    if isinstance(v, torch.Tensor) and v.layout != torch.sparse_coo:
        v = v.reshape(entry.shape)

    orig = entry.clone()
    entry.copy_(orig - v)
    outa = fn()
    entry.copy_(orig + v)
    outb = fn()
    entry.copy_(orig)

    def compute(a, b):
        nbhd_checks_fn(a, b)
        ret = (b - a) / (2 * norm_v)
        return ret.detach().reshape(-1)

    return tuple(compute(a, b) for (a, b) in zip(outa, outb))


def compute_numerical_jacobian_cols(jvp_fn, delta, input_is_complex, grad_out) -> List[torch.Tensor]:
    # Computing the jacobian only works for pure real or pure imaginary delta
    # for details on the algorithm used here, refer:
    # Section 3.5.3 https://arxiv.org/pdf/1701.00392.pdf
    # s = fn(z) where z = x for real valued input
    # and z = x + yj for complex valued input
    jacobians_cols: List[torch.Tensor] = []
    ds_dx_tup = jvp_fn(delta)

    if input_is_complex:            # C -> C, C -> R
        ds_dy_tup = jvp_fn(delta * 1j)
        for ds_dx, ds_dy in zip(ds_dx_tup, ds_dy_tup):
            if not ds_dx.is_complex() and isinstance(grad_out, complex):
                # placeholder if grad_out is complex but output is not
                jacobians_cols.append(torch.zeros_like(ds_dx))
                continue
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
                if isinstance(grad_out, complex):
                    # placeholder if grad_out is complex but output is not
                    jacobians_cols.append(torch.zeros_like(ds_dx))
                    continue
                jacobians_cols.append(ds_dx * grad_out)
    return jacobians_cols


def combine_jacobian_cols(jacobians_cols: Dict[int, List[torch.Tensor]], outputs, input,
                          numel) -> Tuple[torch.Tensor, ...]:
    # jacobian_cols is a data structure that maps column_idx -> output_idx -> column of jacobian Tensor
    # we return a list that maps output_idx -> full jacobian Tensor
    jacobians = allocate_jacobians_with_outputs(outputs, input.dtype, input.device, numel)
    for i, jacobian in enumerate(jacobians):
        for k, v in jacobians_cols.items():
            jacobian[k] = v[i]
    return jacobians


def prepped_input(input: torch.Tensor, maybe_perturbed_input: Optional[torch.Tensor],
                  fast_mode=False) -> torch.Tensor:
    # Prepares the inputs to be passed into the function while including the new modified input.
    if input.layout == torch._mkldnn:  # type: ignore # no attr _mkldnn
        # Convert back to mkldnn
        if maybe_perturbed_input is not None:
            return maybe_perturbed_input.to_mkldnn()
        else:
            return input
    elif input.layout == torch.sparse_coo:
        if fast_mode and maybe_perturbed_input is not None:
            # entry is already a "cloned" version of the original tensor
            # thus changes to entry are not reflected in the input
            return maybe_perturbed_input
        else:
            # Modifications to entry are reflected in input so we could've just returned `input` here
            # but there is an issue where calling .coalesce on a tensor moves it off the graph when the
            # tensor is already coalesced, so analytical would always return 0 wrt to that input if it
            # is previously used to compute forward pass. To get around this, we need to do an extra clone here.
            # TODO: get rid of this extra clone once https://github.com/pytorch/pytorch/pull/52874 is landed
            # Make this new tensor require again in case the function has hooks
            return torch.sparse_coo_tensor(input._indices(), input._values(), input.size()).requires_grad_(True)
    else:
        # We cannot use entry (input.data) if we want gradgrad to work because
        # fn (in the gradgrad case) needs to compute grad wrt input
        return input


def check_outputs_same_dtype_and_shape_in_neighborhood(output1, output2, eps, idx=None) -> None:
    # Check that the returned outputs don't have different dtype or shape when you
    # perturb the input
    on_index = "on index {idx}" if idx is not None else " "
    assert output1.shape == output2.shape, \
        (f"Expected `func` to return outputs with the same shape"
         f" when inputs are perturbed {on_index}by {eps}, but got:"
         f" shapes {output1.shape} and {output2.shape}.")
    assert output1.dtype == output2.dtype, \
        (f"Expected `func` to return outputs with the same dtype"
         f" when inputs are perturbed {on_index}by {eps}, but got:"
         f" dtypes {output1.dtype} and {output2.dtype}.")


def get_numerical_jacobian_wrt_specific_input(fn, input, input_idx, inputs, outputs, eps,
                                              grad_out) -> Tuple[torch.Tensor, ...]:
    # Computes the numerical jacobians wrt to a single input. Returns N jacobian
    # tensors, where N is the number of outputs. Input must require grad.
    assert input.requires_grad
    # We need a dictionary because for sparse inputs, d_idx aren't necessarily consecutive
    # the ith entry of jacobian_cols[j] is the jth column of jacobian w.r.t. the ith output
    # and input at `input_idx`. The ith entry may be None for when grad_out is 1j but the ith output
    # is not complex.
    jacobian_cols: Dict[int, List[torch.Tensor]] = {}

    for x, idx, d_idx in iter_tensor(input):
        def wrapped_fn():
            inp = tuple(prepped_input(a, x if i == input_idx else None) if is_tensor_like(a) else a
                        for i, a in enumerate(_as_tuple(inputs)))
            return tuple(a.clone() for a in _as_tuple(fn(*inp)))

        input_to_perturb = x[idx]
        nbhd_checks_fn = functools.partial(check_outputs_same_dtype_and_shape_in_neighborhood, idx=idx, eps=eps)

        def jvp_fn(delta):
            return compute_numerical_gradient(wrapped_fn, input_to_perturb, delta, eps, nbhd_checks_fn)

        jacobian_cols[d_idx] = compute_numerical_jacobian_cols(jvp_fn, eps, x.is_complex(), grad_out)

    return combine_jacobian_cols(jacobian_cols, outputs, input, input.numel())


def get_fast_numerical_jacobian_wrt_specific_input(fn, input_idx, input, inputs, outputs, u, eps,
                                                   grad_out) -> Tuple[torch.Tensor, ...]:
    # If fast_mode=False, iter_tensor handles the below cases:
    # basically we want to prepare the input so that it can be modified in-place and do certain
    # operations that require the tensor to have strides
    if input.layout == torch._mkldnn:  # type: ignore # no attr _mkldnn
        # Convert to dense so we can perform operations that require strided tensors
        input_to_perturb = input.to_dense()
    elif input.layout == torch.sparse_coo:
        # TODO: get rid of this extra clone once https://github.com/pytorch/pytorch/pull/52874 is landed
        input_to_perturb = input.clone()
    else:
        input_to_perturb = input.data

    def wrapped_fn():
        inp = tuple(prepped_input(a, input_to_perturb if i == input_idx else None, True) if is_tensor_like(a) else a
                    for i, a in enumerate(_as_tuple(inputs)))
        return tuple(a.clone() for a in _as_tuple(fn(*inp)))

    nbhd_checks_fn = functools.partial(check_outputs_same_dtype_and_shape_in_neighborhood, eps=eps)

    def jvp_fn(delta):
        return compute_numerical_gradient(wrapped_fn, input_to_perturb, delta, eps, nbhd_checks_fn)

    jacobian_cols = compute_numerical_jacobian_cols(jvp_fn, u * eps, input.is_complex(), grad_out)
    jacobians = allocate_jacobians_with_outputs(outputs, dtype=input.dtype, device=input.device)

    for i, jacobian in enumerate(jacobians):
        jacobian.copy_(jacobian_cols[i])
    return jacobians


def check_jacobians_equal(j1, j2, atol):
    # Check whether the max diff betwen two jacobians are within some tolerance `atol`
    for j1_x, j2_x in zip(j1, j2):
        if j1_x.numel() != 0 and (j1_x - j2_x).abs().max() > atol:
            return False
    return True


def combine_jacobian_rows(jacobians_rows, inputs, numel_outputs) -> Tuple[Tuple[torch.Tensor, ...], bool, bool]:
    out_jacobians = allocate_jacobians_with_inputs(inputs, numel_outputs)
    diff_input_list = list(iter_tensors(inputs, True))
    correct_grad_sizes = True
    correct_grad_types = True
    for i, rows in enumerate(jacobians_rows):
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
                out_jacobian[:, j] = row_dense.reshape(-1)
    return out_jacobians, correct_grad_sizes, correct_grad_types


FAILED_NONDET_MSG = """

NOTE: If your op relies on non-deterministic operations i.e., it is listed here:
https://pytorch.org/docs/stable/generated/torch.use_deterministic_algorithms.html
this failure might be expected.

If you are adding a new operator, please file an issue and then use one of the
workarounds. The workaround depends on how your test invokes gradcheck/gradgradcheck.
If the test
- manually invokes gradcheck/gradgradcheck, then call gradcheck/gradgradcheck
  with `nondet_tol=<tol>` as a keyword argument.
- is OpInfo-based (e.g., in test_ops.py), then modify the OpInfo for the test
  to have `gradcheck_nondet_tol=<tol>`.
- is a Module test (e.g., in common_nn.py), then modify the corresponding
  module_test entry to have `gradcheck_nondet_tol=<tol>`
"""


def check_analytical_jacobian_attributes(inputs, output, nondet_tol, grad_out_scale, check_grad_dtypes,
                                         raise_exception, custom_vjp_fn=None, fast_mode=False,
                                         v=None) -> Tuple[Tuple[torch.Tensor, ...], bool]:
    diff_input_list = list(iter_tensors(inputs, True))

    def backward_fn(grad_output):
        return torch.autograd.grad(output, diff_input_list, grad_output,
                                   retain_graph=True, allow_unused=True)
    vjp_fn = custom_vjp_fn if custom_vjp_fn is not None else backward_fn

    if fast_mode:
        jacobians_rows = compute_fast_analytical_jacobian_rows(vjp_fn, output.clone(), v, grad_out_scale)
        jacobians_rows_reentrant = compute_fast_analytical_jacobian_rows(vjp_fn, output.clone(), v, grad_out_scale)
    else:
        jacobians_rows = compute_analytical_jacobian_rows(vjp_fn, output.clone(), grad_out_scale)
        jacobians_rows_reentrant = compute_analytical_jacobian_rows(vjp_fn, output.clone(), grad_out_scale)
    output_numel = output.numel() if not fast_mode else 1

    jacobians, correct_grad_types, correct_grad_sizes = combine_jacobian_rows(jacobians_rows, inputs, output_numel)
    jacobians_reentrant, _, _ = combine_jacobian_rows(jacobians_rows_reentrant, inputs, output_numel)

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
                  'although analytical gradient matches numerical gradient.'
                  f'The tolerance for nondeterminism was {nondet_tol}.' + FAILED_NONDET_MSG)
    failed = not (reentrant and correct_grad_sizes and correct_grad_types)
    return jacobians, failed


def get_analytical_jacobian(inputs, output, nondet_tol=0.0, grad_out=1.0):
    # Replicates the behavior of the old get_analytical_jacobian before the refactor
    warnings.warn("get_analytical_jacobian was part of PyTorch's private API and not "
                  "meant to be exposed. We are deprecating it and it will be removed "
                  "in a future version of PyTorch. If you have a specific use for "
                  "this or feature request for this to be a stable API, please file "
                  "us an issue at https://github.com/pytorch/pytorch/issues/new")

    diff_input_list = list(iter_tensors(inputs, True))

    def backward_fn(grad_output):
        return torch.autograd.grad(output, diff_input_list, grad_output,
                                   retain_graph=True, allow_unused=True)

    jacobians_rows = compute_analytical_jacobian_rows(backward_fn, output.clone(), grad_out)
    jacobians_rows_reentrant = compute_analytical_jacobian_rows(backward_fn, output.clone(), grad_out)

    output_numel = output.numel()
    jacobians, correct_grad_types, correct_grad_sizes = combine_jacobian_rows(jacobians_rows, inputs, output_numel)
    jacobians_reentrant, _, _ = combine_jacobian_rows(jacobians_rows_reentrant, inputs, output_numel)
    reentrant = check_jacobians_equal(jacobians, jacobians_reentrant, nondet_tol)

    return jacobians, reentrant, correct_grad_sizes, correct_grad_types


def _get_analytical_jacobian(inputs, outputs, input_idx, output_idx):
    # Computes the analytical jacobian in slow mode for a single input_idx, output_idx pair
    # without performing checks for dtype, shape, and reentrancy
    jacobians, _ = check_analytical_jacobian_attributes(inputs, outputs[output_idx], float('inf'), 1.0, False, False)
    return jacobians[input_idx]


def compute_analytical_jacobian_rows(vjp_fn, sample_output, grad_out_scale) -> List[List[Optional[torch.Tensor]]]:
    # Computes Jacobian row-by-row using backward function `vjp_fn` = v^T J
    # NB: this function does not assume vjp_fn(v) to return tensors with
    # the same number of elements for different v. This is checked when we
    # later combine the rows into a single tensor.
    grad_out_base = torch.zeros_like(sample_output, memory_format=torch.legacy_contiguous_format)
    flat_grad_out = grad_out_base.view(-1)
    # jacobians_rows[i][j] represents the jth row of the ith input
    jacobians_rows: List[List[Optional[torch.Tensor]]] = []

    for j in range(flat_grad_out.numel()):
        flat_grad_out.zero_()
        flat_grad_out[j] = grad_out_scale
        grad_inputs = vjp_fn(grad_out_base)
        for i, d_x in enumerate(grad_inputs):
            if j == 0:
                jacobians_rows.append([])
            jacobians_rows[i] += [d_x.clone() if isinstance(d_x, torch.Tensor) else None]
    return jacobians_rows


def compute_fast_analytical_jacobian_rows(vjp_fn, sample_output, v,
                                          grad_out_scale) -> List[List[Optional[torch.Tensor]]]:
    # For each input, computes vjp_fn(v), which is *supposed* to be v^T J
    jacobians_rows: List[List[Optional[torch.Tensor]]] = []
    grad_inputs = vjp_fn(v.reshape(sample_output.shape) * grad_out_scale)
    for i, d_x in enumerate(grad_inputs):
        jacobians_rows.append([d_x.clone() if isinstance(d_x, torch.Tensor) else None])
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
    jacobians_all_inputs_outputs = _get_numerical_jacobian(func, inputs, func_out, eps=eps)
    for jacobians_all_outputs_and_fixed_input in jacobians_all_inputs_outputs:
        for jacobian in jacobians_all_outputs_and_fixed_input:
            if torch.ne(jacobian, 0).sum() > 0:
                return fail_test('Numerical gradient for function expected to be zero')
    return True


def check_no_differentiable_outputs_fast(fail_test, func, func_out, all_inputs, input_tensors, inputs_indices,
                                         all_u, eps, nondet_tol):
    for inp_idx, inp, u in zip(inputs_indices, input_tensors, all_u):
        numerical_jacobians = get_fast_numerical_jacobian_wrt_specific_input(func, inp_idx, inp, all_inputs,
                                                                             _as_tuple(func_out), u, eps, 1.0)
        for jacobian in numerical_jacobians:
            if jacobian.numel() == 0:
                continue
            if (jacobian - torch.zeros_like(jacobian)).abs().max() > nondet_tol:
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

def transpose(matrix_of_tensors):
    # returns list of tuples
    return list(zip(*matrix_of_tensors))


def slow_gradcheck(fail_test, func, func_out, tupled_inputs, outputs, eps, rtol,
                   atol, raise_exception, check_grad_dtypes, nondet_tol):
    if not outputs:
        return check_no_differentiable_outputs(fail_test, func, tupled_inputs, _as_tuple(func_out), eps)

    numerical = transpose(_get_numerical_jacobian(func, tupled_inputs, outputs, eps=eps))
    if any(isinstance(o, torch.Tensor) and o.is_complex() for o in _as_tuple(func_out)):
        numerical_from_imag_grad_out = transpose(_get_numerical_jacobian(func, tupled_inputs, outputs, eps=eps, grad_out=1j))

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
    elif u.is_complex() and v.is_complex():
        return u.to(dtype=torch.complex128).dot(v.to(dtype=torch.complex128))
    else:
        return u.to(dtype=torch.float64).dot(v.to(dtype=torch.float64))


def all_close(a, b, rtol, atol):
    if a.is_complex() and not b.is_complex():
        if not torch.allclose(a, b.to(a.dtype), rtol, atol):
            return False
    elif b.is_complex() and not a.is_complex():
        if not torch.allclose(a.to(b.dtype), b, rtol, atol):
            return False
    elif a.is_complex() and b.is_complex():
        if not torch.allclose(a.to(dtype=torch.complex128), b.to(dtype=torch.complex128), rtol, atol):
            return False
    else:
        if not torch.allclose(a.to(dtype=torch.float64), b.to(dtype=torch.float64), rtol, atol):
            return False
    return True


def vec_from_tensor(x, generator):
    # Create a random vector with the same number of elements as x and the same dtype/device
    # If x is complex, we create a complex tensor with only real component
    if x.layout == torch.sparse_coo:
        # For sparse, create a random sparse vec with random values in the same
        # indices. Make sure size is set so that it isn't inferred to be smaller.
        x_values = x._values()
        values = torch.rand(x_values.numel(), generator=generator) \
            .to(dtype=x.dtype, device=x.device) \
            .reshape(x_values.shape)
        values /= values.norm()
        vec = torch.sparse_coo_tensor(x._indices(), values, x.size())
    else:
        vec = torch.rand(x.numel(), generator=generator).to(dtype=x.dtype, device=x.device)
        vec /= vec.norm()
    return vec


def adjusted_atol(atol, u, v):
    # In slow gradcheck, we compare A and B element-wise, i.e., for some a, b we allow |a - b| < atol + rtol * b
    # but since we now compare q1 = v^T A u and q2 = v^T B u, we must allow |q1 - q2| < v^T E u + rtol * v^T B u
    # where E is the correctly sized matrix where each entry is atol
    #
    # We see that atol needs to be scaled by v^T M u (where M is an all-ones M x N matrix):
    # v^T M u = \sum_{i} \sum_{j} u_i * v_j = (\sum_{i} u_i)(\sum_{i} v_i)
    sum_u = torch.sparse.sum(u) if u.layout == torch.sparse_coo else u.sum()
    sum_v = torch.sparse.sum(v) if v.layout == torch.sparse_coo else v.sum()
    return atol * sum_u.item() * sum_v.item()


FAST_FAIL_SLOW_OK_MSG = """
Fast gradcheck failed but element-wise differences are small. This means that the
test might've passed in slow_mode!

If you are adding a new operator, please file an issue and then use one of the
workarounds. The workaround depends on how your test invokes gradcheck/gradgradcheck:

If the test
- manually invokes gradcheck/gradgradcheck, then call gradcheck/gradgradcheck
  with `fast_mode=False` as a keyword argument.
- is OpInfo-based (e.g., in test_ops.py), then modify the OpInfo for the test
  to have `gradcheck_fast_mode=False`
- is a Module test (e.g., in common_nn.py), then modify the corresponding
  module_test entry to have `gradcheck_fast_mode=False`
""".strip()


def slow_mode_jacobian_message(func, tupled_inputs, outputs, input_idx, output_idx, rtol, atol):
    # Compute jacobians in slow mode for better error message
    slow_numerical = _get_numerical_jacobian(func, tupled_inputs, outputs)[input_idx][output_idx]
    slow_analytical = _get_analytical_jacobian(tupled_inputs, outputs, input_idx, output_idx)

    # Assume jacobians are non-empty and have the same shape
    slow_max_diff = (slow_numerical - slow_analytical).abs().max()

    slow_allclose = torch.allclose(slow_analytical, slow_numerical, rtol, atol)
    msg = ("\nThe above quantities relating the numerical and analytical jacobians are computed \n"
           "in fast mode. See: https://github.com/pytorch/pytorch/issues/53876 for more background \n"
           "about fast mode. Below, we recompute numerical and analytical jacobians in slow mode:\n\n"
           f"Numerical:\n {slow_numerical}\n"
           f"Analytical:\n{slow_analytical}\n\n"
           f"The max per-element difference (slow mode) is: {slow_max_diff}.\n")
    if slow_allclose:
        # Slow gradcheck would've passed!
        msg += FAST_FAIL_SLOW_OK_MSG
    return msg


def fast_gradcheck(fail_test, func, func_out, tupled_inputs, outputs, eps, rtol,
                   atol, raise_exception, check_grad_dtypes, nondet_tol):
    # Perform the fast version of gradcheck
    # See https://github.com/pytorch/pytorch/issues/53876 for details
    inp_tensors = [t for t in tupled_inputs if is_tensor_like(t) and t.requires_grad]
    inp_tensor_indices = [i for i, t in enumerate(tupled_inputs) if is_tensor_like(t) and t.requires_grad]

    g_cpu = torch.Generator()
    all_u = [vec_from_tensor(inp, g_cpu) for inp in inp_tensors]
    all_u_dense = [u.to_dense().reshape(-1) if u.layout == torch.sparse_coo else u for u in all_u]
    all_v = [vec_from_tensor(out, g_cpu) for out in outputs]

    if not outputs:
        if not check_no_differentiable_outputs_fast(fail_test, func, func_out, tupled_inputs, inp_tensors,
                                                    inp_tensor_indices, all_u, eps, nondet_tol):
            return False

    any_complex = any(o.is_complex() for o in outputs)
    complex_output_indices = [i for i, o in enumerate(outputs) if o.is_complex()]

    # Initialize list of lists to store jacobians for each input, output pair
    all_analytical: List[List[torch.Tensor]] = [[] for _ in outputs]
    all_numerical: List[List[torch.Tensor]] = [[] for _ in inp_tensors]
    all_analytical_from_imag_grad_out: List[List[torch.Tensor]] = [[] for _ in complex_output_indices]
    all_numerical_from_imag_grad_out: List[List[torch.Tensor]] = [[] for _ in inp_tensors]

    # Numerically approximate v^T (J u)
    for i, (inp, input_idx, u) in enumerate(zip(inp_tensors, inp_tensor_indices, all_u)):
        numerical = get_fast_numerical_jacobian_wrt_specific_input(func, input_idx, inp, tupled_inputs,
                                                                   outputs, u, eps, 1.0)
        for j, (a, v) in enumerate(zip(numerical, all_v)):
            out = dot(a, v.to(device=a.device))
            all_numerical[i].append(dot(a, v.to(device=a.device)))

        if any_complex:
            numerical_from_imag_grad_out = get_fast_numerical_jacobian_wrt_specific_input(
                func, input_idx, inp, tupled_inputs, outputs, u, eps, 1j)
            for j in complex_output_indices:
                a, v = numerical_from_imag_grad_out[j], all_v[j]
                all_numerical_from_imag_grad_out[i].append(dot(a, v.to(device=a.device)))

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

    prefix = "Gradients failed to compare equal for grad output = 1j (in fast mode). "
    # Make sure analytical and numerical is same when calcaluted using grad_out = 1j
    for i, all_numerical_for_input_i in enumerate(all_numerical_from_imag_grad_out):
        for j, n in enumerate(all_numerical_for_input_i):
            a = all_analytical_from_imag_grad_out[j][i]
            if not all_close(a, n, rtol, adjusted_atol(atol, all_u[i], all_v[j])):
                jacobians_str = slow_mode_jacobian_message(func, tupled_inputs, outputs, i, j, rtol, atol)
                return fail_test(get_notallclose_msg(a, n, i, j, prefix) + jacobians_str)

    # Make sure analytical and numerical is the same
    for i, (all_numerical_for_input_i, inp) in enumerate(zip(all_numerical, inp_tensors)):
        prefix = "" if not inp.is_complex() else \
            "Gradients failed to compare equal for grad output = 1 (in fast mode). "
        for j, n in enumerate(all_numerical_for_input_i):
            a = all_analytical[j][i]
            if not all_close(a, n, rtol, adjusted_atol(atol, all_u[i], all_v[j])):
                jacobians_str = slow_mode_jacobian_message(func, tupled_inputs, outputs, i, j, rtol, atol)
                return fail_test(get_notallclose_msg(a, n, i, j, prefix) + jacobians_str)

    return True


def has_complex_inputs_or_outputs(tupled_inputs, func_out):
    return any(is_tensor_like(o) and o.is_complex() for o in _as_tuple(func_out)) or \
        any(is_tensor_like(i) and i.is_complex() for i in tupled_inputs)


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
    fast_mode: bool = False,
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
        fast_mode (bool, optional): Fast mode for gradcheck and gradgradcheck is currently only
            implemented for R to R functions. If none of the inputs and outputs are complex
            a faster implementation of gradcheck that no longer computes the entire jacobian
            is run; otherwise, we fall back to the slow implementation.

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

    if fast_mode and has_complex_inputs_or_outputs(tupled_inputs, func_out):
        raise NotImplementedError("Fast mode for gradcheck and gradgradcheck is currently only implemented"
                                  " for R to R functions.")

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
    fast_mode: bool = False,
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
            batched gradients using prototype
        fast_mode (bool, optional): if True, run a faster implementation of gradgradcheck that
            no longer computes the entire jacobian.vmap support. Defaults to False.

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
