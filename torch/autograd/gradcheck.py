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


class GradcheckError(RuntimeError):
    # Custom error so that user errors are not caught in the main try catch (see gradcheck())
    pass


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


def allocate_jacobians_with_outputs(output_tensors: Tuple, numel_input, dtype=None,
                                    device=None) -> Tuple[torch.Tensor, ...]:
    # Makes zero-filled tensors from outputs. If `dim` is not None, for each tensor in
    # `output_tensors`, returns a new zero-filled tensor with height of `dim` and width of
    # `t.numel`. Otherwise, for each tensor, returns a 1-d tensor with size (t.numel,).
    out: List[torch.Tensor] = []
    options = {"dtype": dtype, "device": device, "layout": torch.strided}
    for t in output_tensors:
        if is_float_or_complex_tensor(t):
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


def _get_numerical_jacobian(fn, inputs, outputs=None, target=None, eps=1e-3) -> List[Tuple[torch.Tensor, ...]]:
    """Computes the numerical jacobian for a given fn and inputs. Returns M * N jacobians
    where M is the number of input tensors that require grad, and N is the number of output
    float/complex tensors.

    Args:
        fn: the function to compute the jacobian for
        inputs: inputs to `fn`
        outputs: provide precomputed outputs to avoid one extra invocation of fn
        target: the Tensors wrt whom Jacobians are calculated (default=`inputs`)
        eps: the magnitude of the perturbation during finite differencing (default=`1e-3`)

    Returns:
        A list of M N-tuples of tensors

    Note that `target` may not even be part of `input` to `fn`, so please be
    **very careful** in this to not clone `target`.
    """
    jacobians: List[Tuple[torch.Tensor, ...]] = []
    if outputs is None:
        outputs = _as_tuple(fn(*_as_tuple(inputs)))
    if any(o.is_complex() for o in outputs):
        raise ValueError("Expected output to be non-complex. get_numerical_jacobian no "
                         "longer supports functions that return complex outputs.")
    if target is None:
        target = inputs
    inp_indices = [i for i, a in enumerate(target) if is_tensor_like(a) and a.requires_grad]
    for inp_idx in inp_indices:
        jacobians += [get_numerical_jacobian_wrt_specific_input(fn, inp_idx, inputs, outputs, eps)]
    return jacobians


def get_numerical_jacobian(fn, inputs, target=None, eps=1e-3, grad_out=1.0):
    """Deprecated api to compute numerical jacobian for a given fn and inputs.
    Args:
        fn: the function to compute the jacobian for (must take inputs as a tuple)
        input: input to `fn`
        target: the Tensors wrt whom Jacobians are calculated (default=`input`)
        eps: the magnitude of the perturbation during finite differencing (default=`1e-3`)

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
    if grad_out != 1.0:
        raise ValueError("Expected grad_out to be 1.0. get_numerical_jacobian no longer "
                         "supports values of grad_out != 1.0.")

    def fn_pack_inps(*inps):
        return fn(inps)
    jacobians = _get_numerical_jacobian(fn_pack_inps, inputs, None, target, eps)

    return tuple(jacobian_for_each_output[0] for jacobian_for_each_output in jacobians)


def compute_numerical_gradient(fn, entry, v, norm_v, nbhd_checks_fn):
    # Performs finite differencing by perturbing `entry` in-place by `v` and
    # returns the gradient of each of the outputs wrt to x at idx.
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


def compute_numerical_jacobian_cols(jvp_fn, delta, input_is_complex, delta_i=None) -> List[torch.Tensor]:
    # Computing the jacobian only works for pure real or pure imaginary delta
    # For details on the algorithm used here, refer:
    # Section 3.5.3 https://arxiv.org/pdf/1701.00392.pdf
    # s = fn(z) where z = x for real valued input
    # and z = x + yj for complex valued input
    jacobians_cols: List[torch.Tensor] = []
    ds_dx_tup = jvp_fn(delta)

    if input_is_complex:  # C -> R
        ds_dy_tup = jvp_fn(delta * 1j) if delta_i is None else jvp_fn(delta_i * 1j)
        for ds_dx, ds_dy in zip(ds_dx_tup, ds_dy_tup):
            assert(not ds_dx.is_complex())
            # conjugate wirtinger derivative
            conj_w_d = ds_dx + ds_dy * 1j
            jacobians_cols.append(conj_w_d)
    else:
        for ds_dx in ds_dx_tup:  # R-> R
            assert(not ds_dx.is_complex())
            jacobians_cols.append(ds_dx)
    return jacobians_cols


def combine_jacobian_cols(jacobians_cols: Dict[int, List[torch.Tensor]], outputs, input,
                          numel) -> Tuple[torch.Tensor, ...]:
    # jacobian_cols is a data structure that maps column_idx -> output_idx -> column of jacobian Tensor
    # we return a list that maps output_idx -> full jacobian Tensor
    jacobians = allocate_jacobians_with_outputs(outputs, numel, input.dtype, input.device)
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
            return input
    else:
        # We cannot use entry (input.data) if we want gradgrad to work because
        # fn (in the gradgrad case) needs to compute grad wrt input
        return input


def check_outputs_same_dtype_and_shape(output1, output2, eps, idx=None) -> None:
    # Check that the returned outputs don't have different dtype or shape when you
    # perturb the input
    on_index = "on index {idx} " if idx is not None else ""
    assert output1.shape == output2.shape, \
        (f"Expected `func` to return outputs with the same shape"
         f" when inputs are perturbed {on_index}by {eps}, but got:"
         f" shapes {output1.shape} and {output2.shape}.")
    assert output1.dtype == output2.dtype, \
        (f"Expected `func` to return outputs with the same dtype"
         f" when inputs are perturbed {on_index}by {eps}, but got:"
         f" dtypes {output1.dtype} and {output2.dtype}.")


def get_numerical_jacobian_wrt_specific_input(fn, input_idx, inputs, outputs, eps) -> Tuple[torch.Tensor, ...]:
    # Computes the numerical jacobians wrt to a single input. Returns N jacobian
    # tensors, where N is the number of outputs
    # We use a dictionary because for sparse inputs, d_idx aren't necessarily consecutive
    jacobian_cols: Dict[int, List[torch.Tensor]] = {}
    input = inputs[input_idx]
    assert input.requires_grad
    for x, idx, d_idx in iter_tensor(input):
        wrapped_fn = with_prepped_inputs(fn, inputs, input_idx, x)
        input_to_perturb = x[idx]
        nbhd_checks_fn = functools.partial(check_outputs_same_dtype_and_shape, idx=idx, eps=eps)
        jvp_fn = get_jvp_fn(wrapped_fn, input_to_perturb, eps, nbhd_checks_fn)
        jacobian_cols[d_idx] = compute_numerical_jacobian_cols(jvp_fn, eps, x.is_complex())
    return combine_jacobian_cols(jacobian_cols, outputs, input, input.numel())


def get_input_to_perturb(input):
    if input.layout == torch._mkldnn:  # type: ignore # no attr _mkldnn
        # Convert to dense so we can perform operations that require strided tensors
        input_to_perturb = input.to_dense()
    elif input.layout == torch.sparse_coo:
        # Clone because input may require grad, and copy_ calls resize_,
        # which is not allowed for .data
        input_to_perturb = input.clone()
    else:
        input_to_perturb = input.data
    return input_to_perturb


def with_prepped_inputs(fn, inputs, input_idx, input_to_perturb, fast_mode=False):
    def wrapped_fn():
        inp = tuple(prepped_input(a, input_to_perturb if i == input_idx else None, fast_mode) if is_tensor_like(a) else a
                    for i, a in enumerate(_as_tuple(inputs)))
        return tuple(a.clone() for a in _as_tuple(fn(*inp)))
    return wrapped_fn


def get_jvp_fn(wrapped_fn, input_to_perturb, eps, nbhd_checks_fn):
    def jvp_fn(delta):
        return compute_numerical_gradient(wrapped_fn, input_to_perturb, delta, eps, nbhd_checks_fn)
    return jvp_fn


def get_fast_numerical_jacobian_wrt_specific_input(fn, input_idx, inputs, outputs, ur, ui,
                                                   eps) -> List[torch.Tensor]:
    # If fast_mode=False, iter_tensor handles the below cases:
    # basically we want to prepare the input so that it can be modified in-place and do certain
    # operations that require the tensor to have strides
    input = inputs[input_idx]
    input_to_perturb = get_input_to_perturb(input)
    wrapped_fn = with_prepped_inputs(fn, inputs, input_idx, input_to_perturb, True)
    nbhd_checks_fn = functools.partial(check_outputs_same_dtype_and_shape, eps=eps)
    jvp_fn = get_jvp_fn(wrapped_fn, input_to_perturb, eps, nbhd_checks_fn)
    if ur.layout != torch.sparse_coo:
        ur = ur.reshape(input_to_perturb.shape)
        ui = ui.reshape(input_to_perturb.shape)
    return compute_numerical_jacobian_cols(jvp_fn, ur * eps, input.is_complex(), ui * eps)


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


def check_analytical_jacobian_attributes(inputs, output, nondet_tol, check_grad_dtypes,
                                         fast_mode=False, v=None) -> Tuple[torch.Tensor, ...]:
    diff_input_list = list(iter_tensors(inputs, True))

    def vjp_fn(grad_output):
        return torch.autograd.grad(output, diff_input_list, grad_output,
                                   retain_graph=True, allow_unused=True)
    if fast_mode:
        jacobians_rows = compute_fast_analytical_jacobian_rows(vjp_fn, output.clone(), v)
        jacobians_rows_reentrant = compute_fast_analytical_jacobian_rows(vjp_fn, output.clone(), v)
    else:
        jacobians_rows = compute_analytical_jacobian_rows(vjp_fn, output.clone())
        jacobians_rows_reentrant = compute_analytical_jacobian_rows(vjp_fn, output.clone())
    output_numel = output.numel() if not fast_mode else 1

    jacobians, correct_grad_types, correct_grad_sizes = combine_jacobian_rows(jacobians_rows, inputs, output_numel)
    jacobians_reentrant, _, _ = combine_jacobian_rows(jacobians_rows_reentrant, inputs, output_numel)

    reentrant = check_jacobians_equal(jacobians, jacobians_reentrant, nondet_tol)

    if not correct_grad_types and check_grad_dtypes:
        raise GradcheckError('Gradient has dtype mismatch')
    if not correct_grad_sizes:
        raise GradcheckError('Analytical gradient has incorrect size')
    if not reentrant:
        raise GradcheckError('Backward is not reentrant, i.e., running backward with '
                             'same input and grad_output multiple times gives different values, '
                             'although analytical gradient matches numerical gradient.'
                             f'The tolerance for nondeterminism was {nondet_tol}.' + FAILED_NONDET_MSG)
    return jacobians


def get_analytical_jacobian(inputs, output, nondet_tol=0.0, grad_out=1.0):
    # Replicates the behavior of the old get_analytical_jacobian before the refactor
    warnings.warn("get_analytical_jacobian was part of PyTorch's private API and not "
                  "meant to be exposed. We are deprecating it and it will be removed "
                  "in a future version of PyTorch. If you have a specific use for "
                  "this or feature request for this to be a stable API, please file "
                  "us an issue at https://github.com/pytorch/pytorch/issues/new")
    if grad_out != 1.0:
        raise ValueError("Expected grad_out to be 1.0. get_analytical_jacobian no longer "
                         "supports values of grad_out != 1.0.")
    if output.is_complex():
        raise ValueError("Expected output to be non-complex. get_analytical_jacobian no "
                         "longer supports functions that return complex outputs.")

    diff_input_list = list(iter_tensors(inputs, True))

    def backward_fn(grad_output):
        return torch.autograd.grad(output, diff_input_list, grad_output,
                                   retain_graph=True, allow_unused=True)

    jacobians_rows = compute_analytical_jacobian_rows(backward_fn, output.clone())
    jacobians_rows_reentrant = compute_analytical_jacobian_rows(backward_fn, output.clone())

    output_numel = output.numel()
    jacobians, correct_grad_types, correct_grad_sizes = combine_jacobian_rows(jacobians_rows, inputs, output_numel)
    jacobians_reentrant, _, _ = combine_jacobian_rows(jacobians_rows_reentrant, inputs, output_numel)
    reentrant = check_jacobians_equal(jacobians, jacobians_reentrant, nondet_tol)

    return jacobians, reentrant, correct_grad_sizes, correct_grad_types


def _get_analytical_jacobian(inputs, outputs, input_idx, output_idx):
    # Computes the analytical jacobian in slow mode for a single input_idx, output_idx pair
    # without performing checks for dtype, shape, and reentrancy
    jacobians = check_analytical_jacobian_attributes(inputs, outputs[output_idx],
                                                     nondet_tol=float('inf'), check_grad_dtypes=False)
    return jacobians[input_idx]


def compute_analytical_jacobian_rows(vjp_fn, sample_output) -> List[List[Optional[torch.Tensor]]]:
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
        flat_grad_out[j] = 1.0
        grad_inputs = vjp_fn(grad_out_base)
        for i, d_x in enumerate(grad_inputs):
            if j == 0:
                jacobians_rows.append([])
            jacobians_rows[i] += [d_x.clone() if isinstance(d_x, torch.Tensor) else None]
    return jacobians_rows


def compute_fast_analytical_jacobian_rows(vjp_fn, sample_output, v) -> List[List[Optional[torch.Tensor]]]:
    # For each input, computes vjp_fn(v), which is *supposed* to be v^T J
    jacobians_rows: List[List[Optional[torch.Tensor]]] = []
    grad_inputs = vjp_fn(v.reshape(sample_output.shape))
    for i, d_x in enumerate(grad_inputs):
        jacobians_rows.append([d_x.clone() if isinstance(d_x, torch.Tensor) else None])
    return jacobians_rows


def check_inputs(tupled_inputs, check_sparse_nnz) -> bool:
    if not check_sparse_nnz and any(t.is_sparse for t in tupled_inputs if isinstance(t, torch.Tensor)):
        raise GradcheckError('gradcheck expects all tensor inputs are dense when check_sparse_nnz is set to False.')
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
                    raise ValueError(
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


def check_no_differentiable_outputs(func, inputs, func_out, eps) -> bool:
    # When there are no differentiable outputs, numerical gradient for a function is
    # expected to be zero.
    jacobians_all_inputs_outputs = _get_numerical_jacobian(func, inputs, func_out, eps=eps)
    for jacobians_all_outputs_and_fixed_input in jacobians_all_inputs_outputs:
        for jacobian in jacobians_all_outputs_and_fixed_input:
            if torch.ne(jacobian, 0).sum() > 0:
                raise GradcheckError('Numerical gradient for function expected to be zero')
    return True


def check_no_differentiable_outputs_fast(func, func_out, all_inputs, inputs_indices,
                                         all_ur, all_ui, eps, nondet_tol):
    for inp_idx, ur, ui in zip(inputs_indices, all_ur, all_ui):
        numerical_jacobians = get_fast_numerical_jacobian_wrt_specific_input(func, inp_idx, all_inputs,
                                                                             _as_tuple(func_out), ur, ui, eps)
        for jacobian in numerical_jacobians:
            if jacobian.numel() == 0:
                continue
            if (jacobian - torch.zeros_like(jacobian)).abs().max() > nondet_tol:
                raise GradcheckError('Numerical gradient for function expected to be zero')
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


def test_batched_grad(input, output, output_idx) -> bool:
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
            raise GradcheckError(
                f'While computing batched gradients, got: {ex}\n\n{FAILED_BATCHED_GRAD_MSG}')

    for input_idx, (res, exp) in enumerate(zip(result, expected)):
        if torch.allclose(res, exp):
            continue
        raise GradcheckError(get_failed_batched_grad_test_msg(output_idx, input_idx, res, exp))
    return True


def test_backward_mul_by_grad_output(outputs, inputs, check_sparse_nnz) -> bool:
    # Tests that backward is multiplied by grad_output
    diff_input_list: List[torch.Tensor] = list(iter_tensors(inputs, True))
    if not diff_input_list:
        raise GradcheckError("no Tensors requiring grad found in input")
    grads_input = torch.autograd.grad(outputs, diff_input_list,
                                      [torch.zeros_like(o, memory_format=torch.legacy_contiguous_format) for o in outputs],
                                      allow_unused=True)
    for gi, di in zip(grads_input, diff_input_list):
        if gi is None:
            continue
        if isinstance(gi, torch.Tensor) and gi.layout != torch.strided:
            if gi.layout != di.layout:
                raise GradcheckError('grad is incorrect layout (' + str(gi.layout) + ' is not ' + str(di.layout) + ')')
            if gi.layout == torch.sparse_coo:
                if gi.sparse_dim() != di.sparse_dim():
                    raise GradcheckError('grad is sparse tensor, but has incorrect sparse_dim')
                if gi.dense_dim() != di.dense_dim():
                    raise GradcheckError('grad is sparse tensor, but has incorrect dense_dim')
            gi = gi.to_dense()
            di = di.to_dense()

        if check_sparse_nnz:
            if not torch.allclose(gi, torch.zeros_like(gi)):
                raise GradcheckError('backward not multiplied by grad_output')
        elif not gi.eq(0).all():
            raise GradcheckError('backward not multiplied by grad_output')
        if gi.dtype != di.dtype or gi.device != di.device or gi.is_sparse != di.is_sparse:
            raise GradcheckError("grad is incorrect type")
        if gi.size() != di.size():
            raise GradcheckError('grad is incorrect size')
    return True


def test_undefined_grad(func, outputs, inputs) -> bool:
    diff_input_list: List[torch.Tensor] = list(iter_tensors(inputs, True))
    if not diff_input_list:
        raise GradcheckError("no Tensors requiring grad found in input")

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
            raise GradcheckError((
                'Expected backward function to handle undefined output grads. '
                'Please look at "Notes about undefined output gradients" in '
                '"tools/autograd/derivatives.yaml"'))

        for gi, i in zip(grads_input, diff_input_list):
            if (gi is not None) and (not gi.eq(0).all()):
                warn_bc_breaking()
                raise GradcheckError((
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


def get_notallclose_msg(analytical, numerical, output_idx, input_idx, complex_indices, inp_is_complex) -> str:
    out_is_imag = complex_indices and output_idx in complex_indices
    grad_output = "1j" if out_is_imag else "1"
    prefix = "" if not inp_is_complex and not out_is_imag else \
        f"Gradients failed to compare equal for grad output = {grad_output}"
    return prefix + 'Jacobian mismatch for output %d with respect to input %d,\n' \
        'numerical:%s\nanalytical:%s\n' % (output_idx, input_idx, numerical, analytical)


def transpose(matrix_of_tensors):
    # returns list of tuples
    return list(zip(*matrix_of_tensors))


def real_and_imag(fn, sample_outputs):
    # returns new functions real(fn), and imag(fn) where real(fn) and imag(fn) behave the same as
    # the original fn, except torch.real or torch.imag are applied to the complex outputs
    def apply_to_c_outs(fn, fn_to_apply):
        def wrapped_fn(*inputs):
            outs = _as_tuple(fn(*inputs))
            return tuple(fn_to_apply(o) if o.is_complex() else o for o in outs)
        return wrapped_fn
    return apply_to_c_outs(fn, torch.real), apply_to_c_outs(fn, torch.imag)


def gradcheck_real_imag(gradcheck_fn, func, func_out, tupled_inputs, outputs, eps, rtol,
                        atol, check_grad_dtypes, nondet_tol, complex_indices, any_outputs_complex):
    if any_outputs_complex:
        real_fn, imag_fn = real_and_imag(func, outputs)

        imag_func_out = imag_fn(*tupled_inputs)
        imag_outputs = _differentiable_outputs(imag_func_out)
        gradcheck_fn(imag_fn, imag_func_out, tupled_inputs, imag_outputs, eps,
                     rtol, atol, check_grad_dtypes, nondet_tol, complex_indices)

        real_func_out = real_fn(*tupled_inputs)
        real_outputs = _differentiable_outputs(real_func_out)
        gradcheck_fn(real_fn, real_func_out, tupled_inputs, real_outputs, eps,
                     rtol, atol, check_grad_dtypes, nondet_tol)
    else:
        gradcheck_fn(func, func_out, tupled_inputs, outputs, eps,
                     rtol, atol, check_grad_dtypes, nondet_tol)


def slow_gradcheck(func, func_out, tupled_inputs, outputs, eps, rtol,
                   atol, check_grad_dtypes, nondet_tol, complex_indices=None):
    if not outputs:
        return check_no_differentiable_outputs(func, tupled_inputs, _as_tuple(func_out), eps)

    numerical = transpose(_get_numerical_jacobian(func, tupled_inputs, outputs, eps=eps))

    for i, o in enumerate(outputs):
        analytical = check_analytical_jacobian_attributes(tupled_inputs, o, nondet_tol, check_grad_dtypes)
        inp_tensors = iter_tensors(tupled_inputs, True)

        for j, (a, n, inp) in enumerate(zip(analytical, numerical[i], inp_tensors)):
            if a.numel() != 0 or n.numel() != 0:
                if not torch.allclose(a, n, rtol, atol):
                    raise GradcheckError(get_notallclose_msg(a, n, i, j, complex_indices, inp.is_complex()))
    return True


def dot(u, v):
    return (u * v).sum()


def allclose_with_type_promotion(a, b, rtol, atol):
    promoted_type = torch.promote_types(a.dtype, b.dtype)
    a = a.to(dtype=promoted_type)
    b = b.to(dtype=promoted_type)
    return torch.allclose(a, b, rtol, atol)


def vec_from_tensor(x, generator, always_float64=False):
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
        dtype = x.dtype if not always_float64 else torch.float64
        vec = torch.rand(x.numel(), generator=generator).to(dtype=dtype, device=x.device)
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


def fast_gradcheck(func, func_out, tupled_inputs, outputs, eps, rtol,
                   atol, check_grad_dtypes, nondet_tol, complex_indices=None):
    # Perform the fast version of gradcheck
    # See https://github.com/pytorch/pytorch/issues/53876 for details
    inp_tensors = [t for t in tupled_inputs if is_tensor_like(t) and t.requires_grad]
    inp_tensor_indices = [i for i, t in enumerate(tupled_inputs) if is_tensor_like(t) and t.requires_grad]

    g_cpu = torch.Generator()
    all_ur = [vec_from_tensor(inp, g_cpu, True) for inp in inp_tensors]
    all_ur_dense = [u.to_dense().reshape(-1) if u.layout == torch.sparse_coo else u for u in all_ur]
    all_ui = [vec_from_tensor(inp, g_cpu, True) for inp in inp_tensors]
    all_ui_dense = [u.to_dense().reshape(-1) if u.layout == torch.sparse_coo else u for u in all_ui]
    all_v = [vec_from_tensor(out, g_cpu) for out in outputs]

    if not outputs:
        check_no_differentiable_outputs_fast(func, func_out, tupled_inputs, inp_tensor_indices,
                                             all_ur, all_ui, eps, nondet_tol)

    # Initialize list of lists to store jacobians for each input, output pair
    all_analytical: List[List[torch.Tensor]] = [[] for _ in outputs]
    all_numerical: List[List[torch.Tensor]] = [[] for _ in inp_tensors]

    # Numerically approximate v^T (J u)
    for i, (input_idx, ur, ui) in enumerate(zip(inp_tensor_indices, all_ur, all_ui)):
        numerical = get_fast_numerical_jacobian_wrt_specific_input(func, input_idx, tupled_inputs,
                                                                   outputs, ur, ui, eps)
        for j, (a, v) in enumerate(zip(numerical, all_v)):
            all_numerical[i].append(dot(a, v))

    # Analytically calculate (v^T J) u
    for i, (out, v) in enumerate(zip(outputs, all_v)):
        analytical = check_analytical_jacobian_attributes(tupled_inputs, out, nondet_tol, check_grad_dtypes,
                                                          fast_mode=True, v=v)
        for a, ur, ui in zip(analytical, all_ur_dense, all_ui_dense):
            if a.is_complex():
                av = torch.view_as_real(a.T.squeeze(0))
                ar = av.select(-1, 0)
                ai = av.select(-1, 1)
                all_analytical[i].append(ar.dot(ur) + 1j * ai.dot(ui))
            else:
                all_analytical[i].append(dot(a.T.squeeze(0), ur))

    # Make sure analytical and numerical is the same
    for i, (all_numerical_for_input_i, inp) in enumerate(zip(all_numerical, inp_tensors)):
        for j, n in enumerate(all_numerical_for_input_i):
            a = all_analytical[j][i]
            n = n.to(device=a.device)
            # TODO: Update adjusted atol
            if not allclose_with_type_promotion(a, n, rtol, adjusted_atol(atol, all_ur[i], all_v[j])):
                jacobians_str = slow_mode_jacobian_message(func, tupled_inputs, outputs, i, j, rtol, atol)
                raise GradcheckError(get_notallclose_msg(a, n, j, i, complex_indices, inp.is_complex()) + jacobians_str)
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
    For functions with complex output, gradcheck treats them as if they are two separate functions
    with real output. For more details, check out
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
    # This is just a wrapper that handles the raise_exception logic
    args = locals()
    args.pop("raise_exception")
    if not raise_exception:
        try:
            return gradcheck_helper(**args)
        except GradcheckError:
            return False
    else:
        return gradcheck_helper(**args)


def gradcheck_helper(func, inputs, eps, atol, rtol, check_sparse_nnz, nondet_tol, check_undefined_grad,
                     check_grad_dtypes, check_batched_grad, fast_mode):
    tupled_inputs = _as_tuple(inputs)
    check_inputs(tupled_inputs, check_sparse_nnz)

    func_out = func(*tupled_inputs)
    outputs = _differentiable_outputs(func_out)
    check_outputs(outputs)

    complex_indices = [i for i, o in enumerate(outputs) if o.is_complex()]
    any_complex = any(o.is_complex() for o in _as_tuple(func_out))
    gradcheck_fn = fast_gradcheck if fast_mode else slow_gradcheck
    gradcheck_real_imag(gradcheck_fn, func, func_out, tupled_inputs, outputs, eps,
                        rtol, atol, check_grad_dtypes, nondet_tol, complex_indices,
                        any_complex)

    for i, o in enumerate(outputs):
        if check_batched_grad:
            test_batched_grad(tupled_inputs, o, i)

    test_backward_mul_by_grad_output(outputs, tupled_inputs, check_sparse_nnz)

    if check_undefined_grad:
        test_undefined_grad(func, outputs, tupled_inputs)
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
            batched gradients using prototype vmap support. Defaults to False.
        fast_mode (bool, optional): if True, run a faster implementation of gradgradcheck that
            no longer computes the entire jacobian.

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
