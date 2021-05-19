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
    # Custom error so that user errors are not caught in the gradcheck's try-catch
    pass


def _is_float_or_complex_tensor(obj):
    return is_tensor_like(obj) and (obj.is_floating_point() or obj.is_complex())


def _allocate_jacobians_with_inputs(input_tensors: Tuple, numel_output) -> Tuple[torch.Tensor, ...]:
    # Makes zero-filled tensors from inputs. If `numel_output` is not None, for
    # each tensor in `input_tensors`, returns a new zero-filled tensor with height
    # of `t.numel` and width of `numel_output`. Otherwise, for each tensor, returns
    # a 1-d tensor with size `(t.numel,)`. Each new tensor will be strided and have
    # the same dtype and device as those of the corresponding input.
    out: List[torch.Tensor] = []
    for t in input_tensors:
        if _is_float_or_complex_tensor(t) and t.requires_grad:
            out.append(t.new_zeros((t.numel(), numel_output), layout=torch.strided))
    return tuple(out)


def _allocate_jacobians_with_outputs(output_tensors: Tuple, numel_input, dtype=None,
                                     device=None) -> Tuple[torch.Tensor, ...]:
    # Makes zero-filled tensors from outputs. If `dim` is not None, for each tensor
    # in `output_tensors`, returns a new zero-filled tensor with height of `dim` and
    # width of `t.numel`. Otherwise, for each tensor, returns a 1-d tensor with size
    # (t.numel,).
    out: List[torch.Tensor] = []
    options = {"dtype": dtype, "device": device, "layout": torch.strided}
    for t in output_tensors:
        if _is_float_or_complex_tensor(t):
            out.append(t.new_zeros((numel_input, t.numel()), **options))
    return tuple(out)


def _iter_tensors(x: Union[torch.Tensor, Iterable[torch.Tensor]],
                  only_requiring_grad: bool = False) -> Iterable[torch.Tensor]:
    if is_tensor_like(x):
        # mypy doesn't narrow type of `x` to torch.Tensor
        if x.requires_grad or not only_requiring_grad:  # type: ignore[union-attr]
            yield x  # type: ignore[misc]
    elif isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
        for elem in x:
            for result in _iter_tensors(elem, only_requiring_grad):
                yield result


def _iter_tensor(x_tensor):
    # (Only used for slow gradcheck) Returns a generator that yields the following
    # elements at each iteration:
    #  1) a tensor: the same tensor is returned across all iterations. The tensor
    #     is not the same as the original x_tensor as given as input - it is
    #     prepared so that it can be modified in-place. Depending on whether the
    #     input tensor is strided, sparse, or dense, the returned tensor may or may
    #     not share storage with x_tensor.
    #  2) a tuple of indices that can be used with advanced indexing (yielded in
    #     dictionary order)
    #  3) flattened index that will be used to index into the Jacobian tensor
    #
    # For a tensor t with size (2, 2), _iter_tensor yields:
    #     `x, (0, 0), 0`, `x, (0, 1), 1`, `x, (1, 0), 2`, `x, (1, 1), 3`
    #
    # where x is the t.data of the original tensor. Perturbing the entry of x
    # at index (1, 1) yields the 3rd column of the overall Jacobian matrix.
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
    elif x_tensor.layout == torch._mkldnn:  # type: ignore[attr-defined]
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
                            is_forward_ad=False) -> List[Tuple[torch.Tensor, ...]]:
    """Computes the numerical Jacobian of `fn(inputs)` with respect to `target`. If
    not specified, targets are the input. Returns M * N Jacobians where N is the
    number of tensors in target that require grad and M is the number of non-integral
    outputs.

    Args:
        fn: the function to compute the jacobian for
        inputs: inputs to `fn`
        outputs: provide precomputed outputs to avoid one extra invocation of fn
        target: the Tensors wrt whom Jacobians are calculated (default=`inputs`)
        eps: the magnitude of the perturbation during finite differencing
             (default=`1e-3`)
        is_forward_ad: if this numerical jacobian is computed to be checked wrt
                       forward AD gradients (this is used for error checking only)

    Returns:
        A list of M N-tuples of tensors

    Note that `target` may not even be part of `input` to `fn`, so please be
    **very careful** in this to not clone `target`.
    """
    jacobians: List[Tuple[torch.Tensor, ...]] = []
    if outputs is None:
        outputs = _as_tuple(fn(*_as_tuple(inputs)))
    if not is_forward_ad and any(o.is_complex() for o in outputs):
        raise ValueError("Expected output to be non-complex. get_numerical_jacobian no "
                         "longer supports functions that return complex outputs.")
    if target is None:
        target = inputs
    inp_indices = [i for i, a in enumerate(target) if is_tensor_like(a) and a.requires_grad]
    for i, (inp, inp_idx) in enumerate(zip(_iter_tensors(target, True), inp_indices)):
        jacobians += [get_numerical_jacobian_wrt_specific_input(fn, inp_idx, inputs, outputs, eps,
                                                                input=inp, is_forward_ad=is_forward_ad)]
    return jacobians


def get_numerical_jacobian(fn, inputs, target=None, eps=1e-3, grad_out=1.0):
    """Deprecated API to compute the numerical Jacobian for a given fn and its inputs.

    Args:
        fn: the function to compute the Jacobian for (must take inputs as a tuple)
        input: input to `fn`
        target: the Tensors wrt whom Jacobians are calculated (default=`input`)
        eps: the magnitude of the perturbation during finite differencing
             (default=`1e-3`)

    Returns:
        A list of Jacobians of `fn` (restricted to its first output) with respect to
        each input or target, if provided.

    Note that `target` may not even be part of `input` to `fn`, so please be
    **very careful** in this to not clone `target`.
    """
    warnings.warn("get_numerical_jacobian was part of PyTorch's private API and not "
                  "meant to be exposed. We are deprecating it and it will be removed "
                  "in a future version of PyTorch. If you have a specific use for "
                  "this or feature request for this to be a stable API, please file "
                  "us an issue at https://github.com/pytorch/pytorch/issues/new")
    if grad_out != 1.0:  # grad_out param is only kept for backward compatibility reasons
        raise ValueError("Expected grad_out to be 1.0. get_numerical_jacobian no longer "
                         "supports values of grad_out != 1.0.")

    def fn_pack_inps(*inps):
        return fn(inps)
    jacobians = _get_numerical_jacobian(fn_pack_inps, inputs, None, target, eps)

    return tuple(jacobian_for_each_output[0] for jacobian_for_each_output in jacobians)


def _compute_numerical_gradient(fn, entry, v, norm_v, nbhd_checks_fn):
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


def _compute_numerical_jvps_wrt_specific_input(jvp_fn, delta, input_is_complex,
                                               is_forward_ad=False) -> List[torch.Tensor]:
    # Computing the jacobian only works for real delta
    # For details on the algorithm used here, refer:
    # Section 3.5.3 https://arxiv.org/pdf/1701.00392.pdf
    # s = fn(z) where z = x for real valued input
    # and z = x + yj for complex valued input
    jvps: List[torch.Tensor] = []
    ds_dx_tup = jvp_fn(delta[0] if isinstance(delta, tuple) else delta)

    if input_is_complex:  # C -> R
        ds_dy_tup = jvp_fn(delta[1] * 1j) if isinstance(delta, tuple) else jvp_fn(delta * 1j)
        for ds_dx, ds_dy in zip(ds_dx_tup, ds_dy_tup):
            assert(not ds_dx.is_complex())
            # conjugate wirtinger derivative
            conj_w_d = ds_dx + ds_dy * 1j
            jvps.append(conj_w_d)
    else:
        for ds_dx in ds_dx_tup:  # R -> R or (R -> C for the forward AD case)
            assert(is_forward_ad or not ds_dx.is_complex())
            jvps.append(ds_dx)
    return jvps


def _combine_jacobian_cols(jacobians_cols: Dict[int, List[torch.Tensor]], outputs, input,
                           numel) -> Tuple[torch.Tensor, ...]:
    # jacobian_cols maps column_idx -> output_idx -> single column of jacobian Tensor
    # we return a list that maps output_idx -> full jacobian Tensor
    jacobians = _allocate_jacobians_with_outputs(outputs, numel, dtype=input.dtype if input.dtype.is_complex else None)
    for i, jacobian in enumerate(jacobians):
        for k, v in jacobians_cols.items():
            jacobian[k] = v[i]
    return jacobians


def _prepare_input(input: torch.Tensor, maybe_perturbed_input: Optional[torch.Tensor],
                   fast_mode=False) -> torch.Tensor:
    # Prepares the inputs to be passed into the function while including the new
    # modified input.
    if input.layout == torch._mkldnn:  # type: ignore[attr-defined] # no attr _mkldnn
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


def get_numerical_jacobian_wrt_specific_input(fn, input_idx, inputs, outputs, eps,
                                              input=None, is_forward_ad=False) -> Tuple[torch.Tensor, ...]:
    # Computes the numerical jacobians wrt to a single input. Returns N jacobian
    # tensors, where N is the number of outputs. We use a dictionary for
    # jacobian_cols because indices aren't necessarily consecutive for sparse inputs
    # When we perturb only a single element of the input tensor at a time, the jvp
    # is equivalent to a single col of the Jacobian matrix of fn.
    jacobian_cols: Dict[int, List[torch.Tensor]] = {}
    input = inputs[input_idx] if input is None else input
    assert input.requires_grad
    for x, idx, d_idx in _iter_tensor(input):
        wrapped_fn = _with_prepare_inputs(fn, inputs, input_idx, x)
        input_to_perturb = x[idx]
        nbhd_checks_fn = functools.partial(check_outputs_same_dtype_and_shape, idx=idx, eps=eps)
        jvp_fn = _get_numerical_jvp_fn(wrapped_fn, input_to_perturb, eps, nbhd_checks_fn)
        jacobian_cols[d_idx] = _compute_numerical_jvps_wrt_specific_input(jvp_fn, eps, x.is_complex(), is_forward_ad)
    return _combine_jacobian_cols(jacobian_cols, outputs, input, input.numel())

def _get_analytical_jacobian_forward_ad(fn, inputs, outputs, *, check_grad_dtypes=False,
                                        all_u=None) -> Tuple[Tuple[torch.Tensor, ...], ...]:
    """Computes the analytical Jacobian using forward mode AD of `fn(inputs)` using forward mode AD with respect
    to `target`. Returns N * M Jacobians where N is the number of tensors in target that require grad and
    M is the number of non-integral outputs.
    Contrary to other functions here, this function requires "inputs" to actually be used by the function.
    The computed value is expected to be wrong if the function captures the inputs by side effect instead of
    using the passed ones (many torch.nn tests do this).

    Args:
        fn: the function to compute the jacobian for
        inputs: inputs to `fn`
        outputs: provide precomputed outputs to avoid one extra invocation of fn
        check_grad_dtypes: if True, will check that the gradient dtype are valid
        all_u (optional): if provided, the Jacobian will be right multiplied with this vector

    Returns:
        A tuple of M N-tuples of tensors
    """
    # To avoid early import issues
    fwAD = torch.autograd.forward_ad

    tensor_inputs = tuple(i for i in inputs if is_tensor_like(i) and i.requires_grad)

    if any(i.is_complex() for i in tensor_inputs):
        raise ValueError("Expected inputs to be non-complex for _get_analytical_jacobian_forward_ad.")

    if all_u:
        jacobians = tuple(_allocate_jacobians_with_outputs(outputs, 1) for i in tensor_inputs)
    else:
        jacobians = tuple(_allocate_jacobians_with_outputs(outputs, i.numel()) for i in tensor_inputs)

    with fwAD.dual_level():
        fw_grads = []
        dual_inputs = []
        for i, inp in enumerate(inputs):
            if is_tensor_like(inp) and inp.requires_grad:
                if inp.layout == torch._mkldnn:  # type: ignore[attr-defined]
                    raise ValueError("MKLDNN inputs are not support for forward AD gradcheck.")

                inp = fwAD.make_dual(inp, torch.zeros_like(inp))
                # If inp is a differentiable view, the dual might not be the tangent given to
                # make_dual, so read it explicitly from the dual tensor
                fw_grads.append(fwAD.unpack_dual(inp)[1])
            dual_inputs.append(inp)

        if all_u:
            # Do the full reduction in one pass
            # To be consistent with numerical evaluation, we actually compute one reduction per input
            for i, (fw_grad, u) in enumerate(zip(fw_grads, all_u)):
                fw_grad.copy_(u.view_as(fw_grad))
                dual_outputs = _as_tuple(fn(*dual_inputs))
                for index_o, d_o in enumerate(dual_outputs):
                    val, res = fwAD.unpack_dual(d_o)
                    if check_grad_dtypes and val.is_complex() != res.is_complex():
                        raise GradcheckError('Forward AD gradient has dtype mismatch.')

                    # Remove extra dimension of size 1 corresponding to the reduced input
                    jacobians[i][index_o].squeeze_(0)
                    if res is None:
                        jacobians[i][index_o].zero_()
                    else:
                        jacobians[i][index_o].copy_(res.reshape(-1))
                fw_grad.zero_()
        else:
            # Reconstruct the full Jacobian column by column
            for i, fw_grad in enumerate(fw_grads):
                for lin_idx, grad_idx in enumerate(product(*[range(m) for m in fw_grad.size()])):
                    fw_grad[grad_idx] = 1.
                    dual_outputs = _as_tuple(fn(*dual_inputs))
                    for index_o, d_o in enumerate(dual_outputs):
                        _, res = fwAD.unpack_dual(d_o)
                        if res is None:
                            jacobians[i][index_o][lin_idx].zero_()
                        else:
                            jacobians[i][index_o][lin_idx].copy_(res.reshape(-1))
                    fw_grad[grad_idx] = 0.

    return jacobians

def _get_input_to_perturb(input):
    # Prepare the input so that it can be modified in-place and do certain
    # operations that require the tensor to have strides. If fast_mode=False,
    # _iter_tensor would handle the below cases:
    if input.layout == torch._mkldnn:  # type: ignore[attr-defined] # no attr _mkldnn
        # Convert to dense so we can perform operations that require strided tensors
        input_to_perturb = input.to_dense()
    elif input.layout == torch.sparse_coo:
        # Clone because input may require grad, and copy_ calls resize_,
        # which is not allowed for .data
        input_to_perturb = input.clone()
    else:
        input_to_perturb = input.data
    return input_to_perturb


def _with_prepare_inputs(fn, inputs, input_idx, input_to_perturb, fast_mode=False):
    # Wraps `fn` so that its inputs are already supplied
    def wrapped_fn():
        inp = tuple(_prepare_input(a, input_to_perturb if i == input_idx else None, fast_mode)
                    if is_tensor_like(a) else a for i, a in enumerate(_as_tuple(inputs)))
        return tuple(a.clone() for a in _as_tuple(fn(*inp)))
    return wrapped_fn


def _get_numerical_jvp_fn(wrapped_fn, input_to_perturb, eps, nbhd_checks_fn):
    # Wraps jvp_fn so that certain arguments are already supplied
    def jvp_fn(delta):
        return _compute_numerical_gradient(wrapped_fn, input_to_perturb, delta, eps, nbhd_checks_fn)
    return jvp_fn


def _reshape_tensor_or_tuple(u, shape):
    # We don't need to reshape when input corresponding to u is sparse
    if isinstance(u, tuple):
        if u[0].layout != torch.sparse_coo:
            return (u[0].reshape(shape), u[1].reshape(shape))
    else:
        if u.layout != torch.sparse_coo:
            return u.reshape(shape)
    return u


def _mul_tensor_or_tuple(u, k):
    if isinstance(u, tuple):
        return (k * u[0], k * u[1])
    else:
        return k * u


def _get_numerical_jvp_wrt_specific_input(fn, input_idx, inputs, outputs, u, eps, is_forward_ad=False) -> List[torch.Tensor]:
    input = inputs[input_idx]
    input_to_perturb = _get_input_to_perturb(input)
    wrapped_fn = _with_prepare_inputs(fn, inputs, input_idx, input_to_perturb, True)
    nbhd_checks_fn = functools.partial(check_outputs_same_dtype_and_shape, eps=eps)
    jvp_fn = _get_numerical_jvp_fn(wrapped_fn, input_to_perturb, eps, nbhd_checks_fn)
    u = _reshape_tensor_or_tuple(u, input_to_perturb.shape)
    u = _mul_tensor_or_tuple(u, eps)
    return _compute_numerical_jvps_wrt_specific_input(jvp_fn, u, input.is_complex(), is_forward_ad)


def _get_numerical_vJu(fn, inputs, inp_indices, outputs, all_u, all_v, eps, is_forward_ad):
    # Note that all_v can also be None, in that case, this function only computes Ju.
    reduced_jacobians: List[List[torch.Tensor]] = []
    for i, (inp_idx, u) in enumerate(zip(inp_indices, all_u)):
        all_Ju = _get_numerical_jvp_wrt_specific_input(fn, inp_idx, inputs, outputs, u, eps, is_forward_ad)
        if all_v is not None:
            jacobian_scalars: List[torch.Tensor] = []
            for v, Ju in zip(all_v, all_Ju):
                jacobian_scalars.append(_dot_with_type_promotion(v, Ju))
            reduced_jacobians.append(jacobian_scalars)
        else:
            reduced_jacobians.append(all_Ju)
    return reduced_jacobians


def _check_jacobians_equal(j1, j2, atol):
    # Check whether the max difference between two Jacobian tensors are within some
    # tolerance `atol`.
    for j1_x, j2_x in zip(j1, j2):
        if j1_x.numel() != 0 and (j1_x - j2_x).abs().max() > atol:
            return False
    return True


def _stack_and_check_tensors(list_of_list_of_tensors, inputs,
                             numel_outputs) -> Tuple[Tuple[torch.Tensor, ...], bool, bool]:
    # For the ith tensor in the inner list checks whether it has the same size and
    # dtype as the ith differentiable input.
    out_jacobians = _allocate_jacobians_with_inputs(inputs, numel_outputs)
    diff_input_list = list(_iter_tensors(inputs, True))
    correct_grad_sizes = True
    correct_grad_types = True
    for i, tensor_list in enumerate(list_of_list_of_tensors):
        inp = diff_input_list[i]
        out_jacobian = out_jacobians[i]
        for j, tensor in enumerate(tensor_list):
            if tensor is not None and tensor.size() != inp.size():
                correct_grad_sizes = False
            elif tensor is not None and tensor.dtype != inp.dtype:
                correct_grad_types = False
            if tensor is None:
                out_jacobian[:, j].zero_()
            else:
                dense = tensor.to_dense() if not tensor.layout == torch.strided else tensor
                assert out_jacobian[:, j].numel() == dense.numel()
                out_jacobian[:, j] = dense.reshape(-1)
    return out_jacobians, correct_grad_sizes, correct_grad_types


FAILED_NONDET_MSG = """\n
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


def _check_analytical_jacobian_attributes(inputs, output, nondet_tol, check_grad_dtypes,
                                          fast_mode=False, v=None) -> Tuple[torch.Tensor, ...]:
    # This is used by both fast and slow mode:
    #  - For slow mode, vjps[i][j] is the jth row the Jacobian wrt the ith
    #    input.
    #  - For fast mode, vjps[i][0] is a linear combination of the rows
    #    of the Jacobian wrt the ith input
    diff_input_list = list(_iter_tensors(inputs, True))

    def vjp_fn(grad_output):
        return torch.autograd.grad(output, diff_input_list, grad_output,
                                   retain_graph=True, allow_unused=True)
    # Compute everything twice to check for nondeterminism (which we call reentrancy)
    if fast_mode:
        vjps1 = _get_analytical_vjps_wrt_specific_output(vjp_fn, output.clone(), v)
        vjps2 = _get_analytical_vjps_wrt_specific_output(vjp_fn, output.clone(), v)
    else:
        vjps1 = _compute_analytical_jacobian_rows(vjp_fn, output.clone())
        vjps2 = _compute_analytical_jacobian_rows(vjp_fn, output.clone())

    output_numel = output.numel() if not fast_mode else 1
    jacobians1, types_ok, sizes_ok = _stack_and_check_tensors(vjps1, inputs, output_numel)
    jacobians2, _, _ = _stack_and_check_tensors(vjps2, inputs, output_numel)
    reentrant = _check_jacobians_equal(jacobians1, jacobians2, nondet_tol)

    if not types_ok and check_grad_dtypes:
        raise GradcheckError('Gradient has dtype mismatch')
    if not sizes_ok:
        raise GradcheckError('Analytical gradient has incorrect size')
    if not reentrant:
        raise GradcheckError('Backward is not reentrant, i.e., running backward with '
                             'same input and grad_output multiple times gives different values, '
                             'although analytical gradient matches numerical gradient.'
                             f'The tolerance for nondeterminism was {nondet_tol}.' +
                             FAILED_NONDET_MSG)
    return jacobians1


def _get_analytical_vJu_backward_mode(inputs, outputs, nondet_tol, check_grad_dtypes, all_v, all_u):
    reduced_jacobians: List[List[torch.Tensor]] = []
    for output, v in zip(outputs, all_v):
        all_vJ = _check_analytical_jacobian_attributes(inputs, output, nondet_tol, check_grad_dtypes,
                                                       fast_mode=True, v=v)
        jacobian_scalars: List[torch.Tensor] = []
        for vJ, u in zip(all_vJ, all_u):
            # Why do we need squeeze here? vJ is a 2-d tensor so that we can reuse
            # the error checking logic from slow mode
            vJ = vJ.T.squeeze(0)
            if vJ.is_complex():  # C -> R
                tv = torch.view_as_real(vJ)
                tr = tv.select(-1, 0)
                ti = tv.select(-1, 1)
                jacobian_scalars.append(tr.dot(u[0]) + 1j * ti.dot(u[1]))
            else:  # R -> R
                jacobian_scalars.append(vJ.dot(u))
        reduced_jacobians.append(jacobian_scalars)
    return reduced_jacobians

def get_analytical_jacobian(inputs, output, nondet_tol=0.0, grad_out=1.0):
    # Replicates the behavior of the old get_analytical_jacobian before the refactor
    # This shares much of its code with _check_analytical_jacobian_attributes
    warnings.warn("get_analytical_jacobian was part of PyTorch's private API and not "
                  "meant to be exposed. We are deprecating it and it will be removed "
                  "in a future version of PyTorch. If you have a specific use for "
                  "this or feature request for this to be a stable API, please file "
                  "us an issue at https://github.com/pytorch/pytorch/issues/new")
    if grad_out != 1.0:  # grad_out param is only kept for backward compatibility reasons
        raise ValueError("Expected grad_out to be 1.0. get_analytical_jacobian no longer "
                         "supports values of grad_out != 1.0.")
    if output.is_complex():
        raise ValueError("Expected output to be non-complex. get_analytical_jacobian no "
                         "longer supports functions that return complex outputs.")
    diff_input_list = list(_iter_tensors(inputs, True))

    def vjp_fn(grad_output):
        return torch.autograd.grad(output, diff_input_list, grad_output,
                                   retain_graph=True, allow_unused=True)
    # Compute everything twice to check for nondeterminism (which we call reentrancy)
    vjps1 = _compute_analytical_jacobian_rows(vjp_fn, output.clone())
    vjps2 = _compute_analytical_jacobian_rows(vjp_fn, output.clone())

    output_numel = output.numel()
    jacobians1, types_ok, sizes_ok = _stack_and_check_tensors(vjps1, inputs, output_numel)
    jacobians2, _, _ = _stack_and_check_tensors(vjps2, inputs, output_numel)
    reentrant = _check_jacobians_equal(jacobians1, jacobians2, nondet_tol)

    return jacobians1, reentrant, sizes_ok, types_ok


def _get_analytical_jacobian(inputs, outputs, input_idx, output_idx):
    # Computes the analytical Jacobian in slow mode for a single input-output pair.
    # Forgoes performing checks on dtype, shape, and reentrancy.
    jacobians = _check_analytical_jacobian_attributes(inputs, outputs[output_idx],
                                                      nondet_tol=float('inf'), check_grad_dtypes=False)
    return jacobians[input_idx]


def _compute_analytical_jacobian_rows(vjp_fn, sample_output) -> List[List[Optional[torch.Tensor]]]:
    # Computes Jacobian row-by-row using backward function `vjp_fn` = v^T J
    # NB: this function does not assume vjp_fn(v) to return tensors with the same
    # number of elements for different v. This is checked when we later combine the
    # rows into a single tensor.
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


def _get_analytical_vjps_wrt_specific_output(vjp_fn, sample_output, v) -> List[List[Optional[torch.Tensor]]]:
    vjps: List[List[Optional[torch.Tensor]]] = []
    grad_inputs = vjp_fn(v.reshape(sample_output.shape))
    for vjp in grad_inputs:
        vjps.append([vjp.clone() if isinstance(vjp, torch.Tensor) else None])
    return vjps


def _check_inputs(tupled_inputs, check_sparse_nnz) -> bool:
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
            if content.layout is not torch._mkldnn:  # type: ignore[attr-defined]
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


def _check_outputs(outputs) -> None:
    if any(t.layout == torch.sparse_coo for t in outputs if isinstance(t, torch.Tensor)):
        # it is easier to call to_dense() on the sparse output than
        # to modify analytical jacobian
        raise ValueError('Sparse output is not supported at gradcheck yet. '
                         'Please call to_dense() on the output of fn for gradcheck.')
    if any(t.layout == torch._mkldnn for t in outputs if isinstance(t, torch.Tensor)):  # type: ignore[attr-defined]
        raise ValueError('MKLDNN output is not supported at gradcheck yet. '
                         'Please call to_dense() on the output of fn for gradcheck.')


def _check_no_differentiable_outputs(func, inputs, func_out, eps) -> bool:
    # When there are no differentiable outputs, numerical gradient for a function is
    # expected to be zero.
    jacobians_all_inputs_outputs = _get_numerical_jacobian(func, inputs, func_out, eps=eps)
    for jacobians_all_outputs_and_fixed_input in jacobians_all_inputs_outputs:
        for jacobian in jacobians_all_outputs_and_fixed_input:
            if torch.ne(jacobian, 0).sum() > 0:
                raise GradcheckError('Numerical gradient for function expected to be zero')
    return True


def _check_no_differentiable_outputs_fast(func, func_out, all_inputs, inputs_indices,
                                          all_u, eps, nondet_tol):
    for inp_idx, u in zip(inputs_indices, all_u):
        jvps = _get_numerical_jvp_wrt_specific_input(func, inp_idx, all_inputs, _as_tuple(func_out), u, eps)
        for jvp in jvps:
            if jvp.numel() == 0:
                continue
            if (jvp - torch.zeros_like(jvp)).abs().max() > nondet_tol:
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

def _get_failed_batched_grad_test_msg(output_idx, input_idx, res, exp):
    return f"""
For output {output_idx} and input {input_idx}:

{FAILED_BATCHED_GRAD_MSG}

Got:
{res}

Expected:
{exp}
""".strip()


def _test_batched_grad(input, output, output_idx) -> bool:
    # NB: _test_batched_grad compares two autograd.grad invocations with a single
    # vmap(autograd.grad) invocation. It's not exactly a "gradcheck" in the
    # sense that we're not comparing an analytical jacobian with a numeric one,
    # but it is morally similar (we could have computed a full analytic jac
    # via vmap, but that is potentially slow)
    diff_input_list = list(_iter_tensors(input, True))
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
        raise GradcheckError(_get_failed_batched_grad_test_msg(output_idx, input_idx, res, exp))
    return True


def _test_backward_mul_by_grad_output(outputs, inputs, check_sparse_nnz) -> bool:
    # Tests that backward is multiplied by grad_output
    diff_input_list: List[torch.Tensor] = list(_iter_tensors(inputs, True))
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


def _test_undefined_grad(func, outputs, inputs) -> bool:
    diff_input_list: List[torch.Tensor] = list(_iter_tensors(inputs, True))
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


def _get_notallclose_msg(analytical, numerical, output_idx, input_idx, complex_indices,
                         test_imag=False, is_forward_ad=False) -> str:
    out_is_complex = (not is_forward_ad) and complex_indices and output_idx in complex_indices
    inp_is_complex = is_forward_ad and complex_indices and input_idx in complex_indices
    part = "imaginary" if test_imag else "real"
    element = "inputs" if is_forward_ad else "outputs"
    prefix = "" if not (out_is_complex or inp_is_complex) else \
        f"While considering the {part} part of complex {element} only, "
    mode = "computed with forward mode " if is_forward_ad else ""
    return prefix + 'Jacobian %smismatch for output %d with respect to input %d,\n' \
        'numerical:%s\nanalytical:%s\n' % (mode, output_idx, input_idx, numerical, analytical)


def _transpose(matrix_of_tensors):
    # returns list of tuples
    return list(zip(*matrix_of_tensors))


def _real_and_imag_output(fn):
    # returns new functions real(fn), and imag(fn) where real(fn) and imag(fn) behave the same as
    # the original fn, except torch.real or torch.imag are applied to the complex outputs
    def apply_to_c_outs(fn, fn_to_apply):
        def wrapped_fn(*inputs):
            outs = _as_tuple(fn(*inputs))
            return tuple(fn_to_apply(o) if o.is_complex() else o for o in outs)
        return wrapped_fn
    return apply_to_c_outs(fn, torch.real), apply_to_c_outs(fn, torch.imag)

def _real_and_imag_input(fn, complex_inp_indices):
    # returns new functions that take real inputs instead of complex inputs and compute fn(x + 0 * 1j)
    # and f(x * 1j).
    def apply_to_c_inps(fn, fn_to_apply):
        def wrapped_fn(*inputs):
            new_inputs = list(inputs)
            for should_be_complex in complex_inp_indices:
                new_inputs[should_be_complex] = fn_to_apply(new_inputs[should_be_complex])
            return _as_tuple(fn(*new_inputs))
        return wrapped_fn
    return apply_to_c_inps(fn, lambda x: x + 0 * 1j), apply_to_c_inps(fn, lambda x: x * 1j)


def _gradcheck_real_imag(gradcheck_fn, func, func_out, tupled_inputs, outputs, eps, rtol,
                         atol, check_grad_dtypes, check_forward_ad, nondet_tol):
    complex_out_indices = [i for i, o in enumerate(outputs) if o.is_complex()]
    has_any_complex_output = any(o.is_complex() for o in _as_tuple(func_out))
    if has_any_complex_output:
        real_fn, imag_fn = _real_and_imag_output(func)

        imag_func_out = imag_fn(*tupled_inputs)
        imag_outputs = _differentiable_outputs(imag_func_out)
        gradcheck_fn(imag_fn, imag_func_out, tupled_inputs, imag_outputs, eps,
                     rtol, atol, check_grad_dtypes, nondet_tol,
                     complex_indices=complex_out_indices, test_imag=True)

        real_func_out = real_fn(*tupled_inputs)
        real_outputs = _differentiable_outputs(real_func_out)
        gradcheck_fn(real_fn, real_func_out, tupled_inputs, real_outputs, eps,
                     rtol, atol, check_grad_dtypes, nondet_tol, complex_indices=complex_out_indices)
    else:
        gradcheck_fn(func, func_out, tupled_inputs, outputs, eps,
                     rtol, atol, check_grad_dtypes, nondet_tol)

    if check_forward_ad:
        complex_inp_indices = [i for i, inp in enumerate(tupled_inputs) if is_tensor_like(inp) and inp.is_complex()]
        if complex_inp_indices:
            real_fn, imag_fn = _real_and_imag_input(func, complex_inp_indices)

            imag_inputs = [inp.imag if is_tensor_like(inp) and inp.is_complex() else inp for inp in tupled_inputs]
            imag_func_out = imag_fn(*imag_inputs)
            diff_imag_func_out = _differentiable_outputs(imag_func_out)
            gradcheck_fn(imag_fn, imag_func_out, imag_inputs, diff_imag_func_out, eps,
                         rtol, atol, check_grad_dtypes, nondet_tol,
                         complex_indices=complex_inp_indices, test_imag=True, use_forward_ad=True)

            real_inputs = [inp.real if is_tensor_like(inp) and inp.is_complex() else inp for inp in tupled_inputs]
            real_func_out = real_fn(*real_inputs)
            diff_real_func_out = _differentiable_outputs(real_func_out)
            gradcheck_fn(real_fn, real_func_out, real_inputs, diff_real_func_out, eps,
                         rtol, atol, check_grad_dtypes, nondet_tol, complex_indices=complex_inp_indices,
                         use_forward_ad=True)
        else:
            gradcheck_fn(func, func_out, tupled_inputs, outputs, eps,
                         rtol, atol, check_grad_dtypes, nondet_tol, use_forward_ad=True)


def _slow_gradcheck(func, func_out, tupled_inputs, outputs, eps, rtol, atol, check_grad_dtypes,
                    nondet_tol, *, use_forward_ad=False, complex_indices=None, test_imag=False):
    if not outputs:
        return _check_no_differentiable_outputs(func, tupled_inputs, _as_tuple(func_out), eps)

    numerical = _transpose(_get_numerical_jacobian(func, tupled_inputs, outputs, eps=eps, is_forward_ad=use_forward_ad))

    if use_forward_ad:
        analytical_forward = _get_analytical_jacobian_forward_ad(func, tupled_inputs, outputs, check_grad_dtypes=check_grad_dtypes)

        for i, n_per_out in enumerate(numerical):
            for j, n in enumerate(n_per_out):
                a = analytical_forward[j][i]
                if not _allclose_with_type_promotion(a, n.to(a.device), rtol, atol):
                    raise GradcheckError(_get_notallclose_msg(a, n, i, j, complex_indices, test_imag,
                                                              is_forward_ad=True))
    else:
        for i, o in enumerate(outputs):
            analytical = _check_analytical_jacobian_attributes(tupled_inputs, o, nondet_tol, check_grad_dtypes)

            for j, (a, n) in enumerate(zip(analytical, numerical[i])):
                if not _allclose_with_type_promotion(a, n.to(a.device), rtol, atol):
                    raise GradcheckError(_get_notallclose_msg(a, n, i, j, complex_indices, test_imag))

    return True


def _dot_with_type_promotion(u, v):
    assert u.dim() == 1 and v.dim() == 1
    return (u * v).sum()


def _allclose_with_type_promotion(a, b, rtol, atol):
    promoted_type = torch.promote_types(a.dtype, b.dtype)
    a = a.to(dtype=promoted_type)
    b = b.to(dtype=promoted_type)
    return torch.allclose(a, b, rtol, atol)


def _to_real_dtype(dtype):
    if dtype == torch.complex128:
        return torch.float64
    elif dtype == torch.complex64:
        return torch.float32
    else:
        return dtype

def _vec_from_tensor(x, generator, downcast_complex=False):
    # Create a random vector with the same number of elements as x and the same
    # dtype/device. If x is complex and downcast_complex is False, we create a
    # complex tensor with only real component.
    if x.layout == torch.sparse_coo:
        # For sparse, create a random sparse vec with random values in the same
        # indices. Make sure size is set so that it isn't inferred to be smaller.
        x_values = x._values()
        dtype = _to_real_dtype(x.dtype) if downcast_complex else x.dtype
        values = torch.rand(x_values.numel(), generator=generator) \
            .to(dtype=dtype, device=x.device) \
            .reshape(x_values.shape)
        values /= values.norm()
        vec = torch.sparse_coo_tensor(x._indices(), values, x.size())
    else:
        dtype = _to_real_dtype(x.dtype) if downcast_complex else x.dtype
        vec = torch.rand(x.numel(), generator=generator).to(dtype=dtype, device=x.device)
        vec /= vec.norm()
    return vec


def _get_inp_tensors(tupled_inputs):
    inp_idx_tup = [(i, t) for i, t in enumerate(tupled_inputs) if is_tensor_like(t) and t.requires_grad]
    return [tup[0] for tup in inp_idx_tup], [tup[1] for tup in inp_idx_tup]


def _adjusted_atol(atol, u, v):
    # In slow gradcheck, we compare A and B element-wise, i.e., for some a, b we
    # allow: |a - b| < atol + rtol * b. But since we now compare q1 = v^T A u and
    # q2 = v^T B u, we must allow |q1 - q2| < v^T E u + rtol * v^T B u, where E is
    # the correctly sized matrix in which each entry is atol.
    #
    # We see that atol needs to be scaled by v^T M u (where M is an all-ones M x N
    # matrix): v^T M u = \sum_{i} \sum_{j} u_i * v_j = (\sum_{i} u_i)(\sum_{i} v_i)
    # TODO: properly handle case when u is tuple instead of only taking first element
    u = u[0] if isinstance(u, tuple) else u
    sum_u = torch.sparse.sum(u) if u.layout == torch.sparse_coo else u.sum()
    sum_v = 1. if v is None else torch.sparse.sum(v) if v.layout == torch.sparse_coo else v.sum()
    return atol * float(sum_u) * float(sum_v)


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


def _run_slow_mode_and_get_error(func, tupled_inputs, outputs, input_idx, output_idx, rtol, atol, is_forward_ad):
    # Compute jacobians in slow mode for better error message
    slow_numerical = _get_numerical_jacobian(func, tupled_inputs, outputs, is_forward_ad=is_forward_ad)[input_idx][output_idx]
    if is_forward_ad:
        def new_fn(inp):
            new_inputs = list(tupled_inputs)
            new_inputs[input_idx] = inp
            return func(*new_inputs)[output_idx]
        slow_analytical = _get_analytical_jacobian_forward_ad(new_fn, (tupled_inputs[input_idx],), (outputs[output_idx],))[0][0]
    else:
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


def _to_flat_dense_if_sparse(tensor):
    if tensor.layout == torch.sparse_coo:
        return tensor.to_dense().reshape(-1)
    else:
        return tensor


def _make_vectors(inp_tensors, outputs, *, use_forward_ad):
    # Use our own generator to avoid messing with the user's RNG state
    g_cpu = torch.Generator()
    all_u = []
    all_u_dense = []
    for inp in inp_tensors:
        ur = _vec_from_tensor(inp, g_cpu, True)
        ur_dense = _to_flat_dense_if_sparse(ur)
        if inp.is_complex():
            ui = _vec_from_tensor(inp, g_cpu, True)
            all_u.append((ur, ui))
            ui_dense = _to_flat_dense_if_sparse(ui)
            all_u_dense.append((ur_dense, ui_dense))
        else:
            all_u.append(ur)
            all_u_dense.append(ur_dense)
    all_v = None if use_forward_ad else [_vec_from_tensor(out, g_cpu) for out in outputs]
    return all_v, all_u, all_u_dense


def _check_analytical_numerical_equal(all_analytical, all_numerical, complex_indices, tupled_inputs, outputs,
                                      func, all_v, all_u, rtol, atol, test_imag, *, is_forward_ad=False):
    for i, all_numerical_for_input_i in enumerate(all_numerical):
        for j, n in enumerate(all_numerical_for_input_i):
            # Forward AD generates the transpose of what this function expects
            if is_forward_ad:
                a = all_analytical[i][j]
            else:
                a = all_analytical[j][i]
            n = n.to(device=a.device)
            updated_atol = _adjusted_atol(atol, all_u[i], all_v[j] if all_v else None)
            if not _allclose_with_type_promotion(a, n.to(a.device), rtol, updated_atol):
                jacobians_str = _run_slow_mode_and_get_error(func, tupled_inputs, outputs, i, j, rtol, atol, is_forward_ad)
                raise GradcheckError(_get_notallclose_msg(a, n, j, i, complex_indices, test_imag, is_forward_ad) + jacobians_str)


def _fast_gradcheck(func, func_out, inputs, outputs, eps, rtol,
                    atol, check_grad_dtypes, nondet_tol, *, use_forward_ad=False, complex_indices=None, test_imag=False):
    # See https://github.com/pytorch/pytorch/issues/53876 for details
    inp_tensors_idx, inp_tensors = _get_inp_tensors(inputs)
    all_v, all_u, all_u_dense = _make_vectors(inp_tensors, outputs, use_forward_ad=use_forward_ad)

    numerical_vJu = _get_numerical_vJu(func, inputs, inp_tensors_idx, outputs, all_u, all_v, eps, is_forward_ad=use_forward_ad)
    if use_forward_ad:
        assert all_v is None
        analytical_vJu = _get_analytical_jacobian_forward_ad(func, inputs, outputs, all_u=all_u,
                                                             check_grad_dtypes=check_grad_dtypes)
    else:
        if not outputs:
            _check_no_differentiable_outputs_fast(func, func_out, inputs, inp_tensors_idx, all_u, eps, nondet_tol)

        analytical_vJu = _get_analytical_vJu_backward_mode(inputs, outputs, nondet_tol, check_grad_dtypes, all_v, all_u_dense)

    _check_analytical_numerical_equal(analytical_vJu, numerical_vJu, complex_indices,
                                      inputs, outputs, func, all_v, all_u, rtol, atol, test_imag, is_forward_ad=use_forward_ad)

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
    nondet_tol: float = 0.0,
    check_undefined_grad: bool = True,
    check_grad_dtypes: bool = False,
    check_batched_grad: bool = False,
    check_forward_ad: bool = False,
    fast_mode: bool = False,
) -> bool:
    r"""Check gradients computed via small finite differences against analytical
    gradients w.r.t. tensors in :attr:`inputs` that are of floating point or complex type
    and with ``requires_grad=True``.

    The check between numerical and analytical gradients uses :func:`~torch.allclose`.

    For most of the complex functions we consider for optimization purposes, no notion of
    Jacobian exists. Instead, gradcheck verifies if the numerical and analytical values of
    the Wirtinger and Conjugate Wirtinger derivatives are consistent. Because the gradient
    computation is done under the assumption that the overall function has a real-valued
    output, we treat functions with complex output in a special way. For these functions,
    gradcheck is applied to two real-valued functions corresponding to taking the real
    components of the complex outputs for the first, and taking the imaginary components
    of the complex outputs for the second. For more details, check out
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
        check_forward_ad (bool, optional): if True, check that the gradients computed with forward
            mode AD match the numerical ones. Defaults to False.
        fast_mode (bool, optional): Fast mode for gradcheck and gradgradcheck is currently only
            implemented for R to R functions. If none of the inputs and outputs are complex
            a faster implementation of gradcheck that no longer computes the entire jacobian
            is run; otherwise, we fall back to the slow implementation.

    Returns:
        True if all differences satisfy allclose condition
    """
    # This is just a wrapper that handles the raise_exception logic
    args = locals().copy()
    args.pop("raise_exception")
    if not raise_exception:
        try:
            return _gradcheck_helper(**args)
        except GradcheckError as e:
            return False
    else:
        return _gradcheck_helper(**args)


def _gradcheck_helper(func, inputs, eps, atol, rtol, check_sparse_nnz, nondet_tol, check_undefined_grad,
                      check_grad_dtypes, check_batched_grad, check_forward_ad, fast_mode):
    tupled_inputs = _as_tuple(inputs)
    _check_inputs(tupled_inputs, check_sparse_nnz)

    func_out = func(*tupled_inputs)
    outputs = _differentiable_outputs(func_out)
    _check_outputs(outputs)

    gradcheck_fn = _fast_gradcheck if fast_mode else _slow_gradcheck
    _gradcheck_real_imag(gradcheck_fn, func, func_out, tupled_inputs, outputs, eps,
                         rtol, atol, check_grad_dtypes, check_forward_ad=check_forward_ad, nondet_tol=nondet_tol)

    for i, o in enumerate(outputs):
        if check_batched_grad:
            _test_batched_grad(tupled_inputs, o, i)

    _test_backward_mul_by_grad_output(outputs, tupled_inputs, check_sparse_nnz)

    if check_undefined_grad:
        _test_undefined_grad(func, outputs, tupled_inputs)
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
        grad_inputs = torch.autograd.grad(outputs, input_args, grad_outputs, create_graph=True,
                                          allow_unused=True)
        grad_inputs = tuple(g for g in grad_inputs if g is not None)
        return grad_inputs

    return gradcheck(
        new_func, tupled_inputs + tupled_grad_outputs, eps, atol, rtol, raise_exception,
        nondet_tol=nondet_tol, check_undefined_grad=check_undefined_grad,
        check_grad_dtypes=check_grad_dtypes, check_batched_grad=check_batched_grad, fast_mode=fast_mode)
