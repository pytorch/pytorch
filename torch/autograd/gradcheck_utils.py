from dataclasses import dataclass
import torch
from typing import Callable, Tuple, Any
from torch.overrides import is_tensor_like
import functools

def _as_tuple(x):
    if isinstance(x, tuple):
        return x
    elif isinstance(x, list):
        return tuple(x)
    else:
        return x,

def _assert_list_length_equal(a, b, transpose=False):
    if transpose:
        if len(b) > 0:
            assert len(a) == len(b[0]), f"len(a): {len(a)} != len(b[0]): {len(b[0])}"
        if len(a) > 0:
            assert len(b) == len(a[0]), f"len(b): {len(b)} != len(a[0]): {len(a[0])}"
    else:
        assert len(a) == len(b), f"len(a): {len(a)} != len(b): {len(b)}"
        if len(a) > 0:
            assert len(a[0]) == len(b[0]), f"len(a[0]): {len(a[0])} != len(b[0]): {len(b[0])}"

def _safe_zip(*iterables):
    # Copied from https://www.python.org/dev/peps/pep-0618/
    # TODO: We can remove this once we drop support for versions of Python <=3.9
    #       and simply replace with zip(strict=True)
    strict=True
    if not iterables:
        return
    iterators = tuple(iter(iterable) for iterable in iterables)
    try:
        while True:
            items = []
            for iterator in iterators:
                items.append(next(iterator))
            yield tuple(items)
    except StopIteration:
        if not strict:
            return
    if items:
        i = len(items)
        plural = " " if i == 1 else "s 1-"
        msg = f"zip() argument {i+1} is shorter than argument{plural}{i}"
        raise ValueError(msg)
    sentinel = object()
    for i, iterator in enumerate(iterators[1:], 1):
        if next(iterator, sentinel) is not sentinel:
            plural = " " if i == 1 else "s 1-"
            msg = f"zip() argument {i+1} is longer than argument{plural}{i}"
            raise ValueError(msg)

def _is_differentiable(obj: Any, need_requires_grad: bool) -> bool:
    return _is_float_or_complex_tensor(obj) and (obj.requires_grad or not need_requires_grad)

def _is_float_or_complex_tensor(obj: Any) -> bool:
    return is_tensor_like(obj) and (obj.is_floating_point() or obj.is_complex())

def _tuple_indices_where(tup: Tuple[Any, ...], cond_fn: Callable[..., bool]):
    return tuple(i for (i, obj) in enumerate(tup) if cond_fn(obj))

def _tuple_apply_indices(tup: Tuple[Any, ...], indices: Tuple[int, ...]):
    return tuple(obj for (i, obj) in enumerate(tup) if i in indices)

def _is_tuple_indices_subset(tup: Tuple[int, ...], tup_subset: Tuple[int, ...]) -> bool:
    return all(i in set(tup) for i in tup_subset)

class GradcheckError(RuntimeError):
    # Custom error so that user errors are not caught in the gradcheck's try-catch
    pass

class GradcheckFunction(Callable):
    # Wraps a raw function so that the non-differentiable inputs are captured instead
    # of passed, and the non-differentiable outputs are filtered out.
    #
    # Whether we require the tensor to have requires_grad=True to count as differentiable
    # can be customized depending on whether this is being constructed for use with
    # forward-mode AD or reverse-mode AD.
    #
    # If we filter out inputs/outputs that don't require grad (i.e., need_requires_grad=True),
    # we also expose an alternative function `diff_out_indices_no_req_grad`,
    # for the _check_no_differentiable_outputs check

    fn_raw: Callable[..., Any]
    inputs_raw: Tuple[Any]
    outputs_raw: Tuple[Any]

    inputs: Tuple[torch.Tensor, ...]
    outputs: Tuple[torch.Tensor, ...]
    wrapped_fn: Callable[..., Tuple[torch.Tensor, ...]]
    wrapped_fn_out_no_req_grad: Callable[..., Tuple[torch.Tensor, ...]]

    def __init__(self, fn_raw, inputs_raw, outputs_raw, custom_diff_in_indices, custom_diff_out_indices, need_requires_grad):
        self.fn_raw = fn_raw
        self.inputs_raw = inputs_raw
        self.outputs_raw = outputs_raw

        # NB: maybe_req_grad = require to have requires_grad=True when used for bw but not fw AD
        is_diff_maybe_req_grad = functools.partial(_is_differentiable, need_requires_grad=need_requires_grad)
        is_diff_no_req_grad = functools.partial(_is_differentiable, need_requires_grad=False)

        # Get the differentiable indices for inputs and outputs based on dtype and requires grad information
        diff_in_indices = _tuple_indices_where(inputs_raw, is_diff_maybe_req_grad)
        diff_out_indices_maybe_req_grad = _tuple_indices_where(outputs_raw, is_diff_maybe_req_grad)
        diff_out_indices_no_req_grad = _tuple_indices_where(outputs_raw, is_diff_no_req_grad)

        # Combine the default indices with custom indices if provided
        error_if_not_subset = "Expect all inputs or output indices to correspond to differentiable inputs or outputs"
        if custom_diff_in_indices is not None:
            assert _is_tuple_indices_subset(diff_in_indices, custom_diff_in_indices), error_if_not_subset
            diff_in_indices = custom_diff_in_indices
        if custom_diff_out_indices is not None:
            # TODO: if we pass in tensors that don't require grad, but custom indices are provided, it will fail when
            #       needs_requires_grad=True
            # TODO: why is this the corect behavior? If user marks non-differentiable, that means that we should ignore
            #       it completely? or should we verify that its numerical Jacobian is zero
            assert _is_tuple_indices_subset(diff_out_indices_maybe_req_grad, custom_diff_out_indices), error_if_not_subset
            assert _is_tuple_indices_subset(diff_out_indices_no_req_grad, custom_diff_out_indices), error_if_not_subset
            diff_out_indices_maybe_req_grad = custom_diff_out_indices
            diff_out_indices_no_req_grad = custom_diff_out_indices

        # Apply indices to inputs and outputs
        self.inputs = _tuple_apply_indices(self.inputs_raw, diff_in_indices)
        self.outputs = _tuple_apply_indices(self.outputs_raw, diff_out_indices_maybe_req_grad)

        # Wrap the function based in differentiable inputs and outputs
        def make_wrapped_fn(out_maybe_req_grad):
            def wrapped_fn(*diff_inputs):
                inputs_raw = list(self.inputs_raw)
                for diff_idx, idx in enumerate(diff_in_indices):
                    inputs_raw[idx] = diff_inputs[diff_idx]
                if out_maybe_req_grad:
                    return _tuple_apply_indices(_as_tuple(fn_raw(*inputs_raw)), diff_out_indices_maybe_req_grad)
                else:
                    return _tuple_apply_indices(_as_tuple(fn_raw(*inputs_raw)), diff_out_indices_no_req_grad)
            return wrapped_fn

        self.wrapped_fn_maybe_req_grad = make_wrapped_fn(True)
        if need_requires_grad:
            # We don't need _check_no_differentiable_outputs for forward AD
            self.wrapped_fn_out_no_req_grad = make_wrapped_fn(False)

    def __call__(self, *args, **kwargs):
        return self.wrapped_fn_maybe_req_grad(*args, **kwargs)

@dataclass(frozen=True)
class GradcheckInfo():
    eps: int
    atol: float
    rtol: float
    raise_exception: bool
    check_sparse_nnz: bool
    nondet_tol: bool
    fast_mode: bool
    check_undefined_grad: bool
    check_grad_dtypes: bool
    check_batched_grad: bool
    check_batched_forward_grad: bool
    check_forward_ad: bool
    check_backward_ad: bool
    diff_in_indices: Tuple[int, ...]
    diff_out_indices: Tuple[int, ...]
