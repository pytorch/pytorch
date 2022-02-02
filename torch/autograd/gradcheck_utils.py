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

def _is_differentiable(obj, need_requires_grad):
    return _is_float_or_complex_tensor(obj) and (obj.requires_grad or not need_requires_grad)

def _is_float_or_complex_tensor(obj):
    return is_tensor_like(obj) and (obj.is_floating_point() or obj.is_complex())

def _tuple_indices_where(tup, cond_fn: Callable[..., bool]):
    assert isinstance(tup, Tuple)
    return tuple(i for (i, obj) in enumerate(tup) if cond_fn(obj))

def _tuple_apply_indices(tup, indices: Tuple[int, ...]):
    assert isinstance(tup, Tuple)
    return tuple(obj for (i, obj) in enumerate(tup) if i in indices)

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

    def __init__(self, fn_raw, inputs_raw, outputs_raw, diff_in_indices, diff_out_indices, need_requires_grad):
        self.fn_raw = fn_raw
        self.inputs_raw = inputs_raw
        self.outputs_raw = outputs_raw

        # "maybe_req_grad" means that we require outputs to have requires_grad=True when used for backward AD but not forward AD
        is_diff_maybe_req_grad = functools.partial(_is_differentiable, need_requires_grad=need_requires_grad)
        is_diff_no_req_grad = functools.partial(_is_differentiable, need_requires_grad=False)

        diff_in_indices = _tuple_indices_where(inputs_raw, is_diff_maybe_req_grad) if diff_in_indices is None else diff_in_indices
        diff_out_indices_maybe_req_grad = _tuple_indices_where(outputs_raw, is_diff_maybe_req_grad) if diff_out_indices is None else diff_out_indices
        # What should be the behavior here? When user marks an output as non-differentiable with diff_out_indices...
        # Currently, we just ignore it. Alternatively, we can check if its numerical Jacobian is zero
        diff_out_indices_no_req_grad = _tuple_indices_where(outputs_raw, is_diff_no_req_grad) if diff_out_indices is None else diff_out_indices

        self.inputs = _tuple_apply_indices(self.inputs_raw, diff_in_indices)
        self.outputs = _tuple_apply_indices(self.outputs_raw, diff_out_indices_maybe_req_grad)

        # NB: we have two versions: 'maybe_req_grad' and 'no_req_grad'
        #    1)
        #    2) no_req_grad never requires outputs to have requires_grad=True
        #       Currently, this is only useful for check_no_differentiable_outputs
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