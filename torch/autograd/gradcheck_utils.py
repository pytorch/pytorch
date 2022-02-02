from dataclasses import dataclass
from typing import Callable
from torch.overrides import is_tensor_like

def _as_tuple(x):
    if isinstance(x, tuple):
        return x
    elif isinstance(x, list):
        return tuple(x)
    else:
        return x,

def _is_float_or_complex_tensor(obj):
    return is_tensor_like(obj) and (obj.is_floating_point() or obj.is_complex())

class GradcheckError(RuntimeError):
    # Custom error so that user errors are not caught in the gradcheck's try-catch
    pass

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
