import torch

from . import state_dict as _state_dict


_original = _state_dict._init_optim_state


def _init_optim_state(optim: torch.optim.Optimizer) -> None:
    for param_group in optim.param_groups:
        for param in param_group["params"]:
            if param.requires_grad and param.grad is None:
                grad_dtype = getattr(param, "grad_dtype", None)
                if grad_dtype is None:
                    grad_dtype = param.dtype
                param.grad = torch.zeros_like(param, dtype=grad_dtype)

    original_grads = []
    for param_group in optim.param_groups:
        for param in param_group["params"]:
            original_grads.append((param, param.grad))

    try:
        return _original(optim)
    finally:
        for param, grad in original_grads:
            if grad is None:
                param.grad = None


__all__ = ["_init_optim_state"]
