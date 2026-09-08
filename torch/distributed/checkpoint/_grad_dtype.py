import torch


def _init_optim_state(optim: torch.optim.Optimizer) -> None:
    for param_group in optim.param_groups:
        if any(param.grad is not None for param in param_group["params"]):
            return

    for param_group in optim.param_groups:
        missing = [
            param
            for param in param_group["params"]
            if param.requires_grad and param not in optim.state
        ]
        if not missing:
            continue

        original_params = param_group["params"]
        original_lr = param_group.get("lr")
        param_group["params"] = missing
        try:
            for param in missing:
                grad_dtype = getattr(param, "grad_dtype", None) or param.dtype
                param.grad = torch.zeros_like(param, dtype=grad_dtype)
            if "lr" in param_group:
                param_group["lr"] = (
                    torch.tensor(0.0, device=missing[0].device)
                    if isinstance(original_lr, torch.Tensor)
                    else 0.0
                )
            optim.step(closure=None)
        finally:
            param_group["params"] = original_params
            if "lr" in param_group:
                param_group["lr"] = original_lr
            for param in missing:
                param.grad = None


__all__ = ["_init_optim_state"]
