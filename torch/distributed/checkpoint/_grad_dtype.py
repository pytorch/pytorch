import torch


def _init_optim_state(optim: torch.optim.Optimizer) -> None:
    if optim.state:
        return

    for param_group in optim.param_groups:
        for param in param_group["params"]:
            if param.grad is not None:
                return

    for param_group in optim.param_groups:
        for param in param_group["params"]:
            if param.requires_grad:
                grad_dtype = getattr(param, "grad_dtype", None)
                if grad_dtype is None:
                    grad_dtype = param.dtype
                param.grad = torch.zeros_like(param, dtype=grad_dtype)

    lrs = []
    for param_group in optim.param_groups:
        if "lr" in param_group:
            lrs.append(param_group["lr"])
            param_group["lr"] = (
                torch.tensor(0.0)
                if isinstance(param_group["lr"], torch.Tensor)
                else 0.0
            )
    optim.step(closure=None)

    for param_group in optim.param_groups:
        if "lr" in param_group:
            param_group["lr"] = lrs.pop(0)
    optim.zero_grad(set_to_none=True)


__all__ = ["_init_optim_state"]
