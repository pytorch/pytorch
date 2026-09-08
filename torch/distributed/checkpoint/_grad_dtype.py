import torch

from . import state_dict as _state_dict


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


_original_unflatten_optim_state_dict = _state_dict._unflatten_optim_state_dict


def _unflatten_optim_state_dict(optim, state_dict, info):
    params = [param for group in optim.param_groups for param in group["params"]]
    requires_grad = [param.requires_grad for param in params]
    temporarily_removed = {}
    flattened = "state" not in state_dict
    if flattened:
        missing = []
        for index, param in enumerate(params):
            saved_state = optim.state.get(param, {})
            if not saved_state or not requires_grad[index]:
                continue
            fqns = info.fqn_param_mapping.get(param, ())
            for fqn in fqns:
                for name in saved_state:
                    if f"state.{fqn}.{name}" not in state_dict:
                        missing.append((param, fqn))
                        break
        if missing and info.strict:
            raise RuntimeError(
                f"Missing optimizer state for parameter '{missing[0][1]}' in checkpoint. "
                "The parameter requires gradients but has no saved optimizer state. "
                "To load anyway, use StateDictOptions(strict=False)."
            )
        for param, _ in missing:
            if param not in temporarily_removed:
                temporarily_removed[param] = optim.state.pop(param)
    try:
        for param in params:
            param.requires_grad_(True)
        result = _original_unflatten_optim_state_dict(optim, state_dict, info)
        empty = [param for param, value in optim.state.items() if not value]
        for param in empty:
            optim.state.pop(param, None)
        if "state" in result:
            result["state"] = {
                key: value for key, value in result["state"].items() if value
            }
        return result
    finally:
        for param, value in temporarily_removed.items():
            optim.state[param] = value
        for param, value in zip(params, requires_grad):
            param.requires_grad_(value)


_state_dict._unflatten_optim_state_dict = _unflatten_optim_state_dict

__all__ = ["_init_optim_state"]
