import torch
from torch.overrides import TorchFunctionMode


class GradStateOpsFailSafeguard(TorchFunctionMode):
    """
    Detect grad state ops during exporting the graph and fail the process by
    raising an error, to avoid unexpected behavior. Those grad mode ops could be:
    `torch.no_grad`
    `torch.enable_grad`
    `torch.set_grad_enabled`
    """

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        unsupported_grad_mode_ops = [
            torch._C._set_grad_enabled,
        ]
        # To allow the grad ops out of tracing the user-defined func/module, e.g.
        # internal usage before or after tracing, we need to only enable it when
        # the dispatch mode is proxy.
        if func in unsupported_grad_mode_ops and torch._C._get_dispatch_mode(
            torch._C._TorchDispatchModeKey.PROXY
        ):
            raise RuntimeError(
                f"Encountered autograd state manager op {func} while exporting. "
                "This is unsafe because we don't capture this op in torch.export"
                "today, hence we can't reflect the user intention soundly."
            )
        return func(*args, **kwargs)
