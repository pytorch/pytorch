import torch
from torch import Tensor


_NAMESPACE = "libtorch_agn_frozen"


def identity(t) -> Tensor:
    """Returns the input tensor."""
    return torch.ops.libtorch_agn_frozen.identity.default(t)


def my_abs(t) -> Tensor:
    """Returns abs on the input tensor."""
    return torch.ops.libtorch_agn_frozen.my_abs.default(t)


def my_is_cpu(t) -> bool:
    """Returns whether the input tensor is on CPU."""
    return torch.ops.libtorch_agn_frozen.my_is_cpu.default(t)


# 2.13-only ops are registered only when the extension was *built* against
# torch >= 2.13. Probe the loaded library (not runtime torch.__version__),
# since Case 2 builds on 2.9 and runs on a newer runtime.
_ops_ns = getattr(torch.ops, _NAMESPACE, None)
if _ops_ns is not None and hasattr(_ops_ns, "my_exception_what"):

    def my_exception_what() -> str:
        """Stable exception what() shim (present when built with ABI >= 2.13)."""
        return torch.ops.libtorch_agn_frozen.my_exception_what.default()

    def my_exception_get_what_without_backtrace() -> str:
        """Stable exception what-without-backtrace shim (ABI >= 2.13 builds)."""
        return torch.ops.libtorch_agn_frozen.my_exception_get_what_without_backtrace.default()
