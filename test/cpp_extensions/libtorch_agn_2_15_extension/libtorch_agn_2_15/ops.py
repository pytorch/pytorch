import torch


# =============================================================================
# Proxy for inherited ops (from libtorch_agn_2_9, 2_10, 2_11, 2_12, 2_13, and
# 2_14 csrc/)
#
# Ops compiled from previous versions' csrc directories are accessible via
# the module-level __getattr__. For example:
#     libtorch_agn_2_15.ops.sgd_out_of_place(...)  # from 2.9
#     libtorch_agn_2_15.ops.my_sum(...)            # from 2.10
#
# Ops defined in this package's csrc/ (e.g. my_add) are registered on
# ``torch.ops.libtorch_agn_2_15`` and wrapped explicitly below.
# =============================================================================

_NAMESPACE = "libtorch_agn_2_15"


def my_add(self, other, alpha=1.0) -> torch.Tensor:
    """Stable ``add.Tensor`` (2.15+)."""
    return torch.ops.libtorch_agn_2_15.my_add.default(self, other, alpha)


def __getattr__(name):
    """Proxy for inherited ops from previous versions."""
    if name.startswith("_"):
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    ops_namespace = getattr(torch.ops, _NAMESPACE)
    op = getattr(ops_namespace, name, None)
    if op is None:
        raise AttributeError(f"No op named '{name}' in {_NAMESPACE}")
    return op.default


def __dir__():
    """List all available ops (native + inherited)."""
    native = [
        name
        for name in globals()
        if not name.startswith("_") and callable(globals().get(name))
    ]
    ops_namespace = getattr(torch.ops, _NAMESPACE)
    inherited = [n for n in dir(ops_namespace) if not n.startswith("_")]
    return sorted(set(native + inherited))
