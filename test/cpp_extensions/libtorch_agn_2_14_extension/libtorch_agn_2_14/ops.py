import torch


# =============================================================================
# Proxy for inherited ops (from libtorch_agn_2_9, 2_10, 2_11, 2_12, and 2_13
# csrc/)
#
# Ops compiled from previous versions' csrc directories are accessible via
# the module-level __getattr__. For example:
#     libtorch_agn_2_14.ops.sgd_out_of_place(...)  # from 2.9
#     libtorch_agn_2_14.ops.my_sum(...)            # from 2.10
#
# Ops defined in this package's csrc/ (e.g. my_has_storage) are registered on
# ``torch.ops.libtorch_agn_2_14`` and wrapped explicitly below.
# =============================================================================

_NAMESPACE = "libtorch_agn_2_14"


def my_has_storage(t) -> bool:
    """Stable ``Tensor::has_storage()`` (2.14+)."""
    return torch.ops.libtorch_agn_2_14.my_has_storage.default(t)


def my_index_select(self, dim, index):
    """Stable ``index_select`` (build time 2.14+, runtime 2.10+)."""
    return torch.ops.libtorch_agn_2_14.my_index_select.default(self, dim, index)


def my_floor_divide(self, other):
    """Stable ``floor_divide`` (build time 2.14+, runtime 2.10+)."""
    return torch.ops.libtorch_agn_2_14.my_floor_divide.default(self, other)


def my_is_pinned(self) -> bool:
    """Stable ``is_pinned`` (build time 2.14+, runtime 2.10+)."""
    return torch.ops.libtorch_agn_2_14.my_is_pinned.default(self)


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
