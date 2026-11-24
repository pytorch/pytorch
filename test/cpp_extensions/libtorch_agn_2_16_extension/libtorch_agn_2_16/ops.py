import torch


# =============================================================================
# Proxy for inherited ops (from libtorch_agn_2_9, 2_10, 2_11, 2_12, 2_13, 2_14,
# and 2_15 csrc/)
#
# Ops compiled from previous versions' csrc directories are accessible via
# the module-level __getattr__. For example:
#     libtorch_agn_2_16.ops.sgd_out_of_place(...)  # from 2.9
#     libtorch_agn_2_16.ops.my_sum(...)            # from 2.10
#
# Ops defined in this package's csrc/ (e.g. my_add) are
# registered on ``torch.ops.libtorch_agn_2_16`` and wrapped explicitly below.
# =============================================================================

_NAMESPACE = "libtorch_agn_2_16"


def my_add(self, other, alpha=1.0) -> torch.Tensor:
    """Stable ``add.Tensor`` (2.16+)."""
    return torch.ops.libtorch_agn_2_16.my_add.default(self, other, alpha)


def my_is_conj(t: torch.Tensor) -> bool:
    return torch.ops.libtorch_agn_2_16.my_is_conj.default(t)


def my_is_neg(t: torch.Tensor) -> bool:
    return torch.ops.libtorch_agn_2_16.my_is_neg.default(t)


def my_is_inference(t: torch.Tensor) -> bool:
    return torch.ops.libtorch_agn_2_16.my_is_inference.default(t)


def my_inference_mode_guard(enabled: bool) -> bool:
    return torch.ops.libtorch_agn_2_16.my_inference_mode_guard.default(enabled)


def my_default_inference_mode_guard() -> bool:
    return torch.ops.libtorch_agn_2_16.my_default_inference_mode_guard.default()


def my_is_inference_mode_enabled() -> bool:
    return torch.ops.libtorch_agn_2_16.my_is_inference_mode_enabled.default()


def my_is_warn_always_enabled() -> bool:
    """Stable ``torch_is_warn_always_enabled`` (2.16+)."""
    return torch.ops.libtorch_agn_2_16.my_is_warn_always_enabled.default()


def my_cpublas_brgemm_dispatch(a, b, c, add_c=False, is_vnni=False) -> torch.Tensor:
    """Write a supported brgemm dtype combination into c (2.16+)."""
    return torch.ops.libtorch_agn_2_16.my_cpublas_brgemm_dispatch.default(
        a, b, c, add_c, is_vnni
    )


def my_cpublas_gemm_no_downcast_stub_cpu(
    a, b, c, alpha=1.0, beta=0.0, transpose_b=False
) -> torch.Tensor:
    """Write float32 CPU GEMM into c through CPU capability dispatch (2.16+)."""
    return torch.ops.libtorch_agn_2_16.my_cpublas_gemm_no_downcast_stub_cpu.default(
        a, b, c, alpha, beta, transpose_b
    )


def my_cpublas_could_pack(tensor) -> bool:
    """Check whether cpublas can pack the input dtype (2.16+)."""
    return torch.ops.libtorch_agn_2_16.my_cpublas_could_pack.default(tensor)


def my_cpublas_brgemm_release() -> None:
    """Release cpublas brgemm resources (2.16+)."""
    torch.ops.libtorch_agn_2_16.my_cpublas_brgemm_release.default()


def test_stable_torch_warn_once(value) -> None:
    """
    Tests STABLE_TORCH_WARN_ONCE with one call site for negative values and one
    for non-negative values.

    Args:
        value: int - selects the call site
    """
    torch.ops.libtorch_agn_2_16.test_stable_torch_warn_once.default(value)


def my_is_privateuseone(t) -> bool:
    """
    Returns is_privateuseone on the input tensor.

    Args:
        t: any Tensor

    Returns:
        a bool
    """
    return torch.ops.libtorch_agn_2_16.my_is_privateuseone.default(t)


def test_device_is_privateuseone(device) -> bool:
    """
    Tests Device is_privateuseone() method.

    Args:
        device: Device - device to check

    Returns: bool - True if device is privateuseone
    """
    return torch.ops.libtorch_agn_2_16.test_device_is_privateuseone.default(device)


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
