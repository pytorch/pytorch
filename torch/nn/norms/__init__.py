"""Registry and activation of custom norm kernel implementations.

Usage::

    import torch.nn.norms

    torch.nn.norms.list_norm_impls()
    # ['cutedsl_rmsnorm']

    torch.nn.norms.activate_norm_impl("cutedsl_rmsnorm")
    # F.rms_norm / nn.RMSNorm now uses CuteDSL kernels on CUDA

    torch.nn.norms.restore_norm_impl()  # back to default
"""

from . import _registry


register_norm_impl = _registry.register_norm_impl
activate_norm_impl = _registry.activate_norm_impl
list_norm_impls = _registry.list_norm_impls
current_norm_impl = _registry.current_norm_impl
restore_norm_impl = _registry.restore_norm_impl

register_norm_impl.__module__ = __name__
activate_norm_impl.__module__ = __name__
list_norm_impls.__module__ = __name__
current_norm_impl.__module__ = __name__
restore_norm_impl.__module__ = __name__

# Import built-in implementations to trigger self-registration.
# Guarded because the CuteDSL kernels depend on cuda/cutlass packages.
try:
    from . import _cutedsl_norms  # noqa: F401
except ImportError:
    pass
