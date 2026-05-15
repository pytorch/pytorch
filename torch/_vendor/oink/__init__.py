"""Vendored subset of the kernelagent_oink library
(https://github.com/meta-pytorch/KernelAgent).

Upstream SHA: 54b1331d2f5fc7e615c39e3057b836ecc5e2c10a (kernelagent-oink 0.1.0)

Only the modules required by torch._native.ops.norm.oink_rmsnorm_impl are
vendored. Imports are rewritten to be package-relative so this copy is
independent of any ``kernelagent_oink`` top-level package that may be
installed via pip.
"""
__version__ = "0.1.0"

from .rmsnorm import rmsnorm_forward, rmsnorm_backward  # noqa: E402


__all__ = [
    "rmsnorm_forward",
    "rmsnorm_backward",
]
