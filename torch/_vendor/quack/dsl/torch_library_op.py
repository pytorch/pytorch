# Copyright (c) 2025, Wentao Guo, Ted Zadouri, Tri Dao.
"""``cute_op``: ``torch.library.custom_op`` for CuTe DSL kernels.

Same trick as ``torch.library.triton_op`` (register the impl as the fake/meta
kernel too), specialized for our setup: the fake is a pure no-op. Our ops
only mutate their inputs, so Dynamo / AOT autograd need no shape effect from
the fake, and kernel compilation is owned entirely by ``jit_cache`` (plus
the async compile pool) at real execution time.

This removes the need for hand-written ``_*_fake`` twins on each op.

Note: we deliberately do NOT gate on ``torch.compiler.is_compiling()`` —
that flag's underlying ``_is_compiling_flag`` is only set during
``torch.export``, never during ``torch.compile``. Dynamo's
``_get_fake_value_impl`` would otherwise run the body and surface
any ``_compile_*`` ``ValueError`` as a ``TorchRuntimeError`` graph break.
"""

from __future__ import annotations

import functools
from typing import Any, Callable, Iterable, Optional, Union

import torch

__all__ = ["cute_op"]


def cute_op(
    name: str,
    *,
    mutates_args: Union[str, Iterable[str]],
    schema: Optional[str] = None,
    device_types: Optional[Union[str, Iterable[str]]] = None,
) -> Callable:
    """Like ``torch.library.triton_op``, but for CuTe DSL kernels.

    Args:
        name: ``"namespace::op_name"``.
        mutates_args: Names of mutated tensor args.
        schema: Optional explicit schema. Required when mutating an
            ``Optional[Tensor]`` arg (PyTorch can't infer those).
        device_types: Optional device-type restriction.
    """

    def dec(fn: Callable) -> Any:
        kwargs: dict[str, Any] = {"mutates_args": mutates_args}
        if schema is not None:
            kwargs["schema"] = schema
        if device_types is not None:
            kwargs["device_types"] = device_types
        op = torch.library.custom_op(name, fn, **kwargs)

        @op.register_fake
        def _fake(*args, **kw):
            # Pure no-op: our ops only mutate their input tensors, so under
            # torch.compile / AOT autograd tracing there is no fake output to
            # produce, and running the body would pay compile latency at
            # dynamo trace time (or crash for shape/dtype combos the kernel
            # intentionally rejects). Kernel compilation is handled by
            # jit_cache + the async compile pool at real execution time.
            return

        return _EagerBypassOp(op, fn)

    return dec


class _EagerBypassOp:
    """Callable returned by :func:`cute_op`.

    Eager fast path: the torch.library dispatch + functionalization boundary
    costs ~60us/call (measured on rmsnorm_fwd, H100) — for these small
    memory-bound kernels that is often larger than the kernel itself (e.g.
    rmsnorm 1024x4096: ~10us kernel vs ~125us through the boundary). In eager we
    call the raw body directly to skip it; under ``torch.compile`` we route to
    the real op so Dynamo captures it as a graph node (with the no-op fake).
    ``torch.compiler.is_compiling()`` is constant-folded to True by Dynamo at
    trace time (torch/_dynamo/variables/torch.py), so the gate is correct under
    both compile and export. Same trick as ``_launch()`` in gemm_interface.py.

    Attribute access falls through to the underlying ``CustomOpDef`` so
    post-registration hooks keep working (e.g. ``register_effect``,
    ``register_autograd``, ``_opoverload``).
    """

    def __init__(self, op, fn):
        self._custom_op = op
        self._init_fn = fn
        functools.update_wrapper(self, fn)

    def __call__(self, *args, **kwargs):
        if torch.compiler.is_compiling():
            return self._custom_op(*args, **kwargs)
        return self._init_fn(*args, **kwargs)

    def __getattr__(self, name):
        # Only reached for attrs not set on the instance -> forward to the op.
        return getattr(self.__dict__["_custom_op"], name)
