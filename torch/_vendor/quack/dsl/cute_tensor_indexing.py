# Copyright (c) 2026, Tri Dao.
"""Small compatibility layer for more Pythonic CuTe tensor indexing.

CuTe uses ``None`` as its underscore/full-mode slice marker, e.g. ``A[i, None]``.
This module teaches CuTe tensors the equivalent Python spelling ``:`` and expands
``...`` to the right number of full-mode slices.

Importing this module intentionally mutates CuTe's tensor classes process-wide.
The original methods are retained as ``_quack_original_getitem`` and
``_quack_original_setitem`` for debugging or manual rollback.
"""

from __future__ import annotations

from typing import Any

from cutlass.cutlass_dsl import dsl_user_op
import cutlass.cute.tensor as _cute_tensor


_ORIGINAL_GETITEM_ATTR = "_quack_original_getitem"
_ORIGINAL_SETITEM_ATTR = "_quack_original_setitem"
_PATCHED_ATTR = "_quack_extended_indexing"
_PATCHED_SETITEM_ATTR = f"{_PATCHED_ATTR}_setitem"


def _is_full_slice(idx: Any) -> bool:
    return isinstance(idx, slice) and idx.start is None and idx.stop is None and idx.step is None


def _index_uses_ellipsis(idx: Any) -> bool:
    if idx is Ellipsis:
        return True
    if isinstance(idx, tuple):
        return any(_index_uses_ellipsis(item) for item in idx)
    return False


def _shape_rank(shape: Any, idx: Any = None) -> int:
    if shape is None:
        suffix = f" in {idx!r}" if idx is not None else ""
        raise ValueError(f"tensor shape is required to expand ellipsis{suffix}")
    return _cute_tensor.rank(shape)


def _shape_mode(shape: Any, mode: int) -> Any:
    if isinstance(shape, tuple) and mode < len(shape):
        return shape[mode]
    return None


def _canonicalize_cute_tensor_index(idx: Any, tensor_shape: Any = None) -> Any:
    """Convert Python indexing sugar to CuTe's coordinate convention.

    ``:`` becomes ``None`` (CuTe's full-mode/underscore marker) and ``...`` expands
    within the current hierarchy level using ``tensor_shape``. Other slices like
    ``1:4`` are intentionally rejected because CuTe tensor slicing only supports
    keeping an entire mode or selecting a single coordinate.
    """
    if idx is Ellipsis:
        return (None,) * _shape_rank(tensor_shape, idx)
    if _is_full_slice(idx):
        return None
    if isinstance(idx, slice):
        raise ValueError(f"CuTe Tensor indexing only supports full slices ':', got {idx!r}")
    if not isinstance(idx, tuple):
        return idx

    ellipsis_count = sum(item is Ellipsis for item in idx)
    if ellipsis_count > 1:
        raise ValueError("CuTe Tensor indexing supports at most one ellipsis per tuple level")

    explicit_modes = len(idx) - ellipsis_count
    fill_modes = 0
    if ellipsis_count:
        tensor_rank = _shape_rank(tensor_shape, idx)
        fill_modes = tensor_rank - explicit_modes
        if fill_modes < 0:
            raise ValueError(
                f"ellipsis cannot expand index {idx!r} for rank-{tensor_rank} CuTe Tensor mode"
            )

    result: list[Any] = []
    mode = 0
    for item in idx:
        if item is Ellipsis:
            result.extend([None] * fill_modes)
            mode += fill_modes
        else:
            result.append(_canonicalize_cute_tensor_index(item, _shape_mode(tensor_shape, mode)))
            mode += 1
    return tuple(result)


def _make_getitem(original_getitem: Any) -> Any:
    @dsl_user_op
    def _getitem(self: Any, idx: Any, *, loc: Any = None, ip: Any = None) -> Any:
        tensor_shape = self.shape if _index_uses_ellipsis(idx) else None
        idx = _canonicalize_cute_tensor_index(idx, tensor_shape)
        return original_getitem(self, idx, loc=loc, ip=ip)

    return _getitem


def _make_setitem(original_setitem: Any) -> Any:
    @dsl_user_op
    def _setitem(self: Any, idx: Any, data: Any, *, loc: Any = None, ip: Any = None) -> Any:
        tensor_shape = self.shape if _index_uses_ellipsis(idx) else None
        idx = _canonicalize_cute_tensor_index(idx, tensor_shape)
        return original_setitem(self, idx, data, loc=loc, ip=ip)

    return _setitem


def patch_cute_tensor_indexing() -> None:
    """Monkey patch CuTe Tensor indexing with ``:``, ``...`` sugar.

    The patch is idempotent and keeps the original CuTe implementation for all
    canonical coordinates, so existing ``A[i, j, None]`` code continues to behave
    exactly as before. It is a process-wide mutation of CuTe's tensor classes.
    """
    for cls in (_cute_tensor._Tensor, _cute_tensor.TensorSSA):
        if _PATCHED_ATTR not in cls.__dict__:
            setattr(cls, _ORIGINAL_GETITEM_ATTR, cls.__getitem__)
            cls.__getitem__ = _make_getitem(cls.__getitem__)  # type: ignore[method-assign]
            setattr(cls, _PATCHED_ATTR, True)

    # TensorSSA has no upstream __setitem__, so only _Tensor needs the store path patched.
    tensor_cls = _cute_tensor._Tensor
    if _PATCHED_SETITEM_ATTR not in tensor_cls.__dict__:
        setattr(tensor_cls, _ORIGINAL_SETITEM_ATTR, tensor_cls.__setitem__)
        tensor_cls.__setitem__ = _make_setitem(tensor_cls.__setitem__)  # type: ignore[method-assign]
        setattr(tensor_cls, _PATCHED_SETITEM_ATTR, True)


patch_cute_tensor_indexing()


__all__ = ["patch_cute_tensor_indexing"]
