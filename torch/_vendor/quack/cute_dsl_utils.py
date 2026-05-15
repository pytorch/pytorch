# Copyright (c) 2025, Tri Dao.

from typing import Tuple, get_origin
from functools import lru_cache
from dataclasses import dataclass, fields

import os
import re

import torch

try:
    from triton.tools.disasm import extract
except ImportError:
    extract = None

import cutlass
import cutlass.cute as cute
from cutlass import Int32, Int64, Float16, BFloat16, Float32
from cutlass.base_dsl.tvm_ffi_builder import spec
from cutlass.cutlass_dsl import NumericMeta


StaticTypes = (cutlass.Constexpr, NumericMeta, int, bool, str, float, type(None))


load_cubin_module_data_og = cutlass.base_dsl.runtime.cuda.load_cubin_module_data
cute_compile_og = cute.compile


# Patch TVM-FFI converter to handle Constexpr type annotations as compile-time constants.
# Fields annotated with cutlass.Constexpr[T] are emitted as ConstNone (not runtime args).
# At call time, pass None for these fields; the compile-time value is baked in.
import cutlass.cute._tvm_ffi_args_spec_converter as _converter_module  # noqa

_original_convert_single_arg = _converter_module._convert_single_arg


def _patched_convert_single_arg(arg, arg_name, arg_type, ctx):
    if arg_type is not None and get_origin(arg_type) is cutlass.Constexpr:
        return spec.ConstNone(arg_name)
    # If arg is a NamedTuple but arg_type doesn't have _fields (e.g. annotated as tuple),
    # redirect so the converter uses the NamedTuple's own type hints.
    if (
        isinstance(arg, tuple)
        and hasattr(type(arg), "_fields")
        and (arg_type is None or not hasattr(arg_type, "_fields"))
    ):
        return _original_convert_single_arg(arg, arg_name, type(arg), ctx)
    return _original_convert_single_arg(arg, arg_name, arg_type, ctx)


_converter_module._convert_single_arg = _patched_convert_single_arg


torch2cute_dtype_map = {
    torch.float16: Float16,
    torch.bfloat16: BFloat16,
    torch.float32: Float32,
    torch.int32: Int32,
    torch.int64: Int64,
}


@lru_cache
def get_device_multiprocessor_count(device_id: int = 0) -> int:
    return cutlass.utils.HardwareInfo(device_id).get_device_multiprocessor_count()


@lru_cache
def get_max_active_clusters(
    cluster_size: int,
    device_capacity: Tuple[int, int] | None = None,
    device_id: int = 0,
) -> int:
    if device_capacity is None:
        device_capacity = get_device_capacity()
    if device_capacity[0] < 9:
        if cluster_size != 1:
            raise ValueError("SM8x kernels do not support CTA clusters; cluster_size must be 1")
        return get_device_multiprocessor_count(device_id)
    return cutlass.utils.HardwareInfo(device_id).get_max_active_clusters(cluster_size=cluster_size)


def _parse_arch_str(arch_str: str) -> Tuple[int, int]:
    """Parse arch string (e.g. 'sm_90', 'sm90', '90', 'sm_100a') to (major, minor) tuple."""
    match = re.match(r"^(?:sm_?)?(\d+)(\d)([af]?)$", arch_str.strip(), re.IGNORECASE)
    if not match:
        raise ValueError(f"Invalid QUACK_ARCH format: {arch_str!r} (expected e.g. '90', 'sm_90')")
    major, minor, _ = match.groups()
    return int(major), int(minor)


@lru_cache
def _get_device_capacity_cached(device: torch.device = None) -> Tuple[int, int]:
    """Return (major, minor) device capability.

    Override with QUACK_ARCH (e.g. 'sm_90' or '90') for CPU-only compilation
    without a GPU present.
    """
    arch_override = os.environ.get("QUACK_ARCH")
    if arch_override is not None:
        return _parse_arch_str(arch_override)
    return torch.cuda.get_device_capability(device)


def get_device_capacity(
    device: torch.device | torch.Tensor | None = None,
) -> Tuple[int, int]:
    """Return (major, minor) device capability.

    Override with QUACK_ARCH (e.g. 'sm_90' or '90') for CPU-only compilation
    without a GPU present.

    Accepts either a ``torch.device`` or a tensor and canonicalizes to the
    underlying device before consulting the cached helper. This avoids leaking
    tensors through the LRU cache key.
    """
    if isinstance(device, torch.Tensor):
        device = device.device
    return _get_device_capacity_cached(device)


def _partition_fields(obj):
    """Split dataclass fields into (constexpr_dict, non_constexpr_dict) by type."""
    all_fields = {field.name: getattr(obj, field.name) for field in fields(obj)}
    constexpr = {n: f for n, f in all_fields.items() if isinstance(f, StaticTypes)}
    non_constexpr = {n: f for n, f in all_fields.items() if not isinstance(f, StaticTypes)}
    return constexpr, non_constexpr


def _new_from_mlir_values(self, values):
    constexpr_fields, non_constexpr_fields = _partition_fields(self)
    for (name, field), n_items in zip(non_constexpr_fields.items(), self._values_pos):
        non_constexpr_fields[name] = cutlass.new_from_mlir_values(field, values[:n_items])
        values = values[n_items:]
    return self.__class__(**non_constexpr_fields, **constexpr_fields)


def _namedtuple_new_from_mlir_values(self, values):
    """Generic __new_from_mlir_values__ for NamedTuples.

    Applied to NamedTuple classes via the ``@mlir_namedtuple`` decorator.

    Fields that are None or Constexpr (StaticTypes) are preserved from ``self`` (the compile-time
    template). Only non-static fields consume MLIR values. Multi-value fields (e.g. cute.Tensor)
    consume the correct number of values via ``cutlass.new_from_mlir_values``.

    Constexpr fields (annotated ``cutlass.Constexpr[T]``) are baked into the compiled kernel via
    a converter patch (see above). At call time, pass None for these fields.
    """
    from cutlass.base_dsl.typing import get_mlir_types

    values = list(values)
    new_fields = []
    for field_val in self:
        if field_val is None or isinstance(field_val, StaticTypes):
            new_fields.append(field_val)
        else:
            n_items = len(get_mlir_types(field_val))
            new_fields.append(cutlass.new_from_mlir_values(field_val, values[:n_items]))
            values = values[n_items:]
    return self.__class__(*new_fields)


def mlir_namedtuple(cls):
    """Decorator that adds MLIR value reconstruction to a NamedTuple class.

    Usage::

        @mlir_namedtuple
        class MyArgs(NamedTuple):
            tensor_arg: cute.Tensor
            const_arg: cutlass.Constexpr[int] = 0
    """
    cls.__new_from_mlir_values__ = _namedtuple_new_from_mlir_values
    return cls


@dataclass
class ParamsBase:
    def __extract_mlir_values__(self):
        _, non_constexpr_fields = _partition_fields(self)
        values, self._values_pos = [], []
        for obj in non_constexpr_fields.values():
            obj_values = cutlass.extract_mlir_values(obj)
            values += obj_values
            self._values_pos.append(len(obj_values))
        return values

    __new_from_mlir_values__ = _new_from_mlir_values
