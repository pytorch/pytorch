import torch

from torch._prims_common import TensorLikeType

# Data conversion references.
#
# Note: this module breaks the usual _refs to torch naming scheme where
# _refs.foo.bar is a ref for torch.foo.bar.  The following definitions are not
# part of _refs/__init__.py to avoid name clashes with Python builtin types
# (like int).

__all__ = [
    "bfloat16",
    "bool",
    "byte",
    "char",
    "double",
    "float",
    "half",
    "int",
    "long",
    "short",
    "chalf",
    "cfloat",
    "cdouble",
]


def bool(
    self: TensorLikeType, memory_format: torch.memory_format = torch.preserve_format
) -> TensorLikeType:
    return self.to(torch.bool, memory_format=memory_format)  # type: ignore[call-overload]


def bfloat16(
    self: TensorLikeType, memory_format: torch.memory_format = torch.preserve_format
) -> TensorLikeType:
    return self.to(torch.bfloat16, memory_format=memory_format)  # type: ignore[call-overload]


def byte(
    self: TensorLikeType, memory_format: torch.memory_format = torch.preserve_format
) -> TensorLikeType:
    return self.to(torch.uint8, memory_format=memory_format)  # type: ignore[call-overload]


def char(
    self: TensorLikeType, memory_format: torch.memory_format = torch.preserve_format
) -> TensorLikeType:
    return self.to(torch.int8, memory_format=memory_format)  # type: ignore[call-overload]


def double(
    self: TensorLikeType, memory_format: torch.memory_format = torch.preserve_format
) -> TensorLikeType:
    return self.to(torch.double, memory_format=memory_format)  # type: ignore[call-overload]


def float(
    self: TensorLikeType, memory_format: torch.memory_format = torch.preserve_format
) -> TensorLikeType:
    return self.to(torch.float, memory_format=memory_format)  # type: ignore[call-overload]


def half(
    self: TensorLikeType, memory_format: torch.memory_format = torch.preserve_format
) -> TensorLikeType:
    return self.to(torch.half, memory_format=memory_format)  # type: ignore[call-overload]


def int(
    self: TensorLikeType, memory_format: torch.memory_format = torch.preserve_format
) -> TensorLikeType:
    return self.to(torch.int, memory_format=memory_format)  # type: ignore[call-overload]


def long(
    self: TensorLikeType, memory_format: torch.memory_format = torch.preserve_format
) -> TensorLikeType:
    return self.to(torch.long, memory_format=memory_format)  # type: ignore[call-overload]


def short(
    self: TensorLikeType, memory_format: torch.memory_format = torch.preserve_format
) -> TensorLikeType:
    return self.to(torch.short, memory_format=memory_format)  # type: ignore[call-overload]


def chalf(
    self: TensorLikeType, memory_format: torch.memory_format = torch.preserve_format
) -> TensorLikeType:
    return self.to(torch.complex32, memory_format=memory_format)  # type: ignore[call-overload]


def cfloat(
    self: TensorLikeType, memory_format: torch.memory_format = torch.preserve_format
) -> TensorLikeType:
    return self.to(torch.cfloat, memory_format=memory_format)  # type: ignore[call-overload]


def cdouble(
    self: TensorLikeType, memory_format: torch.memory_format = torch.preserve_format
) -> TensorLikeType:
    return self.to(torch.cdouble, memory_format=memory_format)  # type: ignore[call-overload]
