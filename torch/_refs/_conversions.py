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
    "cdouble",
    "cfloat",
    "chalf",
    "char",
    "double",
    "float",
    "half",
    "int",
    "long",
    "short",
]


def _make_conversion_method(name: str, dtype: torch.dtype):
    def fn(
        self: TensorLikeType, memory_format: torch.memory_format = torch.preserve_format
    ) -> TensorLikeType:
        return self.to(dtype, memory_format=memory_format)  # type: ignore[call-overload]

    fn.__name__ = name
    return fn


bfloat16 = _make_conversion_method("bfloat16", torch.bfloat16)

bool = _make_conversion_method("bool", torch.bool)

byte = _make_conversion_method("byte", torch.uint8)

cdouble = _make_conversion_method("cdouble", torch.cdouble)

cfloat = _make_conversion_method("cfloat", torch.cfloat)

chalf = _make_conversion_method("chalf", torch.complex32)

char = _make_conversion_method("char", torch.int8)

double = _make_conversion_method("double", torch.double)

float = _make_conversion_method("float", torch.float)

half = _make_conversion_method("half", torch.half)

int = _make_conversion_method("int", torch.int)

long = _make_conversion_method("long", torch.long)

short = _make_conversion_method("short", torch.short)
