import torch
import torch._prims_common as utils

# Utilities should come BEFORE this import
from torch._decomp import register_decomposition

from torch._prims_common import check, TensorLikeType
from torch._prims_common.wrappers import out_wrapper
from torch._refs import _broadcast_shapes

# Data conversion references.
#
# Note: this module breaks the usual _refs to torch naming scheme where
# _refs.foo.bar is a ref for torch.foo.bar.  The following definitions are not
# part of _refs/__init__.py to avoid name clashes with Python builtin types
# (like int).

__all__ = [
    # dtypes
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
    # misc
    "complex",
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


@register_decomposition(torch.ops.aten.complex)
@out_wrapper()
def complex(real: TensorLikeType, imag: TensorLikeType) -> TensorLikeType:
    def _allowed_dtypes():
        return {torch.float32, torch.float64, torch.float16}

    check(
        real.dtype in _allowed_dtypes() and imag.dtype in _allowed_dtypes(),
        lambda: (
            f"Expected both inputs to be Half, Float or Double tensors but got "
            f"{real.dtype} and {imag.dtype}"
        ),
    )
    higher_dtype = utils.get_higher_dtype(real.dtype, imag.dtype)
    check(
        higher_dtype is not None,
        lambda: f"Cannot get higher dtype: got {real.dtype} and {imag.dtype}",
    )
    result_dtype = utils.corresponding_complex_dtype(higher_dtype)  # type: ignore[arg-type]
    common_shape = _broadcast_shapes(real.shape, imag.shape)
    a = torch.empty(common_shape, dtype=result_dtype)
    a.real = real
    a.imag = imag
    return a
