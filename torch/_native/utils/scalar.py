import math
import struct

import torch


def check_scalar(value: bool | int | float, dtype: torch.dtype) -> None:
    """Check scalar conversion without narrowing integers before validation."""
    if dtype is torch.bool:
        return
    if dtype.is_floating_point:
        round_float(value, dtype)
        return
    limits = torch.iinfo(dtype)
    if not limits.min <= value <= limits.max:
        raise OverflowError(f"scalar {value} overflows {dtype}")


def f32(value: int | float) -> float:
    r"""Round a Python number to FP32 using ties to even."""
    if isinstance(value, int):
        # Resolve the FP32 tie while the integer remainder is still exact.
        magnitude = abs(value)
        shift = magnitude.bit_length() - 24
        if shift > 0:
            bias = (1 << (shift - 1)) - 1 + ((magnitude >> shift) & 1)
            magnitude = ((magnitude + bias) >> shift) << shift
            value = -magnitude if value < 0 else magnitude
    try:
        return struct.unpack("f", struct.pack("f", value))[0]
    except OverflowError:
        return -math.inf if value < 0 else math.inf


def round_float(value: int | float, dtype: torch.dtype) -> float:
    r"""Match scalar conversion through FP32 for narrow floating dtypes."""
    if math.isfinite(value) and abs(value) > torch.finfo(dtype).max:
        raise OverflowError(f"scalar {value} overflows {dtype}")
    if dtype is torch.float64:
        return float(value)
    value = f32(value)
    if dtype is torch.float32:
        return value
    if dtype is torch.float16:
        return struct.unpack("e", struct.pack("e", value))[0]
    if dtype is torch.bfloat16:
        bits = struct.unpack("I", struct.pack("f", value))[0]
        if bits & 0x7FFFFFFF > 0x7F800000:
            bits |= 0x00400000
        else:
            bits += 0x7FFF + ((bits >> 16) & 1)
        return struct.unpack("f", struct.pack("I", bits & 0xFFFF0000))[0]
    raise ValueError(f"unsupported floating dtype: {dtype}")
