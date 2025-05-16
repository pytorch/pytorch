#pragma once

// This header mimics APIs in aoti_torch/c/shim.h in a standalone way

// TODO: Move ScalarType to a header-only directory
#include <c10/core/ScalarType.h>

#ifdef __cplusplus
extern "C" {
#endif
#define AOTI_TORCH_DTYPE_IMPL(dtype, stype)   \
  inline int32_t aoti_torch_dtype_##dtype() { \
    return (int32_t)c10::ScalarType::stype;   \
  }

AOTI_TORCH_DTYPE_IMPL(float8_e5m2, Float8_e5m2)
AOTI_TORCH_DTYPE_IMPL(float8_e4m3fn, Float8_e4m3fn)
AOTI_TORCH_DTYPE_IMPL(float8_e5m2fnuz, Float8_e5m2fnuz)
AOTI_TORCH_DTYPE_IMPL(float8_e4m3fnuz, Float8_e4m3fnuz)
AOTI_TORCH_DTYPE_IMPL(bfloat16, BFloat16)
AOTI_TORCH_DTYPE_IMPL(float16, Half)
AOTI_TORCH_DTYPE_IMPL(float32, Float)
AOTI_TORCH_DTYPE_IMPL(float64, Double)
AOTI_TORCH_DTYPE_IMPL(uint8, Byte)
AOTI_TORCH_DTYPE_IMPL(uint16, UInt16)
AOTI_TORCH_DTYPE_IMPL(uint32, UInt32)
AOTI_TORCH_DTYPE_IMPL(uint64, UInt64)
AOTI_TORCH_DTYPE_IMPL(int8, Char)
AOTI_TORCH_DTYPE_IMPL(int16, Short)
AOTI_TORCH_DTYPE_IMPL(int32, Int)
AOTI_TORCH_DTYPE_IMPL(int64, Long)
AOTI_TORCH_DTYPE_IMPL(bool, Bool)
AOTI_TORCH_DTYPE_IMPL(complex32, ComplexHalf)
AOTI_TORCH_DTYPE_IMPL(complex64, ComplexFloat)
AOTI_TORCH_DTYPE_IMPL(complex128, ComplexDouble)
#undef AOTI_TORCH_DTYPE_IMPL

#ifdef __cplusplus
} // extern "C"
#endif
