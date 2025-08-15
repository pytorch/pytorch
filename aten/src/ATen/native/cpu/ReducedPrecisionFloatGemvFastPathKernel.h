#pragma once

#include <ATen/native/DispatchStub.h>
#include <c10/macros/Macros.h>
#include <c10/util/BFloat16.h>
#include <c10/util/Half.h>

namespace at::native {
#if !defined(C10_MOBILE)
using fp16_gemv_fn = void(*)(int, int, float, const Half*, int, const Half*, int, float, Half*, int);
DECLARE_DISPATCH(fp16_gemv_fn, fp16_gemv_trans_stub)

using bf16_gemv_fn = void(*)(int, int, BFloat16, const BFloat16*, int, const BFloat16*, int, BFloat16, BFloat16*, int);
DECLARE_DISPATCH(bf16_gemv_fn, bf16_gemv_trans_stub)

using fp16_dot_fn = float(*)(const int64_t, const Half*, const int64_t, const Half*, const int64_t);
DECLARE_DISPATCH(fp16_dot_fn, fp16_dot_stub)

using bf16_dot_fn = float(*)(const int64_t, const BFloat16*, const int64_t, const BFloat16*, const int64_t);
DECLARE_DISPATCH(bf16_dot_fn, bf16_dot_stub)

inline namespace CPU_CAPABILITY {
float fp16_dot_with_fp32_arith(const Half* vec1, const Half* vec2, int64_t len);
float bf16_dot_with_fp32_arith(const BFloat16* vec1, const BFloat16* vec2, int64_t len);
} // inline namespace CPU_CAPABILITY
#endif // !defined(C10_MOBILE)
} // namespace at::native
