#pragma once

#include <ATen/native/DispatchStub.h>
#include <c10/macros/Macros.h>
#include <c10/util/Half.h>

namespace at::native {
#if !defined(C10_MOBILE)
using fp16_dot_fn = float(*)(const Half*, const Half*, int64_t);
using fp16_gemv_fn = void(*)(int, int, float, const Half*, int, const Half*, int, float, Half*, int);
DECLARE_DISPATCH(fp16_dot_fn, fp16_dot_with_fp32_arith_stub)
DECLARE_DISPATCH(fp16_gemv_fn, fp16_gemv_trans_stub)
#endif // !defined(C10_MOBILE)
} // namespace at::native
