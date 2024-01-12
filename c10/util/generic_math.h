#pragma once

#include <c10/macros/Macros.h>
#include <c10/util/TypeSafeSignMath.h>
#include <cmath>

#if defined(__CUDA_ARCH__)
#include <c10/cuda/CUDAMathCompat.h>
#define C10_COMPAT_COPYSIGN c10::cuda::compat::copysign
#elif defined(__HIPCC__)
#include <c10/hip/HIPMathCompat.h>
#define C10_COMPAT_COPYSIGN c10::hip::compat::copysign
#else
#include <c10/util/copysign.h>
#define C10_COMPAT_COPYSIGN c10::copysign
#endif

// The functions in this file should be header-only as it is used under
// ABI-compatibility mode.

namespace c10 {

// NOTE: [Floor Division in Python]
// Python's __floordiv__ operator is more complicated than just floor(a / b).
// It aims to maintain the property: a == (a // b) * b + remainder(a, b)
// which can otherwise fail due to rounding errors in the remainder.
// So, instead it is calculated as: a // b = (a - remainder(a, b)) / b
// With some additional fix-ups added to the result.
//
// For reference, see CPython's implementation:
// https://github.com/python/cpython/blob/ace008c531dd685a30c1dd68f9b5ba35f20171cf/Objects/floatobject.c#L636

template <typename scalar_t>
inline C10_HOST_DEVICE scalar_t div_floor_floating(scalar_t a, scalar_t b)
    __ubsan_ignore_float_divide_by_zero__ {
  if (C10_UNLIKELY(b == 0)) {
    // Divide by zero: return standard IEEE result
    return a / b;
  }

  auto mod = std::fmod(a, b);
  auto div = (a - mod) / b;
  if ((mod != 0) && (b < 0) != (mod < 0)) {
    div -= scalar_t(1);
  }

  scalar_t floordiv;
  if (div != 0) {
    floordiv = std::floor(div);
    if (div - floordiv > scalar_t(0.5)) {
      floordiv += scalar_t(1.0);
    }
  } else {
    floordiv = C10_COMPAT_COPYSIGN(scalar_t(0), a / b);
  }
  return floordiv;
}

template <typename scalar_t>
inline C10_HOST_DEVICE scalar_t div_floor_integer(scalar_t a, scalar_t b) {
  if (c10::signs_differ(a, b)) {
    // Subtracts one from the results of truncation division if the
    // divisor and dividend have different sign(bit)s and the remainder of
    // the division is nonzero
    const auto quot = a / b;
    const auto rem = a % b;
    return rem ? quot - 1 : quot;
  }
  return a / b;
}

} // namespace c10
