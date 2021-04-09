#pragma once

/* This file defines math functions compatible across different gpu
 * platforms (currently CUDA and HIP).
 */
#if defined(__CUDACC__) || defined(__HIPCC__)

#include <c10/macros/Macros.h>
#include <c10/util/Exception.h>

#ifdef __HIPCC__
#define __MATH_FUNCTIONS_DECL__ inline C10_DEVICE
#else /* __HIPCC__ */
#ifdef __CUDACC_RTC__
#define __MATH_FUNCTIONS_DECL__ C10_HOST_DEVICE
#else /* __CUDACC_RTC__ */
#define __MATH_FUNCTIONS_DECL__ static inline C10_HOST_DEVICE
#endif /* __CUDACC_RTC__ */
#endif /* __HIPCC__ */

namespace c10 {
namespace cuda {
namespace compat {

__MATH_FUNCTIONS_DECL__ float abs(float x) {
  return ::fabsf(x);
}
__MATH_FUNCTIONS_DECL__ double abs(double x) {
  return ::fabs(x);
}

__MATH_FUNCTIONS_DECL__ float exp(float x) {
  return ::expf(x);
}
__MATH_FUNCTIONS_DECL__ double exp(double x) {
  return ::exp(x);
}

__MATH_FUNCTIONS_DECL__ float ceil(float x) {
  return ::ceilf(x);
}
__MATH_FUNCTIONS_DECL__ double ceil(double x) {
  return ::ceil(x);
}

__MATH_FUNCTIONS_DECL__ float copysign(float x, float y) {
#if defined(__CUDA_ARCH__) || defined(__HIPCC__)
  return ::copysignf(x, y);
#else
  // std::copysign gets ICE/Segfaults with gcc 7.5/8 on arm64
  // (e.g. Jetson), see PyTorch PR #51834
  // This host function needs to be here for the compiler but is never used
  TORCH_INTERNAL_ASSERT(
      false, "CUDAMathCompat copysign should not run on the CPU");
#endif
}
__MATH_FUNCTIONS_DECL__ double copysign(double x, double y) {
#if defined(__CUDA_ARCH__) || defined(__HIPCC__)
  return ::copysign(x, y);
#else
  // see above
  TORCH_INTERNAL_ASSERT(
      false, "CUDAMathCompat copysign should not run on the CPU");
#endif
}

__MATH_FUNCTIONS_DECL__ float floor(float x) {
  return ::floorf(x);
}
__MATH_FUNCTIONS_DECL__ double floor(double x) {
  return ::floor(x);
}

__MATH_FUNCTIONS_DECL__ float log(float x) {
  return ::logf(x);
}
__MATH_FUNCTIONS_DECL__ double log(double x) {
  return ::log(x);
}

__MATH_FUNCTIONS_DECL__ float log1p(float x) {
  return ::log1pf(x);
}

__MATH_FUNCTIONS_DECL__ double log1p(double x) {
  return ::log1p(x);
}

__MATH_FUNCTIONS_DECL__ float max(float x, float y) {
  return ::fmaxf(x, y);
}
__MATH_FUNCTIONS_DECL__ double max(double x, double y) {
  return ::fmax(x, y);
}

__MATH_FUNCTIONS_DECL__ float min(float x, float y) {
  return ::fminf(x, y);
}
__MATH_FUNCTIONS_DECL__ double min(double x, double y) {
  return ::fmin(x, y);
}

__MATH_FUNCTIONS_DECL__ float pow(float x, float y) {
  return ::powf(x, y);
}
__MATH_FUNCTIONS_DECL__ double pow(double x, double y) {
  return ::pow(x, y);
}

__MATH_FUNCTIONS_DECL__ void sincos(float x, float* sptr, float* cptr) {
  return ::sincosf(x, sptr, cptr);
}
__MATH_FUNCTIONS_DECL__ void sincos(double x, double* sptr, double* cptr) {
  return ::sincos(x, sptr, cptr);
}

__MATH_FUNCTIONS_DECL__ float sqrt(float x) {
  return ::sqrtf(x);
}
__MATH_FUNCTIONS_DECL__ double sqrt(double x) {
  return ::sqrt(x);
}

__MATH_FUNCTIONS_DECL__ float rsqrt(float x) {
  return ::rsqrtf(x);
}
__MATH_FUNCTIONS_DECL__ double rsqrt(double x) {
  return ::rsqrt(x);
}

__MATH_FUNCTIONS_DECL__ float tan(float x) {
  return ::tanf(x);
}
__MATH_FUNCTIONS_DECL__ double tan(double x) {
  return ::tan(x);
}

__MATH_FUNCTIONS_DECL__ float tanh(float x) {
  return ::tanhf(x);
}
__MATH_FUNCTIONS_DECL__ double tanh(double x) {
  return ::tanh(x);
}

__MATH_FUNCTIONS_DECL__ float normcdf(float x) {
  return ::normcdff(x);
}
__MATH_FUNCTIONS_DECL__ double normcdf(double x) {
  return ::normcdf(x);
}

} // namespace compat
} // namespace cuda
} // namespace c10

#endif
