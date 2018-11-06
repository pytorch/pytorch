#pragma once

/* This file defines math functions compatible across different gpu
 * platforms (currently CUDA and HIP).
 */

#ifdef __HIPCC__
#define __MATH_FUNCTIONS_DECL__ __device__ inline
#else /* __HIPCC__ */
#ifdef __CUDACC_RTC__
#define __MATH_FUNCTIONS_DECL__ __host__ __device__
#else /* __CUDACC_RTC__ */
#define __MATH_FUNCTIONS_DECL__ static inline __host__ __device__
#endif /* __CUDACC_RTC__ */
#endif /* __HIPCC__ */

namespace c10 {
namespace cuda {
namespace compat {

__MATH_FUNCTIONS_DECL__ float abs(float x) {
  return fabsf(x);
}
__MATH_FUNCTIONS_DECL__ double abs(double x) {
  return fabs(x);
}

__MATH_FUNCTIONS_DECL__ float max(float x, float y) {
  return fmaxf(x, y);
}
__MATH_FUNCTIONS_DECL__ double max(double x, double y) {
  return fmax(x, y);
}

__MATH_FUNCTIONS_DECL__ float pow(float x, float y) {
  return powf(x, y);
}

} // namespace compat
} // namespace cuda
} // namespace c10
