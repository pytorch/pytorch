#pragma once

#include <cuda.h>
#include <limits.h>
#include <assert.h>
#include <typeinfo>

// CUDANumerics.cuh contains mathematical functions that are either
// not in the std namespace or are specialized for compilation
// with CUDA NVCC or CUDA NVRTC. This header is derived from the
// legacy THCNumerics.cuh.

namespace at{

template <typename T>
struct numerics {
};

template <typename scalar_t>
static inline __host__ __device__ scalar_t powi(scalar_t a, scalar_t b) {
  // ge is only needed to not cause "pointless comparison warning"
  assert(numerics<scalar_t>::ge(b, 0));
  scalar_t result = 1;
  while (b) {
    if (b & 1) {
       result *= a;
    }
    b /= 2;
    a *= a;
  }
  return result;
}

template <>
struct numerics<uint8_t> {
  static inline __host__ __device__  bool ge(uint8_t a, uint8_t b) { return a >= b; }
  static inline __host__ __device__  uint8_t abs(uint8_t a) { return a; }
  static inline __host__ __device__  uint8_t pow(uint8_t a, uint8_t b) { return powi<uint8_t>(a, b); }
  static inline __host__ __device__  bool isnan(uint8_t a) { return false; }
  static inline __host__ __device__  bool isinf(uint8_t a) { return false; }
};

template <>
struct numerics<int8_t> {
  static inline __host__ __device__   bool ge(int8_t a, int8_t b) { return a >= b; }
  static inline __host__ __device__  int8_t abs(int8_t a) { return ::abs((int)a); }
  static inline __host__ __device__  int8_t pow(int8_t a, int8_t b) { return powi<int8_t>(a, b); }
  static inline __host__ __device__  bool isnan(int8_t a) { return false; }
  static inline __host__ __device__  bool isinf(int8_t a) { return false; }
};

template <>
struct numerics<int16_t> {
  static inline __host__ __device__  bool ge(int16_t a, int16_t b) { return a >= b; }
  static inline __host__ __device__  int16_t abs(int16_t a) { return ::abs((int)a); }
  static inline __host__ __device__  int16_t pow(int16_t a, int16_t b) { return powi<int16_t>(a, b); }
  static inline __host__ __device__  bool isnan(int16_t a) { return false; }
  static inline __host__ __device__  bool isinf(int16_t a) { return false; }
};

template <>
struct numerics<int32_t> {
  static inline __host__ __device__  bool ge(int32_t a, int32_t b) { return a >= b; }
  static inline __host__ __device__  int32_t abs(int32_t a) { return ::abs(a); }
  static inline __host__ __device__  int32_t pow(int32_t a, int32_t b) { return powi<int32_t>(a, b); }
  static inline __host__ __device__  bool isnan(int32_t a) { return false; }
  static inline __host__ __device__  bool isinf(int32_t a) { return false; }
};

template <>
struct numerics<int64_t> {
  static inline __host__ __device__  bool ge(int64_t a, int64_t b) { return a >= b; }
  static inline __host__ __device__  int64_t abs(int64_t a) { return labs(a); }
  static inline __host__ __device__  int64_t pow(int64_t a, int64_t b) { return powi<int64_t>(a, b); }
  static inline __host__ __device__  bool isnan(int64_t a) { return false; }
  static inline __host__ __device__  bool isinf(int64_t a) { return false; }
};

template <>
struct numerics<at::Half> {

  static inline __host__ __device__ at::Half exp10(at::Half a) {
      return (at::Half)(exp10f((float)a));
  }

  static inline __host__ __device__ at::Half rsqrt(at::Half a) {
      return (at::Half)(rsqrtf((float)a));
  }

  static inline __host__ __device__ at::Half erfinv(at::Half a) {
    return (at::Half)(erfinvf((float)a));
  }

  static inline __host__ __device__ at::Half frac(at::Half a) {
    #ifdef __CUDA_ARCH__
        return a - (at::Half)(truncf((float)a));
    #else // __CUDA_ARCH__
        return a - (at::Half)(floorf((float)a));
    #endif
  }

  static inline __host__ __device__ at::Half cinv(at::Half a) {
    return (at::Half)(1.0f / (float)a);
  }

};

template <>
struct numerics<float> {
  static inline __host__ __device__  float erfinv(float a) { return erfinvf(a);}
  static inline __host__ __device__  float exp10(float a) { return exp10f(a); }
  static inline __host__ __device__  float rsqrt(float a) { return rsqrtf(a); }
  static inline __host__ __device__  float frac (float a) { return a - truncf(a); }
  static inline __host__ __device__  float cinv (float a) { return 1.0f / a; }
};

template <>
struct numerics<double> {
  static inline __host__ __device__  double erfinv(double a) { return ::erfinv(a);}
  static inline __host__ __device__  double exp10(double a) { return ::exp10(a); }
  static inline __host__ __device__  double rsqrt(double a) { return ::rsqrt(a); }
  static inline __host__ __device__  double frac (double a) { return a - ::trunc(a); }
  static inline __host__ __device__  double cinv (double a) { return 1.0 / a; }
};

} // namespace at