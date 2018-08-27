#pragma once

#include <cuda.h>
#include <limits.h>

// NumericLimits.cuh is a holder for numeric limits definitions of commonly used
// types. This header is very specific to ROCm HIP and may be removed in the future. 
// This header is derived from the legacy THCNumerics.cuh.

namespace at{

template <typename T>
struct numeric_limits {
};

// WARNING: the following at::numeric_limits definitions are there only to support
//          HIP compilation for the moment. Use std::numeric_limits if you are not
//          compiling for ROCm.
//          from @colesbury: "The functions on numeric_limits aren't marked with 
//          __device__ which is why they don't work with ROCm. CUDA allows them 
//          because they're constexpr."
template <>
struct numeric_limits<uint8_t> {
  static inline __host__ __device__ uint8_t lowest() { return 0; }
  static inline __host__ __device__ uint8_t max() { return UINT8_MAX; }
};

template <>
struct numeric_limits<int8_t> {
  static inline __host__ __device__ int8_t lowest() { return INT8_MIN; }
  static inline __host__ __device__ int8_t max() { return INT8_MAX; }
};

template <>
struct numeric_limits<int16_t> {
  static inline __host__ __device__ int16_t lowest() { return INT16_MIN; }
  static inline __host__ __device__ int16_t max() { return INT16_MAX; }
};

template <>
struct numeric_limits<int32_t> {
  static inline __host__ __device__ int32_t lowest() { return INT32_MIN; }
  static inline __host__ __device__ int32_t max() { return INT32_MAX; }
};

template <>
struct numeric_limits<int64_t> {
#ifdef _MSC_VER
  static inline __host__ __device__ int64_t lowest() { return _I64_MIN; }
  static inline __host__ __device__ int64_t max() { return _I64_MAX; }
#else
  static inline __host__ __device__ int64_t lowest() { return INT64_MIN; }
  static inline __host__ __device__ int64_t max() { return INT64_MAX; }
#endif
};

template <>
struct numeric_limits<at::Half> {
  static inline __host__ __device__ at::Half lowest() { return at::Half(0xFBFF, at::Half::from_bits); }
  static inline __host__ __device__ at::Half max() { return at::Half(0x7BFF, at::Half::from_bits); }
};

template <>
struct numeric_limits<float> {
  static inline __host__ __device__ float lowest() { return -FLT_MAX; }
  static inline __host__ __device__ float max() { return FLT_MAX; }
};

template <>
struct numeric_limits<double> {
  static inline __host__ __device__ double lowest() { return -DBL_MAX; }
  static inline __host__ __device__ double max() { return DBL_MAX; }
};

} // namespace at