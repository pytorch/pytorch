#pragma once

#include <cuda.h>
#include <c10/util/Half.h>
#include <c10/util/BFloat16.h>

#include <ATen/NumericUtils.h>

#if !(defined(USE_ROCM) || ((defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 800))))
#include <cuda_bf16.h>
#endif

#if defined(USE_ROCM)
#include <ATen/cuda/detail/ROCmMacros.cuh>
#endif
#include <torch/headeronly/cuda/Atomic.h>

// Atomic multiplication implementation.

ATOMIC_INTEGER_IMPL(Mul)
GPU_ATOMIC_INTEGER(Mul, a * b, uint8_t)
GPU_ATOMIC_INTEGER(Mul, a * b, int8_t)
GPU_ATOMIC_INTEGER(Mul, a * b, int16_t)
GPU_ATOMIC_INTEGER(Mul, a * b, int32_t)
GPU_ATOMIC_INTEGER(Mul, a * b, int64_t)

inline __device__ at::Half gpuAtomicMul(at::Half * address, at::Half val) {
  return AtomicFPOp<at::Half>()(address, val,
                                [](at::Half bsum, at::Half val) {
                                  return bsum * val;
                                });
}

inline __device__ at::BFloat16 gpuAtomicMul(at::BFloat16 * address, at::BFloat16 val) {
  return AtomicFPOp<at::BFloat16>()(address, val,
                                    [](at::BFloat16 bsum, at::BFloat16 val) {
                                      return bsum * val;
                                    });
}

inline __device__ double gpuAtomicMul(double * address, double val) {
  return AtomicFPOp<double>()(address, val,
                              [](double val, unsigned long long int assumed) {
                                return __double_as_longlong(val * __longlong_as_double(assumed));
                              });
}

// Dont use a templated function for this since the addition function defaults to the CUDA built-in.
inline __device__ float gpuAtomicMul (float * address, float val) {
  unsigned int* address_as_ull = (unsigned int*)address;
  unsigned int old = *address_as_ull;
  unsigned int assumed;

  do {
    assumed = old;
    old = atomicCAS(address_as_ull, assumed,
                    __float_as_int(val *
                                   __int_as_float(assumed)));

    // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
  } while (assumed != old);

  return __int_as_float(old);
}

// Atomic maximum implementation.

template <typename T>
__host__ __device__ T safe_max(T a, T b) {
  T max = at::_isnan(b) ? b : std::max<T>(a, b);
  return max;
}

ATOMIC_INTEGER_IMPL(Max)
GPU_ATOMIC_INTEGER(Max, safe_max(a, b), uint8_t)
GPU_ATOMIC_INTEGER(Max, safe_max(a, b), int8_t)
GPU_ATOMIC_INTEGER(Max, safe_max(a, b), int16_t)
GPU_ATOMIC_INTEGER(Max, safe_max(a, b), int32_t)
GPU_ATOMIC_INTEGER(Max, safe_max(a, b), int64_t)

inline __device__ at::Half gpuAtomicMax(at::Half * address, at::Half val) {
  return AtomicFPOp<at::Half>()(address, val,
                                [](at::Half bsum, at::Half val) {
                                  return safe_max(bsum, val);
                                });
}

inline __device__ at::BFloat16 gpuAtomicMax(at::BFloat16 * address, at::BFloat16 val) {
  return AtomicFPOp<at::BFloat16>()(address, val,
                                    [](at::BFloat16 bsum, at::BFloat16 val) {
                                      return safe_max(bsum, val);
                                    });
}

inline __device__ double gpuAtomicMax(double * address, double val) {
  return AtomicFPOp<double>()(address, val,
                              [](double val, unsigned long long int assumed) {
                                return __double_as_longlong(safe_max(val, __longlong_as_double(assumed)));
                              });
}

// Dont use a templated function for this since the addition function defaults to the CUDA built-in.
inline __device__ float gpuAtomicMax(float * address, float val) {
  unsigned int* address_as_ull = (unsigned int*)address;
  unsigned int old = *address_as_ull;
  unsigned int assumed;

  do {
    assumed = old;
    old = atomicCAS(address_as_ull, assumed,
                    __float_as_int(safe_max(val, __int_as_float(assumed))));

    // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
  } while (assumed != old);

  return __int_as_float(old);
}

// Atomic minimum implementation.

template <typename T>
__host__ __device__ T safe_min(T a, T b) {
  T min = at::_isnan(b) ? b : std::min<T>(a, b);
  return min;
}

ATOMIC_INTEGER_IMPL(Min)
GPU_ATOMIC_INTEGER(Min, safe_min(a, b), uint8_t)
GPU_ATOMIC_INTEGER(Min, safe_min(a, b), int8_t)
GPU_ATOMIC_INTEGER(Min, safe_min(a, b), int16_t)
GPU_ATOMIC_INTEGER(Min, safe_min(a, b), int32_t)
GPU_ATOMIC_INTEGER(Min, safe_min(a, b), int64_t)

inline __device__ at::Half gpuAtomicMin(at::Half * address, at::Half val) {
  return AtomicFPOp<at::Half>()(address, val,
                                [](at::Half bsum, at::Half val) {
                                  return safe_min(bsum, val);
                                });
}

inline __device__ at::BFloat16 gpuAtomicMin(at::BFloat16 * address, at::BFloat16 val) {
  return AtomicFPOp<at::BFloat16>()(address, val,
                                    [](at::BFloat16 bsum, at::BFloat16 val) {
                                      return safe_min(bsum, val);
                                    });
}

inline __device__ double gpuAtomicMin(double * address, double val) {
  return AtomicFPOp<double>()(address, val,
                              [](double val, unsigned long long int assumed) {
                                return __double_as_longlong(safe_min(val, __longlong_as_double(assumed)));
                              });
}

// Dont use a templated function for this since the addition function defaults to the CUDA built-in.
inline __device__ float gpuAtomicMin(float * address, float val) {
  unsigned int* address_as_ull = (unsigned int*)address;
  unsigned int old = *address_as_ull;
  unsigned int assumed;

  do {
    assumed = old;
    old = atomicCAS(address_as_ull, assumed,
                    __float_as_int(safe_min(val, __int_as_float(assumed))));

    // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
  } while (assumed != old);

  return __int_as_float(old);
}
