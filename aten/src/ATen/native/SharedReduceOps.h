#pragma once
// Please note that this file is
// used across both CPU and GPU.

#include <c10/macros/Macros.h>
#if defined(__CUDACC__)
#include <THC/THCDeviceUtils.cuh>
#include <ATen/native/cuda/DeviceSqrt.cuh>
#elif defined(__HIPCC__)
#include <THH/THHDeviceUtils.cuh>
#include <ATen/native/hip/DeviceSqrt.cuh>
#else
#include <cmath>
#define device_sqrt std::sqrt
#endif

namespace at { namespace native {

template <typename scalar_t>
struct WelfordData {
  scalar_t mean;
  scalar_t m2;
  int64_t n;
  C10_HOST_DEVICE WelfordData() : mean(0), m2(0), n(0)  {}
  C10_DEVICE WelfordData(scalar_t mean, scalar_t m2, int64_t n) : mean(mean), m2(m2), n(n) {}
};


template <typename scalar_t, typename acc_scalar_t>
struct WelfordOps {
  bool unbiased;
 public:
  using acc_t = WelfordData<acc_scalar_t>;
  inline C10_DEVICE acc_t reduce(acc_t acc, scalar_t data) const {
    acc_scalar_t delta = data - acc.mean;
    acc_scalar_t new_mean = acc.mean + delta / (acc.n + 1);
    acc_scalar_t new_delta = data - new_mean;
    return {
      new_mean,
      acc.m2 + delta * new_delta,
      acc.n + 1
    };
  }
  inline C10_DEVICE acc_t combine(acc_t a, acc_t b) const {
    if (a.n == 0) {
      return b;
    }
    if (b.n == 0) {
      return a;
    }
    acc_scalar_t delta = b.mean - a.mean;
    int64_t new_count = a.n + b.n;
    acc_scalar_t nb_over_n = (scalar_t)b.n / new_count;
    return {
      a.mean + delta * nb_over_n,
      a.m2 + b.m2 + delta * delta * a.n * nb_over_n,
      new_count
    };
  }
  inline C10_DEVICE scalar_t project(acc_t acc) const {
    int64_t divisor = unbiased ? (acc.n - 1) : acc.n;
    return (divisor > 0) ? (scalar_t)device_sqrt(acc.m2 / divisor) : (scalar_t)NAN;
  }
#if defined(__CUDACC__) || defined(__HIPCC__)
  inline __device__ acc_t warp_shfl_down(acc_t acc, int offset) const {
    return {
      WARP_SHFL_DOWN(acc.mean, offset)
      , WARP_SHFL_DOWN(acc.m2, offset)
      , WARP_SHFL_DOWN(acc.n, offset)
    };
  }
#endif
  WelfordOps(bool unbiased) : unbiased(unbiased) {
  }
};

template <typename acc_t, typename factor_t>
struct MeanOps {
  factor_t factor;

  inline C10_DEVICE acc_t reduce(acc_t a, acc_t b) const {
    return a + b;
  }

  inline C10_DEVICE acc_t combine(acc_t a, acc_t b) const {
    return reduce(a, b);
  }

  inline C10_DEVICE acc_t project(acc_t a) const {
    return a * factor;
  }

#if defined(__CUDACC__) || defined(__HIPCC__)
  inline C10_DEVICE acc_t warp_shfl_down(acc_t data, int offset) const {
    return WARP_SHFL_DOWN(data, offset);
  }
#endif

  MeanOps(factor_t factor): factor(factor) {
  }
};


}} // namespace at::native
