#pragma once

#include <math.h>
#include <utility>

#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>

#include <ATen/Utils.h>
#include <ATen/Generator.h>
#include <ATen/CUDAGeneratorImpl.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAFunctions.h>

// 2^-10 is the step for normal FP16 numbers.
// 2^-24 is the unit in the last place (ULP)/precision limitation.
// 24 is **NOT** related to the number of mantissa bits of single precision format.
__device__ const float TWO_10 = 0.0009765625;
__device__ const float TWO_24 = 0.000000059604644775390625;


template<typename T>
__device__ __forceinline__ T maybe_upcast(__half x){
  return T(__half2float(x));
}

template<>
__device__ __forceinline__ __half maybe_upcast<__half>(__half x){
  return x;
}

__device__ __forceinline__ float get_delta_fp16(float x) {
  int exponent;
  frexpf(x, &exponent);
  exponent -= 1;
  if (exponent >= -14)
    return TWO_10 * std::pow(2, exponent);
  else
    return TWO_24;
}

// Natalia magic
template <typename scalar_t>
__device__ __forceinline__ scalar_t round_stochastically(float x, float random_value) {
  if (x == 0.0) {
    return scalar_t(0.0);
  }
  float delta = get_delta_fp16(x);
  float val;
  if (x < 0.0) {
    val = x - random_value * delta;
  } else {
    val = x + random_value * delta;
  }
  return maybe_upcast<scalar_t>(__float2half_rz(val));
}
