// Copyright (c) 2025 PyTorch Contributors.
// All rights reserved.

#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/ATen.h>

namespace at {
namespace native {
namespace rdfft_utils {

static __device__ __forceinline__
uint32_t reverse_bits_32(uint32_t x) {
  x = ((x & 0xaaaaaaaa) >> 1) | ((x & 0x55555555) << 1);
  x = ((x & 0xcccccccc) >> 2) | ((x & 0x33333333) << 2);
  x = ((x & 0xf0f0f0f0) >> 4) | ((x & 0x0f0f0f0f) << 4);
  x = ((x & 0xff00ff00) >> 8) | ((x & 0x00ff00ff) << 8);
  return (x >> 16) | (x << 16);
}


static __device__ __forceinline__ float device_cos(float x) {
  return ::cosf(x);
}
static __device__ __forceinline__ float device_sin(float x) {
  return ::sinf(x);
}
static __device__ __forceinline__ double device_cos(double x) {
  return ::cos(x);
}
static __device__ __forceinline__ double device_sin(double x) {
  return ::sin(x);
}

} // namespace rdfft_utils
} // namespace native
} // namespace at
