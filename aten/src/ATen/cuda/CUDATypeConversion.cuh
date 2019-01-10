#pragma once

#include <ATen/Half.h>
#include <ATen/cuda/CUDAHalf.cuh>

// Type traits to convert types to CUDA-specific types. Used primarily to
// convert at::Half to CUDA's half type. This makes the conversion explicit.

namespace at { namespace cuda {
template <typename T>
struct TypeConversion {
  using type = T;
};

template <>
struct TypeConversion<Half> {
  using type = half;
};

template <typename T>
using type = typename TypeConversion<T>::type;
}} // namespace at::cuda
