#pragma once

#include <ATen/Half.h>
#include <ATen/cuda/CUDAHalf.cuh>

// Type traits to convert types to CUDA-specific types. Used primarily to
// convert at::Half to CUDA's half type. This makes the conversion explicit.

namespace at { namespace cuda {
template <typename T>
struct IntoTypeConversion {
  using type = T;
};

template <>
struct IntoTypeConversion<Half> {
  using type = half;
};

template <typename T>
using into_type = typename IntoTypeConversion<T>::type;

template <typename T>
struct FromTypeConversion {
  using type = T;
};

template <>
struct FromTypeConversion<half> {
  using type = Half;
};

template <typename T>
using from_type = typename FromTypeConversion<T>::type;

}} // namespace at::cuda
