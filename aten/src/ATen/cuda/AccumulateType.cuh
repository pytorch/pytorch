#pragma once

// Defines the accumulation type for a scalar type.
// Example:
//   using accscalar_t = cuda::acc_type<scalar_t>;

#include <cuda.h>
#include <cuda_fp16.h>

namespace at { namespace cuda {

template <typename T>
struct AccumulateType { };

template <> struct AccumulateType<half> { using type = float; };
template <> struct AccumulateType<float> { using type = float; };
template <> struct AccumulateType<double> { using type = double; };
template <> struct AccumulateType<int8_t> { using type = int64_t; };
template <> struct AccumulateType<uint8_t> { using type = int64_t; };
template <> struct AccumulateType<char> { using type = int64_t; };
template <> struct AccumulateType<int16_t> { using type = int64_t; };
template <> struct AccumulateType<int32_t> { using type = int64_t; };
template <> struct AccumulateType<int64_t> { using type = int64_t; };

template<typename T>
using acc_type = typename AccumulateType<T>::type;


}}  // namespace at::cuda
