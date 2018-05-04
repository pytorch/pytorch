#pragma once
#include "ATen/Config.h"

// Defines the accumulation type for a scalar type.
// Example:
//   using accscalar_t = acc_type<scalar_t, true>;

#if AT_CUDA_ENABLED()
#ifdef __CUDACC__
#include <cuda.h>
#include <cuda_fp16.h>
#endif
#endif

namespace at {

template <typename T, bool is_cuda>
struct AccumulateType { };

#if AT_CUDA_ENABLED()
#ifdef __CUDACC__
template <> struct AccumulateType<half, true> { using type = float; };
#endif
#endif
template <> struct AccumulateType<float, true> { using type = float; };
template <> struct AccumulateType<double, true> { using type = double; };
template <> struct AccumulateType<int8_t, true> { using type = int64_t; };
template <> struct AccumulateType<uint8_t, true> { using type = int64_t; };
template <> struct AccumulateType<char, true> { using type = int64_t; };
template <> struct AccumulateType<int16_t, true> { using type = int64_t; };
template <> struct AccumulateType<int32_t, true> { using type = int64_t; };
template <> struct AccumulateType<int64_t, true> { using type = int64_t; };
template <> struct AccumulateType<float, false> { using type = double; };
template <> struct AccumulateType<double, false> { using type = double; };
template <> struct AccumulateType<int8_t, false> { using type = int64_t; };
template <> struct AccumulateType<uint8_t, false> { using type = int64_t; };
template <> struct AccumulateType<char, false> { using type = int64_t; };
template <> struct AccumulateType<int16_t, false> { using type = int64_t; };
template <> struct AccumulateType<int32_t, false> { using type = int64_t; };
template <> struct AccumulateType<int64_t, false> { using type = int64_t; };

template<typename T, bool is_cuda>
using acc_type = typename AccumulateType<T, is_cuda>::type;

}  // namespace at
