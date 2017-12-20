#pragma once

// Defines the accumulation type for a scalar type.
// Example:
//   using acc_scalar = cuda::acc_type<scalar>;

#include <cuda.h>
#include <cuda_fp16.h>

namespace at { namespace native { namespace cuda {

template <typename T>
struct AccumulateType { using type = T; };

template <>
struct AccumulateType<half> { using type = float; };

template<typename T>
using acc_type = typename AccumulateType<T>::type;


}}}  // namespace at::native::cuda
