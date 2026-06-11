#pragma once
#include <cmath>

namespace at::native {
template<typename scalar_t>
__forceinline__ __device__ double device_sqrt(scalar_t val) {
  return std::sqrt(val);
}
} // namespace at::native
