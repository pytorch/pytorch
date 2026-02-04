#pragma once

#include <limits>

// at::numeric_limits is a historical artifact which was needed for ROCm HIP
// because std::numeric_limits functions are not marked __device__ and did not
// work with ROCm. This is no longer the case according to the discussion on
// #50902 and #52058.
//
// This header cannot be removed because lower_bound/upper_bound functions are
// not present in std::numeric_limits.
//
// The lower_bound and upper_bound constants are same as lowest and max for
// integral types, but are -inf and +inf for floating point types. They are
// useful in implementing min, max, etc.

namespace at {

template <typename T>
struct numeric_limits {
  static inline __host__ __device__ T lowest() {
    return std::numeric_limits<T>::lowest();
  }

  static inline __host__ __device__ T max() {
    return std::numeric_limits<T>::max();
  }

  static inline __host__ __device__ T lower_bound() {
    if constexpr (std::numeric_limits<T>::has_infinity) {
      return -std::numeric_limits<T>::infinity();
    } else {
      return std::numeric_limits<T>::lowest();
    }
  }

  static inline __host__ __device__ T upper_bound() {
    if constexpr (std::numeric_limits<T>::has_infinity) {
      return std::numeric_limits<T>::infinity();
    } else {
      return std::numeric_limits<T>::max();
    }
  }
};

} // namespace at
