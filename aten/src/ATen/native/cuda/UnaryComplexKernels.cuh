#pragma once

#include <complex>
#include <c10/util/complex.h>

namespace at { namespace native {
  
  // We manually overload conj because std::conj does not work types other than c10::complex.
  template<typename scalar_t>
  __host__ __device__ inline scalar_t conj_wrapper(scalar_t v) {
    return v;
  }
  
  template<typename T>
  __host__ __device__ inline c10::complex<T> conj_wrapper(c10::complex<T> v) {
    return std::conj(v);
  }
  
  }} // namespace at::native
