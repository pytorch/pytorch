#pragma once

#include <string>
#include <stdexcept>
#include <sstream>

namespace at { namespace native {

// Since real-to-complex satisfy the Hermitian symmetry, i.e.,
// X[m, \omega] = X[m, N - \omega]*. We return only the first floor(N / 2) + 1
// values by default (onesided=True). This is also the assumption in libraries
// including cuFFT and MKL.
inline int64_t infer_ft_real_to_complex_onesided_size(int64_t real_size) {
  return (real_size / 2) + 1;
}

inline int64_t infer_ft_complex_to_real_onesided_size(int64_t complex_size, int64_t expected_size=-1) {
  int64_t base = (complex_size - 1) * 2;
  if (expected_size < 0) {
    return base + 1;
  } else if (base == expected_size) {
    return base;
  } else if (base + 1 == expected_size) {
    return base + 1;
  } else {
    std::ostringstream ss;
    ss << "expected real signal size " << expected_size << " is incompatible "
       << "with onesided complex frequency size " << complex_size;
    throw std::runtime_error(ss.str());
  }
}

}} // at::native
