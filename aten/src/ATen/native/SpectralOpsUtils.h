#pragma once

#include <string>
#include <stdexcept>
#include <sstream>

namespace at { namespace native {

// NOTE [ Fourier Transform Conjugate Symmetry ]
//
// Real-to-complex Fourier transform satisfies the conjugate symmetry. That is,
// assuming X is the transformed K-dimensionsal signal, we have
//
//     X[i_1, ..., i_K] = X[j_i, ..., j_K]*,
//
//       where j_k  = (N_k - i_k)  mod N_k, N_k being the signal size at dim k,
//             * is the conjugate operator.
//
// Therefore, in such cases, FFT libraries return only roughly half of the
// values to avoid redundancy:
//
//     X[:, :, ..., :floor(N / 2) + 1]
//
// This is also the assumption in cuFFT and MKL. In ATen SpectralOps, such
// halved signal will also be returned by default (flag onesided=True).
// The following infer_ft_real_to_complex_onesided_size function calculates the
// onesided size from the twosided size.
//
// Note that this loses some information about the size of signal at last
// dimension. E.g., both 11 and 10 maps to 6. Hence, the following
// infer_ft_complex_to_real_onesided_size function takes in optional parameter
// to infer the twosided size from given onesided size.
//
// cuFFT doc: http://docs.nvidia.com/cuda/cufft/index.html#multi-dimensional
// MKL doc: https://software.intel.com/en-us/mkl-developer-reference-c-dfti-complex-storage-dfti-real-storage-dfti-conjugate-even-storage#CONJUGATE_EVEN_STORAGE

inline int64_t infer_ft_real_to_complex_onesided_size(int64_t real_size) {
  return (real_size / 2) + 1;
}

inline int64_t infer_ft_complex_to_real_onesided_size(int64_t complex_size,
                                                      int64_t expected_size=-1) {
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
