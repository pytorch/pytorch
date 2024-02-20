#pragma once

#include <string>
#include <stdexcept>
#include <sstream>
#include <c10/core/ScalarType.h>
#include <c10/util/ArrayRef.h>
#include <c10/util/Exception.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/core/TensorBase.h>

namespace at::native {

// Normalization types used in _fft_with_size
enum class fft_norm_mode {
  none,       // No normalization
  by_root_n,  // Divide by sqrt(signal_size)
  by_n,       // Divide by signal_size
};

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
    AT_ERROR(ss.str());
  }
}

using fft_fill_with_conjugate_symmetry_fn =
    void (*)(ScalarType dtype, IntArrayRef mirror_dims, IntArrayRef half_sizes,
             IntArrayRef in_strides, const void* in_data,
             IntArrayRef out_strides, void* out_data);
DECLARE_DISPATCH(fft_fill_with_conjugate_symmetry_fn, fft_fill_with_conjugate_symmetry_stub);

// In real-to-complex transform, cuFFT and MKL only fill half of the values
// due to conjugate symmetry. This function fills in the other half of the full
// fft by using the Hermitian symmetry in the signal.
// self should be the shape of the full signal and dims.back() should be the
// one-sided dimension.
// See NOTE [ Fourier Transform Conjugate Symmetry ]
TORCH_API void _fft_fill_with_conjugate_symmetry_(const Tensor& self, IntArrayRef dims);

} // namespace at::native
