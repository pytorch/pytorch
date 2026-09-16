#pragma once

#include <ATen/core/Tensor.h>
#include <c10/util/ArrayRef.h>

namespace at::native {

// rocFFT path for the batched 1-D last-dimension transforms that stft and istft
// issue. Opt-in via TORCH_ROCM_PREFER_ROCFFT; hipFFT stays the default.
bool use_rocfft_path(const Tensor& self, IntArrayRef dim);

Tensor _fft_r2c_rocfft(const Tensor& self, IntArrayRef dim, int64_t normalization, bool onesided);
Tensor _fft_c2r_rocfft(const Tensor& self, IntArrayRef dim, int64_t normalization, int64_t lastdim);
Tensor _fft_c2c_rocfft(const Tensor& self, IntArrayRef dim, int64_t normalization, bool forward);

} // namespace at::native
