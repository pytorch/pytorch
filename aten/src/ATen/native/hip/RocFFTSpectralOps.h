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

// Fused stft core: gathers the frames out of `self` (batch, signal) and applies
// `window` inside the transform's load, so the framed tensor stft would
// otherwise build is never materialized. Returns an undefined tensor when the
// fused path does not apply, leaving the caller to run the unfused version.
Tensor stft_r2c_rocfft(const Tensor& self, int64_t n_fft, int64_t hop_length, int64_t n_frames,
                       const Tensor& window, bool onesided, int64_t normalization);

// Fused istft core: applies the synthesis `window` inside the transform's store,
// so the windowed frames land in the C2R output rather than in a second tensor
// built by a separate pass. Returns an undefined tensor when the fused path does
// not apply, leaving the caller to run the unfused version.
Tensor istft_c2r_rocfft(const Tensor& self, int64_t n_fft, const Tensor& window,
                        int64_t normalization);

} // namespace at::native
