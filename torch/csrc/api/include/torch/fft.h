#pragma once

#include <ATen/ATen.h>

namespace torch {
namespace fft {

/// Computes the 1 dimensional fast fourier transform over a given axis
///
/// \see https://pytorch.org/docs/master/fft.html#torch.fft.fft
///
/// Example:
/// ```
/// auto t = torch::randn(128, dtype=kComplexDouble);
/// torch::fft::fft(t);
/// ```
inline Tensor fft(const Tensor& self,
                  c10::optional<int64_t> n=c10::nullopt,
                  int64_t axis=-1,
                  c10::optional<std::string> norm=c10::nullopt) {
  return torch::fft_fft(self, n, axis, norm);
}

/// Computes the 1 dimensional inverse fast fourier transform over a given axis
///
/// \see https://pytorch.org/docs/master/fft.html#torch.fft.ifft
///
/// Example:
/// ```
/// auto t = torch::randn(128, dtype=kComplexDouble);
/// torch::fft::ifft(t);
/// ```
inline Tensor ifft(const Tensor& self,
                  c10::optional<int64_t> n=c10::nullopt,
                  int64_t axis=-1,
                  c10::optional<std::string> norm=c10::nullopt) {
  return torch::fft_ifft(self, n, axis, norm);
}

/// Computes the 1 dimensional FFT of real input with half-complex output
///
/// \see https://pytorch.org/docs/master/fft.html#torch.fft.rfft
///
/// Example:
/// ```
/// auto t = torch::randn(128);
/// auto T = torch::fft::rfft(t);
/// assert(T.is_complex() && T.numel() == 128 / 2 + 1);
/// ```
inline Tensor rfft(const Tensor& self,
                  c10::optional<int64_t> n=c10::nullopt,
                  int64_t axis=-1,
                  c10::optional<std::string> norm=c10::nullopt) {
  return torch::fft_rfft(self, n, axis, norm);
}

/// Computes the inverse of torch.fft.rfft
///
/// The input is a half-complex fourier domain signal, with purely real output.
///
/// \see https://pytorch.org/docs/master/fft.html#torch.fft.irfft
///
/// Example:
/// ```
/// auto T = torch::randn(128 / 2 + 1, torch::kComplexDouble);
/// auto t = torch::fft::irfft(t, /*n=*/128);
/// assert(!t.is_complex() && T.numel() == 128);
/// ```
inline Tensor irfft(const Tensor& self,
                    c10::optional<int64_t> n=c10::nullopt,
                    int64_t axis=-1,
                    c10::optional<std::string> norm=c10::nullopt) {
  return torch::fft_irfft(self, n, axis, norm);
}

/// Computes the 1 dimensional FFT of a half-complex signal
///
/// The half-complex input represents a hermitian symmetric time domain signal.
/// The returned fourier domain representation of such a signal is a purely
/// real.
///
/// \see https://pytorch.org/docs/master/fft.html#torch.fft.hfft
///
/// Example:
/// ```
/// auto t = torch::randn(128 / 2 + 1, torch::kComplexDouble);
/// auto T = torch::fft::hfft(t, /*n=*/128);
/// assert(!T.is_complex() && T.numel() == 128);
/// ```
inline Tensor hfft(const Tensor& self,
                   c10::optional<int64_t> n=c10::nullopt,
                   int64_t axis=-1,
                   c10::optional<std::string> norm=c10::nullopt) {
  return torch::fft_hfft(self, n, axis, norm);
}

/// Computes the inverse FFT of a purely real fourier domain signal.
///
/// The output is a half-complex representation of the full hermitian symmetric
/// time domain signal.
///
/// \see https://pytorch.org/docs/master/fft.html#torch.fft.ihfft
///
/// Example:
/// ```
/// auto T = torch::randn(128, torch::kDouble);
/// auto t = torch::fft::ihfft(t);
/// assert(t.is_complex() && T.numel() == 128 / 2 + 1);
/// ```
inline Tensor ihfft(const Tensor& self,
                    c10::optional<int64_t> n=c10::nullopt,
                    int64_t axis=-1,
                    c10::optional<std::string> norm=c10::nullopt) {
  return torch::fft_ihfft(self, n, axis, norm);
}

}} // torch::fft
