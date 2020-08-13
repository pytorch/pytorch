#pragma once

#include <ATen/ATen.h>

namespace torch {
namespace fft {

/// See the documentation of torch.fft.fft.
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

inline Tensor ifft(const Tensor& self,
                  c10::optional<int64_t> n=c10::nullopt,
                  int64_t axis=-1,
                  c10::optional<std::string> norm=c10::nullopt) {
  return torch::fft_ifft(self, n, axis, norm);
}

inline Tensor rfft(const Tensor& self,
                  c10::optional<int64_t> n=c10::nullopt,
                  int64_t axis=-1,
                  c10::optional<std::string> norm=c10::nullopt) {
  return torch::fft_rfft(self, n, axis, norm);
}

inline Tensor irfft(const Tensor& self,
                    c10::optional<int64_t> n=c10::nullopt,
                    int64_t axis=-1,
                    c10::optional<std::string> norm=c10::nullopt) {
  return torch::fft_irfft(self, n, axis, norm);
}

inline Tensor hfft(const Tensor& self,
                   c10::optional<int64_t> n=c10::nullopt,
                   int64_t axis=-1,
                   c10::optional<std::string> norm=c10::nullopt) {
  return torch::fft_hfft(self, n, axis, norm);
}

inline Tensor ihfft(const Tensor& self,
                    c10::optional<int64_t> n=c10::nullopt,
                    int64_t axis=-1,
                    c10::optional<std::string> norm=c10::nullopt) {
  return torch::fft_ihfft(self, n, axis, norm);
}

}} // torch::fft
