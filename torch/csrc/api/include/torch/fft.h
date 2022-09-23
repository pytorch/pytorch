#pragma once

#include <ATen/ATen.h>

namespace torch {
namespace fft {

/// Computes the 1 dimensional fast Fourier transform over a given dimension.
/// See https://pytorch.org/docs/master/fft.html#torch.fft.fft.
///
/// Example:
/// ```
/// auto t = torch::randn(128, dtype=kComplexDouble);
/// torch::fft::fft(t);
/// ```
inline Tensor fft(
    const Tensor& self,
    c10::optional<int64_t> n = c10::nullopt,
    int64_t dim = -1,
    c10::optional<c10::string_view> norm = c10::nullopt) {
  return torch::fft_fft(self, n, dim, norm);
}

/// Computes the 1 dimensional inverse Fourier transform over a given dimension.
/// See https://pytorch.org/docs/master/fft.html#torch.fft.ifft.
///
/// Example:
/// ```
/// auto t = torch::randn(128, dtype=kComplexDouble);
/// torch::fft::ifft(t);
/// ```
inline Tensor ifft(
    const Tensor& self,
    c10::optional<int64_t> n = c10::nullopt,
    int64_t dim = -1,
    c10::optional<c10::string_view> norm = c10::nullopt) {
  return torch::fft_ifft(self, n, dim, norm);
}

/// Computes the 2-dimensional fast Fourier transform over the given dimensions.
/// See https://pytorch.org/docs/master/fft.html#torch.fft.fft2.
///
/// Example:
/// ```
/// auto t = torch::randn({128, 128}, dtype=kComplexDouble);
/// torch::fft::fft2(t);
/// ```
inline Tensor fft2(
    const Tensor& self,
    OptionalIntArrayRef s = c10::nullopt,
    IntArrayRef dim = {-2, -1},
    c10::optional<c10::string_view> norm = c10::nullopt) {
  return torch::fft_fft2(self, s, dim, norm);
}

/// Computes the inverse of torch.fft.fft2
/// See https://pytorch.org/docs/master/fft.html#torch.fft.ifft2.
///
/// Example:
/// ```
/// auto t = torch::randn({128, 128}, dtype=kComplexDouble);
/// torch::fft::ifft2(t);
/// ```
inline Tensor ifft2(
    const Tensor& self,
    at::OptionalIntArrayRef s = c10::nullopt,
    IntArrayRef dim = {-2, -1},
    c10::optional<c10::string_view> norm = c10::nullopt) {
  return torch::fft_ifft2(self, s, dim, norm);
}

/// Computes the N dimensional fast Fourier transform over given dimensions.
/// See https://pytorch.org/docs/master/fft.html#torch.fft.fftn.
///
/// Example:
/// ```
/// auto t = torch::randn({128, 128}, dtype=kComplexDouble);
/// torch::fft::fftn(t);
/// ```
inline Tensor fftn(
    const Tensor& self,
    at::OptionalIntArrayRef s = c10::nullopt,
    at::OptionalIntArrayRef dim = c10::nullopt,
    c10::optional<c10::string_view> norm = c10::nullopt) {
  return torch::fft_fftn(self, s, dim, norm);
}

/// Computes the N dimensional fast Fourier transform over given dimensions.
/// See https://pytorch.org/docs/master/fft.html#torch.fft.ifftn.
///
/// Example:
/// ```
/// auto t = torch::randn({128, 128}, dtype=kComplexDouble);
/// torch::fft::ifftn(t);
/// ```
inline Tensor ifftn(
    const Tensor& self,
    at::OptionalIntArrayRef s = c10::nullopt,
    at::OptionalIntArrayRef dim = c10::nullopt,
    c10::optional<c10::string_view> norm = c10::nullopt) {
  return torch::fft_ifftn(self, s, dim, norm);
}

/// Computes the 1 dimensional FFT of real input with onesided Hermitian output.
/// See https://pytorch.org/docs/master/fft.html#torch.fft.rfft.
///
/// Example:
/// ```
/// auto t = torch::randn(128);
/// auto T = torch::fft::rfft(t);
/// assert(T.is_complex() && T.numel() == 128 / 2 + 1);
/// ```
inline Tensor rfft(
    const Tensor& self,
    c10::optional<int64_t> n = c10::nullopt,
    int64_t dim = -1,
    c10::optional<c10::string_view> norm = c10::nullopt) {
  return torch::fft_rfft(self, n, dim, norm);
}

/// Computes the inverse of torch.fft.rfft
///
/// The input is a onesided Hermitian Fourier domain signal, with real-valued
/// output. See https://pytorch.org/docs/master/fft.html#torch.fft.irfft
///
/// Example:
/// ```
/// auto T = torch::randn(128 / 2 + 1, torch::kComplexDouble);
/// auto t = torch::fft::irfft(t, /*n=*/128);
/// assert(t.is_floating_point() && T.numel() == 128);
/// ```
inline Tensor irfft(
    const Tensor& self,
    c10::optional<int64_t> n = c10::nullopt,
    int64_t dim = -1,
    c10::optional<c10::string_view> norm = c10::nullopt) {
  return torch::fft_irfft(self, n, dim, norm);
}

/// Computes the 2-dimensional FFT of real input. Returns a onesided Hermitian
/// output. See https://pytorch.org/docs/master/fft.html#torch.fft.rfft2
///
/// Example:
/// ```
/// auto t = torch::randn({128, 128}, dtype=kDouble);
/// torch::fft::rfft2(t);
/// ```
inline Tensor rfft2(
    const Tensor& self,
    at::OptionalIntArrayRef s = c10::nullopt,
    IntArrayRef dim = {-2, -1},
    c10::optional<c10::string_view> norm = c10::nullopt) {
  return torch::fft_rfft2(self, s, dim, norm);
}

/// Computes the inverse of torch.fft.rfft2.
/// See https://pytorch.org/docs/master/fft.html#torch.fft.irfft2.
///
/// Example:
/// ```
/// auto t = torch::randn({128, 128}, dtype=kComplexDouble);
/// torch::fft::irfft2(t);
/// ```
inline Tensor irfft2(
    const Tensor& self,
    at::OptionalIntArrayRef s = c10::nullopt,
    IntArrayRef dim = {-2, -1},
    c10::optional<c10::string_view> norm = c10::nullopt) {
  return torch::fft_irfft2(self, s, dim, norm);
}

/// Computes the N dimensional FFT of real input with onesided Hermitian output.
/// See https://pytorch.org/docs/master/fft.html#torch.fft.rfftn
///
/// Example:
/// ```
/// auto t = torch::randn({128, 128}, dtype=kDouble);
/// torch::fft::rfftn(t);
/// ```
inline Tensor rfftn(
    const Tensor& self,
    at::OptionalIntArrayRef s = c10::nullopt,
    at::OptionalIntArrayRef dim = c10::nullopt,
    c10::optional<c10::string_view> norm = c10::nullopt) {
  return torch::fft_rfftn(self, s, dim, norm);
}

/// Computes the inverse of torch.fft.rfftn.
/// See https://pytorch.org/docs/master/fft.html#torch.fft.irfftn.
///
/// Example:
/// ```
/// auto t = torch::randn({128, 128}, dtype=kComplexDouble);
/// torch::fft::irfftn(t);
/// ```
inline Tensor irfftn(
    const Tensor& self,
    at::OptionalIntArrayRef s = c10::nullopt,
    at::OptionalIntArrayRef dim = c10::nullopt,
    c10::optional<c10::string_view> norm = c10::nullopt) {
  return torch::fft_irfftn(self, s, dim, norm);
}

/// Computes the 1 dimensional FFT of a onesided Hermitian signal
///
/// The input represents a Hermitian symmetric time domain signal. The returned
/// Fourier domain representation of such a signal is a real-valued. See
/// https://pytorch.org/docs/master/fft.html#torch.fft.hfft
///
/// Example:
/// ```
/// auto t = torch::randn(128 / 2 + 1, torch::kComplexDouble);
/// auto T = torch::fft::hfft(t, /*n=*/128);
/// assert(T.is_floating_point() && T.numel() == 128);
/// ```
inline Tensor hfft(
    const Tensor& self,
    c10::optional<int64_t> n = c10::nullopt,
    int64_t dim = -1,
    c10::optional<c10::string_view> norm = c10::nullopt) {
  return torch::fft_hfft(self, n, dim, norm);
}

/// Computes the inverse FFT of a real-valued Fourier domain signal.
///
/// The output is a onesided representation of the Hermitian symmetric time
/// domain signal. See https://pytorch.org/docs/master/fft.html#torch.fft.ihfft.
///
/// Example:
/// ```
/// auto T = torch::randn(128, torch::kDouble);
/// auto t = torch::fft::ihfft(T);
/// assert(t.is_complex() && T.numel() == 128 / 2 + 1);
/// ```
inline Tensor ihfft(
    const Tensor& self,
    c10::optional<int64_t> n = c10::nullopt,
    int64_t dim = -1,
    c10::optional<c10::string_view> norm = c10::nullopt) {
  return torch::fft_ihfft(self, n, dim, norm);
}

/// Computes the 2-dimensional FFT of a Hermitian symmetric input signal.
///
/// The input is a onesided representation of the Hermitian symmetric time
/// domain signal. See https://pytorch.org/docs/master/fft.html#torch.fft.hfft2.
///
/// Example:
/// ```
/// auto t = torch::randn({128, 65}, torch::kComplexDouble);
/// auto T = torch::fft::hfft2(t, /*s=*/{128, 128});
/// assert(T.is_floating_point() && T.numel() == 128 * 128);
/// ```
inline Tensor hfft2(
    const Tensor& self,
    at::OptionalIntArrayRef s = c10::nullopt,
    IntArrayRef dim = {-2, -1},
    c10::optional<c10::string_view> norm = c10::nullopt) {
  return torch::fft_hfft2(self, s, dim, norm);
}

/// Computes the 2-dimensional IFFT of a real input signal.
///
/// The output is a onesided representation of the Hermitian symmetric time
/// domain signal. See
/// https://pytorch.org/docs/master/fft.html#torch.fft.ihfft2.
///
/// Example:
/// ```
/// auto T = torch::randn({128, 128}, torch::kDouble);
/// auto t = torch::fft::hfft2(T);
/// assert(t.is_complex() && t.size(1) == 65);
/// ```
inline Tensor ihfft2(
    const Tensor& self,
    at::OptionalIntArrayRef s = c10::nullopt,
    IntArrayRef dim = {-2, -1},
    c10::optional<c10::string_view> norm = c10::nullopt) {
  return torch::fft_ihfft2(self, s, dim, norm);
}

/// Computes the N-dimensional FFT of a Hermitian symmetric input signal.
///
/// The input is a onesided representation of the Hermitian symmetric time
/// domain signal. See https://pytorch.org/docs/master/fft.html#torch.fft.hfftn.
///
/// Example:
/// ```
/// auto t = torch::randn({128, 65}, torch::kComplexDouble);
/// auto T = torch::fft::hfftn(t, /*s=*/{128, 128});
/// assert(T.is_floating_point() && T.numel() == 128 * 128);
/// ```
inline Tensor hfftn(
    const Tensor& self,
    at::OptionalIntArrayRef s = c10::nullopt,
    IntArrayRef dim = {-2, -1},
    c10::optional<c10::string_view> norm = c10::nullopt) {
  return torch::fft_hfftn(self, s, dim, norm);
}

/// Computes the N-dimensional IFFT of a real input signal.
///
/// The output is a onesided representation of the Hermitian symmetric time
/// domain signal. See
/// https://pytorch.org/docs/master/fft.html#torch.fft.ihfftn.
///
/// Example:
/// ```
/// auto T = torch::randn({128, 128}, torch::kDouble);
/// auto t = torch::fft::hfft2(T);
/// assert(t.is_complex() && t.size(1) == 65);
/// ```
inline Tensor ihfftn(
    const Tensor& self,
    at::OptionalIntArrayRef s = c10::nullopt,
    IntArrayRef dim = {-2, -1},
    c10::optional<c10::string_view> norm = c10::nullopt) {
  return torch::fft_ihfftn(self, s, dim, norm);
}

/// Computes the discrete Fourier Transform sample frequencies for a signal of
/// size n.
///
/// See https://pytorch.org/docs/master/fft.html#torch.fft.fftfreq
///
/// Example:
/// ```
/// auto frequencies = torch::fft::fftfreq(128, torch::kDouble);
/// ```
inline Tensor fftfreq(int64_t n, double d, const TensorOptions& options = {}) {
  return torch::fft_fftfreq(n, d, options);
}

inline Tensor fftfreq(int64_t n, const TensorOptions& options = {}) {
  return torch::fft_fftfreq(n, /*d=*/1.0, options);
}

/// Computes the sample frequencies for torch.fft.rfft with a signal of size n.
///
/// Like torch.fft.rfft, only the positive frequencies are included.
/// See https://pytorch.org/docs/master/fft.html#torch.fft.rfftfreq
///
/// Example:
/// ```
/// auto frequencies = torch::fft::rfftfreq(128, torch::kDouble);
/// ```
inline Tensor rfftfreq(int64_t n, double d, const TensorOptions& options) {
  return torch::fft_rfftfreq(n, d, options);
}

inline Tensor rfftfreq(int64_t n, const TensorOptions& options) {
  return torch::fft_rfftfreq(n, /*d=*/1.0, options);
}

/// Reorders n-dimensional FFT output to have negative frequency terms first, by
/// a torch.roll operation.
///
/// See https://pytorch.org/docs/master/fft.html#torch.fft.fftshift
///
/// Example:
/// ```
/// auto x = torch::randn({127, 4});
/// auto centred_fft = torch::fft::fftshift(torch::fft::fftn(x));
/// ```
inline Tensor fftshift(
    const Tensor& x,
    at::OptionalIntArrayRef dim = c10::nullopt) {
  return torch::fft_fftshift(x, dim);
}

/// Inverse of torch.fft.fftshift
///
/// See https://pytorch.org/docs/master/fft.html#torch.fft.ifftshift
///
/// Example:
/// ```
/// auto x = torch::randn({127, 4});
/// auto shift = torch::fft::fftshift(x)
/// auto unshift = torch::fft::ifftshift(shift);
/// assert(torch::allclose(x, unshift));
/// ```
inline Tensor ifftshift(
    const Tensor& x,
    at::OptionalIntArrayRef dim = c10::nullopt) {
  return torch::fft_ifftshift(x, dim);
}

} // namespace fft
} // namespace torch
