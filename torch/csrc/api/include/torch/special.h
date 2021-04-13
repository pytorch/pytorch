#pragma once

#include <ATen/ATen.h>

namespace torch {
namespace special {

/// Computes the natural logarithm of the absolute value of the gamma function
/// See https://pytorch.org/docs/master/special.html#torch.special.gammaln.
///
/// Example:
/// ```
/// auto t = torch::randn(128, dtype=kDouble);
/// torch::special::gammaln(t);
/// ```
inline Tensor gammaln(const Tensor& self) {
  return torch::special_gammaln(self);
}

inline Tensor& gammaln_out(Tensor& result, const Tensor& self) {
  return torch::special_gammaln_out(result, self);
}

/// Computes entropy of input, elementwise
/// See https://pytorch.org/docs/master/special.html#torch.special.entr.
///
/// Example:
/// ```
/// auto t = torch::randn(128, dtype=kDouble);
/// torch::special::entr(t);
/// ```
inline Tensor entr(const Tensor& self) {
  return torch::special_entr(self);
}

inline Tensor& entr_out(Tensor& result, const Tensor& self) {
  return torch::special_entr_out(result, self);
}

/// Computes the error function
/// See https://pytorch.org/docs/master/special.html#torch.special.erf.
///
/// Example:
/// ```
/// auto t = torch::randn(128, dtype=kDouble);
/// torch::special::erf(t);
/// ```
inline Tensor erf(const Tensor& self) {
  return torch::special_erf(self);
}

inline Tensor& erf_out(Tensor& result, const Tensor& self) {
  return torch::special_erf_out(result, self);
}

/// Computes the complementary error function
/// See https://pytorch.org/docs/master/special.html#torch.special.erfc.
///
/// Example:
/// ```
/// auto t = torch::randn(128, dtype=kDouble);
/// torch::special::erfc(t);
/// ```
inline Tensor erfc(const Tensor& self) {
  return torch::special_erfc(self);
}

inline Tensor& erfc_out(Tensor& result, const Tensor& self) {
  return torch::special_erfc_out(result, self);
}

/// Computes the inverse error function
/// See https://pytorch.org/docs/master/special.html#torch.special.erfinv.
///
/// Example:
/// ```
/// auto t = torch::randn(128, dtype=kDouble);
/// torch::special::erfinv(t);
/// ```
inline Tensor erfinv(const Tensor& self) {
  return torch::special_erfinv(self);
}

inline Tensor& erfinv_out(Tensor& result, const Tensor& self) {
  return torch::special_erfinv_out(result, self);
}

/// Computes the logit of input, elementwise.
/// See https://pytorch.org/docs/master/special.html#torch.special.logit.
///
/// Example:
/// ```
/// auto t = torch::randn(128, dtype=kDouble);
/// torch::special::logit(t);
/// ```
inline Tensor logit(const Tensor& self) {
  return torch::special_logit(self);
}

inline Tensor& logit_out(Tensor& result, const Tensor& self) {
  return torch::special_logit_out(result, self);
}

/// Computes the expit (also known as the logistic sigmoid function) of input, elementwise
/// See https://pytorch.org/docs/master/special.html#torch.special.expit.
///
/// Example:
/// ```
/// auto t = torch::randn(128, dtype=kDouble);
/// torch::special::expit(t);
/// ```
inline Tensor expit(const Tensor& self) {
  return torch::special_expit(self);
}

inline Tensor& expit_out(Tensor& result, const Tensor& self) {
  return torch::special_expit_out(result, self);
}

/// Computes the base two exponential function of :attr:`input`, elementwise
/// See https://pytorch.org/docs/master/special.html#torch.special.exp2.
///
/// Example:
/// ```
/// auto t = torch::randn(128, dtype=kDouble);
/// torch::special::exp2(t);
/// ```
inline Tensor exp2(const Tensor& self) {
  return torch::special_exp2(self);
}

inline Tensor& exp2_out(Tensor& result, const Tensor& self) {
  return torch::special_exp2_out(result, self);
}

/// Computes the exponential of the elements minus 1, elementwise
/// See https://pytorch.org/docs/master/special.html#torch.special.expm1.
///
/// Example:
/// ```
/// auto t = torch::randn(128, dtype=kDouble);
/// torch::special::expm1(t);
/// ```
inline Tensor expm1(const Tensor& self) {
  return torch::special_expm1(self);
}

inline Tensor& expm1_out(Tensor& result, const Tensor& self) {
  return torch::special_expm1_out(result, self);
}

/// Computes the sinc of input, elementwise
/// See https://pytorch.org/docs/master/special.html#torch.special.sinc.
///
/// Example:
/// ```
/// auto t = torch::randn(128, dtype=kDouble);
/// torch::special::sinc(t);
/// ```
inline Tensor sinc(const Tensor& self) {
  return torch::special_sinc(self);
}

inline Tensor& sinc_out(Tensor& result, const Tensor& self) {
  return torch::special_sinc_out(result, self);
}

/// Rounds the elements of the input
/// See https://pytorch.org/docs/master/special.html#torch.special.round.
///
/// Example:
/// ```
/// auto t = torch::randn(128, dtype=kDouble);
/// torch::special::round(t);
/// ```
inline Tensor round(const Tensor& self) {
  return torch::special_round(self);
}

inline Tensor& round_out(Tensor& result, const Tensor& self) {
  return torch::special_round_out(result, self);
}

/// Computes log(1 + x) of the input, elementwise
/// See https://pytorch.org/docs/master/special.html#torch.special.log1p.
///
/// Example:
/// ```
/// auto t = torch::randn(128, dtype=kDouble);
/// torch::special::log1p(t);
/// ```
inline Tensor log1p(const Tensor& self) {
  return torch::special_log1p(self);
}

inline Tensor& log1p_out(Tensor& result, const Tensor& self) {
  return torch::special_log1p_out(result, self);
}

}} // torch::special
