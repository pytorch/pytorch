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

/// Computes x * log1p(y) for inputs, elementwise
/// See https://pytorch.org/docs/master/special.html#torch.special.xlog1py.
///
/// Example:
/// ```
/// auto x = torch::randn(128, dtype=kDouble);
/// auto y = torch::randn(128, dtype=kDouble);
/// torch::special::xlog1py(x, y);
/// ```
inline Tensor xlog1py(const Tensor& self, const Tensor& other) {
  return torch::special_xlog1py(self, other);
}

inline Tensor xlog1py(const Scalar& self, const Tensor& other) {
  return torch::special_xlog1py(self, other);
}

inline Tensor xlog1py(const Tensor& self, const Scalar& other) {
  return torch::special_xlog1py(self, other);
}

inline Tensor& xlog1py_out(Tensor& result, const Tensor& self, const Tensor& other) {
  return torch::special_xlog1py_out(result, self, other);
}

inline Tensor& xlog1py_out(Tensor& result, const Scalar& self, const Tensor& other) {
  return torch::special_xlog1py_out(result, self, other);
}

inline Tensor& xlog1py_out(Tensor& result, const Tensor& self, const Scalar& other) {
  return torch::special_xlog1py_out(result, self, other);
}

/// Computes the zeroth order modified Bessel function of the first kind of input, elementwise
/// See https://pytorch.org/docs/master/special.html#torch.special.i0
///
/// Example:
/// ```
/// auto t = torch::randn(128, dtype=kDouble);
/// torch::special::i0(t);
/// ```
inline Tensor i0(const Tensor& self) {
  return torch::special_i0(self);
}

inline Tensor& i0_out(Tensor& result, const Tensor& self) {
  return torch::special_i0_out(result, self);
}

/// Computes the area under the standard Gaussian probability density function,
/// integrated from minus infinity to :attr:`input`, elementwise
/// See https://pytorch.org/docs/master/special.html#torch.special.ndtr
///
/// Example:
/// ```
/// auto t = torch::randn(128, dtype=kDouble);
/// torch::special::ndtr(t);
/// ```
inline Tensor ndtr(const Tensor& self) {
  return torch::special_ndtr(self);
}

inline const Tensor& ndtr_out(const Tensor& result, const Tensor& self) {
  return torch::special_ndtr_out(result, self);
}

/// Computes the exponentially scaled zeroth order modified Bessel function of the first kind
/// See https://pytorch.org/docs/master/special.html#torch.special.i0e.
///
/// Example:
/// ```
/// auto t = torch::randn(128, dtype=kDouble);
/// torch::special::i0e(t);
/// ```
inline Tensor i0e(const Tensor& self) {
  return torch::special_i0e(self);
}

inline Tensor& i0e_out(Tensor& result, const Tensor& self) {
  return torch::special_i0e_out(result, self);
}

/// Computes the first order modified Bessel function of the first kind
/// See https://pytorch.org/docs/master/special.html#torch.special.i1.
///
/// Example:
/// ```
/// auto t = torch::randn(128, dtype=kDouble);
/// torch::special::i1(t);
/// ```
inline Tensor i1(const Tensor& self) {
  return torch::special_i1(self);
}

inline Tensor& i1_out(Tensor& result, const Tensor& self) {
  return torch::special_i1_out(result, self);
}

/// Computes the exponentially scaled first order modified Bessel function of the first kind
/// See https://pytorch.org/docs/master/special.html#torch.special.i1e.
///
/// Example:
/// ```
/// auto t = torch::randn(128, dtype=kDouble);
/// torch::special::i1e(t);
/// ```
inline Tensor i1e(const Tensor& self) {
  return torch::special_i1e(self);
}

inline Tensor& i1e_out(Tensor& result, const Tensor& self) {
  return torch::special_i1e_out(result, self);
}

}} // torch::special
