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

/// Computes the logarithmic derivative of the gamma function on input
/// See https://pytorch.org/docs/master/special.html#torch.special.psi
///
/// Example:
/// ```
/// auto t = torch::randn(128, dtype=kDouble);
/// torch::special::psi(t);
/// ```
inline Tensor psi(const Tensor& self) {
  return torch::special_psi(self);
}

inline Tensor& psi_out(Tensor& result, const Tensor& self) {
  return torch::special_psi_out(result, self);
}

/// Computes the logarithmic derivative of the gamma function on input
/// See https://pytorch.org/docs/master/special.html#torch.special.digamma
///
/// Example:
/// ```
/// auto t = torch::randn(128, dtype=kDouble);
/// torch::special::digamma(t);
/// ```
inline Tensor digamma(const Tensor& self) {
  return torch::special_digamma(self);
}

inline Tensor& digamma_out(Tensor& result, const Tensor& self) {
  return torch::special_digamma_out(result, self);
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

/// Computes the scaled complementary error function
/// See https://pytorch.org/docs/master/special.html#torch.special.erfcx.
///
/// Example:
/// ```
/// auto t = torch::randn(128, dtype=kDouble);
/// torch::special::erfcx(t);
/// ```
inline Tensor erfcx(const Tensor& self) {
  return torch::special_erfcx(self);
}

inline Tensor& erfcx_out(Tensor& result, const Tensor& self) {
  return torch::special_erfcx_out(result, self);
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

/// Computes the log of summed exponentials of each row of input in the given dimension dim
/// See https://pytorch.org/docs/master/special.html#torch.special.logsumexp.
///
/// Example:
/// ```
/// auto t = torch::randn(3, 3);
/// torch::special::logsumexp(t, 1);
/// ```
inline Tensor logsumexp(const Tensor& self, IntArrayRef dims, bool keepdim) {
  return torch::special_logsumexp(self, dims, keepdim);
}

inline Tensor& logsumexp_out(Tensor& result, const Tensor& self, IntArrayRef dims, bool keepdim) {
  return torch::special_logsumexp_out(result, self, dims, keepdim);
}

inline Tensor ndtri(const Tensor& self) {
  return torch::special_ndtri(self);
}

inline Tensor& ndtri_out(Tensor& result, const Tensor& self) {
  return torch::special_ndtri_out(result, self);
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

/// Computes Hurwitz Zeta function for inputs, elementwise
/// See https://pytorch.org/docs/master/special.html#torch.special.zeta.
///
/// Example:
/// ```
/// auto x = torch::randn(128, dtype=kDouble);
/// auto y = torch::randn(128, dtype=kDouble);
/// torch::special::zeta(x, y);
/// ```
inline Tensor zeta(const Tensor& self, const Tensor& other) {
  return torch::special_zeta(self, other);
}

inline Tensor zeta(const Scalar& self, const Tensor& other) {
  return torch::special_zeta(self, other);
}

inline Tensor zeta(const Tensor& self, const Scalar& other) {
  return torch::special_zeta(self, other);
}

inline Tensor& zeta_out(Tensor& result, const Tensor& self, const Tensor& other) {
  return torch::special_zeta_out(result, self, other);
}

inline Tensor& zeta_out(Tensor& result, const Scalar& self, const Tensor& other) {
  return torch::special_zeta_out(result, self, other);
}

inline Tensor& zeta_out(Tensor& result, const Tensor& self, const Scalar& other) {
  return torch::special_zeta_out(result, self, other);
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

inline Tensor& ndtr_out(Tensor& result, const Tensor& self) {
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

/// Computes log followed by softmax(x) of the input
/// See https://pytorch.org/docs/master/special.html#torch.special.log_softmax.
///
/// Example:
/// ```
/// auto t = torch::randn(128, 128, dtype=kDouble);
/// torch::special::log_softmax(t, 0);
/// ```
inline Tensor log_softmax(const Tensor& self, int64_t dim, c10::optional<ScalarType> dtype) {
  return torch::special_log_softmax(self, dim, dtype);
}

}} // torch::special
