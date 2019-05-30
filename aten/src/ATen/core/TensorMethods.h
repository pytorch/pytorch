#pragma once

#include <c10/core/Scalar.h>
#include <c10/core/MemoryFormat.h>
#include <c10/macros/Macros.h>
#include <c10/core/TensorOptions.h>
#include <ATen/core/DeprecatedTypeProperties.h>
#include <ATen/core/ATenDispatch.h>

namespace at {

inline Tensor Tensor::toType(const DeprecatedTypeProperties & t, bool non_blocking) const {
  if(type() == t)
    return *this;
  return to(
      at::device(t.device_type()).layout(t.layout()).dtype(t.scalarType()),
      non_blocking,
      /*copy=*/ true);
}

inline Tensor Tensor::cpu() const {
  return toType(type().cpu());
}

inline Tensor Tensor::cuda() const {
  return toType(type().cuda());
}

inline Tensor Tensor::hip() const {
  return toType(type().hip());
}

inline Tensor Tensor::toType(ScalarType t) const {
  return toType(type().toScalarType(t));
}

inline Tensor Tensor::toBackend(Backend b) const {
  return toType(type().toBackend(b));
}

inline TensorOptions Tensor::options() const {
  return TensorOptions().dtype(dtype())
                        .device(device())
                        .layout(layout())
                        .is_variable(is_variable());
}

inline void Tensor::backward(
    c10::optional<Tensor> gradient,
    bool keep_graph,
    bool create_graph) {
  dispatch_type().backward(*this, std::move(gradient), keep_graph, create_graph);
}

inline void Tensor::set_data(Tensor new_data) {
  dispatch_type().set_data(*this, new_data);
}

// all static inline to allow for inlining of the non-dynamic part of dispatch
inline Tensor Tensor::abs() const {
    return find_op<Tensor, const Tensor &>(tensorTypeIdToBackend(type_id()), is_variable(), 30)(*this);
}
inline Tensor & Tensor::abs_() {
    return find_op<Tensor &, Tensor &>(tensorTypeIdToBackend(type_id()), is_variable(), 31)(*this);
}
inline Tensor Tensor::acos() const {
    return find_op<Tensor, const Tensor &>(tensorTypeIdToBackend(type_id()), is_variable(), 33)(*this);
}
inline Tensor & Tensor::acos_() {
    return find_op<Tensor &, Tensor &>(tensorTypeIdToBackend(type_id()), is_variable(), 34)(*this);
}
inline Tensor Tensor::add(const Tensor & other, Scalar alpha) const {
    return find_op<Tensor, const Tensor &, const Tensor &, Scalar>(tensorTypeIdToBackend(type_id()), is_variable(), 39)(*this, other, alpha);
}
inline Tensor & Tensor::add_(const Tensor & other, Scalar alpha) {
    return find_op<Tensor &, Tensor &, const Tensor &, Scalar>(tensorTypeIdToBackend(type_id()), is_variable(), 40)(*this, other, alpha);
}
inline Tensor Tensor::add(Scalar other, Scalar alpha) const {
    return find_op<Tensor, const Tensor &, Scalar, Scalar>(tensorTypeIdToBackend(type_id()), is_variable(), 42)(*this, other, alpha);
}
inline Tensor & Tensor::add_(Scalar other, Scalar alpha) {
    return find_op<Tensor &, Tensor &, Scalar, Scalar>(tensorTypeIdToBackend(type_id()), is_variable(), 43)(*this, other, alpha);
}
inline Tensor Tensor::addmv(const Tensor & mat, const Tensor & vec, Scalar beta, Scalar alpha) const {
    return find_op<Tensor, const Tensor &, const Tensor &, const Tensor &, Scalar, Scalar>(tensorTypeIdToBackend(type_id()), is_variable(), 44)(*this, mat, vec, beta, alpha);
}
inline Tensor & Tensor::addmv_(const Tensor & mat, const Tensor & vec, Scalar beta, Scalar alpha) {
    return find_op<Tensor &, Tensor &, const Tensor &, const Tensor &, Scalar, Scalar>(tensorTypeIdToBackend(type_id()), is_variable(), 45)(*this, mat, vec, beta, alpha);
}
inline Tensor Tensor::addr(const Tensor & vec1, const Tensor & vec2, Scalar beta, Scalar alpha) const {
    return find_op<Tensor, const Tensor &, const Tensor &, const Tensor &, Scalar, Scalar>(tensorTypeIdToBackend(type_id()), is_variable(), 47)(*this, vec1, vec2, beta, alpha);
}
inline Tensor & Tensor::addr_(const Tensor & vec1, const Tensor & vec2, Scalar beta, Scalar alpha) {
    return find_op<Tensor &, Tensor &, const Tensor &, const Tensor &, Scalar, Scalar>(tensorTypeIdToBackend(type_id()), is_variable(), 48)(*this, vec1, vec2, beta, alpha);
}
inline Tensor Tensor::all(int64_t dim, bool keepdim) const {
    return find_op<Tensor, const Tensor &, int64_t, bool>(tensorTypeIdToBackend(type_id()), is_variable(), 52)(*this, dim, keepdim);
}
inline bool Tensor::allclose(const Tensor & other, double rtol, double atol, bool equal_nan) const {
    return find_op<bool, const Tensor &, const Tensor &, double, double, bool>(tensorTypeIdToBackend(type_id()), is_variable(), 54)(*this, other, rtol, atol, equal_nan);
}
inline Tensor Tensor::any(int64_t dim, bool keepdim) const {
    return find_op<Tensor, const Tensor &, int64_t, bool>(tensorTypeIdToBackend(type_id()), is_variable(), 55)(*this, dim, keepdim);
}
inline Tensor Tensor::argmax(c10::optional<int64_t> dim, bool keepdim) const {
    return find_op<Tensor, const Tensor &, c10::optional<int64_t>, bool>(tensorTypeIdToBackend(type_id()), is_variable(), 63)(*this, dim, keepdim);
}
inline Tensor Tensor::argmin(c10::optional<int64_t> dim, bool keepdim) const {
    return find_op<Tensor, const Tensor &, c10::optional<int64_t>, bool>(tensorTypeIdToBackend(type_id()), is_variable(), 64)(*this, dim, keepdim);
}
inline Tensor Tensor::as_strided(IntArrayRef size, IntArrayRef stride, c10::optional<int64_t> storage_offset) const {
    return find_op<Tensor, const Tensor &, IntArrayRef, IntArrayRef, c10::optional<int64_t>>(tensorTypeIdToBackend(type_id()), is_variable(), 65)(*this, size, stride, storage_offset);
}
inline Tensor & Tensor::as_strided_(IntArrayRef size, IntArrayRef stride, c10::optional<int64_t> storage_offset) {
    return find_op<Tensor &, Tensor &, IntArrayRef, IntArrayRef, c10::optional<int64_t>>(tensorTypeIdToBackend(type_id()), is_variable(), 66)(*this, size, stride, storage_offset);
}
inline Tensor Tensor::asin() const {
    return find_op<Tensor, const Tensor &>(tensorTypeIdToBackend(type_id()), is_variable(), 67)(*this);
}
inline Tensor & Tensor::asin_() {
    return find_op<Tensor &, Tensor &>(tensorTypeIdToBackend(type_id()), is_variable(), 68)(*this);
}
inline Tensor Tensor::atan() const {
    return find_op<Tensor, const Tensor &>(tensorTypeIdToBackend(type_id()), is_variable(), 70)(*this);
}
inline Tensor & Tensor::atan_() {
    return find_op<Tensor &, Tensor &>(tensorTypeIdToBackend(type_id()), is_variable(), 71)(*this);
}
inline Tensor Tensor::baddbmm(const Tensor & batch1, const Tensor & batch2, Scalar beta, Scalar alpha) const {
    return find_op<Tensor, const Tensor &, const Tensor &, const Tensor &, Scalar, Scalar>(tensorTypeIdToBackend(type_id()), is_variable(), 73)(*this, batch1, batch2, beta, alpha);
}
inline Tensor & Tensor::baddbmm_(const Tensor & batch1, const Tensor & batch2, Scalar beta, Scalar alpha) {
    return find_op<Tensor &, Tensor &, const Tensor &, const Tensor &, Scalar, Scalar>(tensorTypeIdToBackend(type_id()), is_variable(), 74)(*this, batch1, batch2, beta, alpha);
}
inline Tensor Tensor::bernoulli(Generator * generator) const {
    return find_op<Tensor, const Tensor &, Generator *>(tensorTypeIdToBackend(type_id()), is_variable(), 82)(*this, generator);
}
inline Tensor & Tensor::bernoulli_(const Tensor & p, Generator * generator) {
    return find_op<Tensor &, Tensor &, const Tensor &, Generator *>(tensorTypeIdToBackend(type_id()), is_variable(), 84)(*this, p, generator);
}
inline Tensor & Tensor::bernoulli_(double p, Generator * generator) {
    return find_op<Tensor &, Tensor &, double, Generator *>(tensorTypeIdToBackend(type_id()), is_variable(), 85)(*this, p, generator);
}
inline Tensor Tensor::bernoulli(double p, Generator * generator) const {
    return find_op<Tensor, const Tensor &, double, Generator *>(tensorTypeIdToBackend(type_id()), is_variable(), 86)(*this, p, generator);
}
inline Tensor Tensor::bincount(const Tensor & weights, int64_t minlength) const {
    return find_op<Tensor, const Tensor &, const Tensor &, int64_t>(tensorTypeIdToBackend(type_id()), is_variable(), 90)(*this, weights, minlength);
}
inline Tensor Tensor::bmm(const Tensor & mat2) const {
    return find_op<Tensor, const Tensor &, const Tensor &>(tensorTypeIdToBackend(type_id()), is_variable(), 93)(*this, mat2);
}
inline Tensor Tensor::ceil() const {
    return find_op<Tensor, const Tensor &>(tensorTypeIdToBackend(type_id()), is_variable(), 98)(*this);
}
inline Tensor & Tensor::ceil_() {
    return find_op<Tensor &, Tensor &>(tensorTypeIdToBackend(type_id()), is_variable(), 99)(*this);
}
inline std::vector<Tensor> Tensor::chunk(int64_t chunks, int64_t dim) const {
    return find_op<std::vector<Tensor>, const Tensor &, int64_t, int64_t>(tensorTypeIdToBackend(type_id()), is_variable(), 102)(*this, chunks, dim);
}
inline Tensor Tensor::clamp(c10::optional<Scalar> min, c10::optional<Scalar> max) const {
    return find_op<Tensor, const Tensor &, c10::optional<Scalar>, c10::optional<Scalar>>(tensorTypeIdToBackend(type_id()), is_variable(), 103)(*this, min, max);
}
inline Tensor & Tensor::clamp_(c10::optional<Scalar> min, c10::optional<Scalar> max) {
    return find_op<Tensor &, Tensor &, c10::optional<Scalar>, c10::optional<Scalar>>(tensorTypeIdToBackend(type_id()), is_variable(), 104)(*this, min, max);
}
inline Tensor Tensor::clamp_max(Scalar max) const {
    return find_op<Tensor, const Tensor &, Scalar>(tensorTypeIdToBackend(type_id()), is_variable(), 106)(*this, max);
}
inline Tensor & Tensor::clamp_max_(Scalar max) {
    return find_op<Tensor &, Tensor &, Scalar>(tensorTypeIdToBackend(type_id()), is_variable(), 107)(*this, max);
}
inline Tensor Tensor::clamp_min(Scalar min) const {
    return find_op<Tensor, const Tensor &, Scalar>(tensorTypeIdToBackend(type_id()), is_variable(), 109)(*this, min);
}
inline Tensor & Tensor::clamp_min_(Scalar min) {
    return find_op<Tensor &, Tensor &, Scalar>(tensorTypeIdToBackend(type_id()), is_variable(), 110)(*this, min);
}
inline Tensor Tensor::contiguous(MemoryFormat memory_format) const {
    return find_op<Tensor, const Tensor &, MemoryFormat>(tensorTypeIdToBackend(type_id()), is_variable(), 114)(*this, memory_format);
}
inline Tensor & Tensor::copy_(const Tensor & src, bool non_blocking) {
    return find_op<Tensor &, Tensor &, const Tensor &, bool>(tensorTypeIdToBackend(type_id()), is_variable(), 127)(*this, src, non_blocking);
}
inline Tensor Tensor::cos() const {
    return find_op<Tensor, const Tensor &>(tensorTypeIdToBackend(type_id()), is_variable(), 129)(*this);
}
inline Tensor & Tensor::cos_() {
    return find_op<Tensor &, Tensor &>(tensorTypeIdToBackend(type_id()), is_variable(), 130)(*this);
}
inline Tensor Tensor::cosh() const {
    return find_op<Tensor, const Tensor &>(tensorTypeIdToBackend(type_id()), is_variable(), 132)(*this);
}
inline Tensor & Tensor::cosh_() {
    return find_op<Tensor &, Tensor &>(tensorTypeIdToBackend(type_id()), is_variable(), 133)(*this);
}
inline Tensor Tensor::cumsum(int64_t dim, ScalarType dtype) const {
    return find_op<Tensor, const Tensor &, int64_t, ScalarType>(tensorTypeIdToBackend(type_id()), is_variable(), 152)(*this, dim, dtype);
}
inline Tensor Tensor::cumsum(int64_t dim) const {
    return find_op<Tensor, const Tensor &, int64_t>(tensorTypeIdToBackend(type_id()), is_variable(), 153)(*this, dim);
}
inline Tensor Tensor::cumprod(int64_t dim, ScalarType dtype) const {
    return find_op<Tensor, const Tensor &, int64_t, ScalarType>(tensorTypeIdToBackend(type_id()), is_variable(), 156)(*this, dim, dtype);
}
inline Tensor Tensor::cumprod(int64_t dim) const {
    return find_op<Tensor, const Tensor &, int64_t>(tensorTypeIdToBackend(type_id()), is_variable(), 157)(*this, dim);
}
inline Tensor Tensor::det() const {
    return find_op<Tensor, const Tensor &>(tensorTypeIdToBackend(type_id()), is_variable(), 164)(*this);
}
inline Tensor Tensor::diag_embed(int64_t offset, int64_t dim1, int64_t dim2) const {
    return find_op<Tensor, const Tensor &, int64_t, int64_t, int64_t>(tensorTypeIdToBackend(type_id()), is_variable(), 165)(*this, offset, dim1, dim2);
}
inline Tensor Tensor::diagflat(int64_t offset) const {
    return find_op<Tensor, const Tensor &, int64_t>(tensorTypeIdToBackend(type_id()), is_variable(), 166)(*this, offset);
}
inline Tensor Tensor::diagonal(int64_t offset, int64_t dim1, int64_t dim2) const {
    return find_op<Tensor, const Tensor &, int64_t, int64_t, int64_t>(tensorTypeIdToBackend(type_id()), is_variable(), 167)(*this, offset, dim1, dim2);
}
inline Tensor Tensor::div(const Tensor & other) const {
    return find_op<Tensor, const Tensor &, const Tensor &>(tensorTypeIdToBackend(type_id()), is_variable(), 168)(*this, other);
}
inline Tensor & Tensor::div_(const Tensor & other) {
    return find_op<Tensor &, Tensor &, const Tensor &>(tensorTypeIdToBackend(type_id()), is_variable(), 169)(*this, other);
}
inline Tensor Tensor::div(Scalar other) const {
    return find_op<Tensor, const Tensor &, Scalar>(tensorTypeIdToBackend(type_id()), is_variable(), 171)(*this, other);
}
inline Tensor & Tensor::div_(Scalar other) {
    return find_op<Tensor &, Tensor &, Scalar>(tensorTypeIdToBackend(type_id()), is_variable(), 172)(*this, other);
}
inline Tensor Tensor::dot(const Tensor & tensor) const {
    return find_op<Tensor, const Tensor &, const Tensor &>(tensorTypeIdToBackend(type_id()), is_variable(), 173)(*this, tensor);
}
inline Tensor & Tensor::resize_(IntArrayRef size) {
    return find_op<Tensor &, Tensor &, IntArrayRef>(tensorTypeIdToBackend(type_id()), is_variable(), 189)(*this, size);
}
inline Tensor Tensor::erf() const {
    return find_op<Tensor, const Tensor &>(tensorTypeIdToBackend(type_id()), is_variable(), 194)(*this);
}
inline Tensor & Tensor::erf_() {
    return find_op<Tensor &, Tensor &>(tensorTypeIdToBackend(type_id()), is_variable(), 195)(*this);
}
inline Tensor Tensor::erfc() const {
    return find_op<Tensor, const Tensor &>(tensorTypeIdToBackend(type_id()), is_variable(), 197)(*this);
}
inline Tensor & Tensor::erfc_() {
    return find_op<Tensor &, Tensor &>(tensorTypeIdToBackend(type_id()), is_variable(), 198)(*this);
}
inline Tensor Tensor::exp() const {
    return find_op<Tensor, const Tensor &>(tensorTypeIdToBackend(type_id()), is_variable(), 200)(*this);
}
inline Tensor & Tensor::exp_() {
    return find_op<Tensor &, Tensor &>(tensorTypeIdToBackend(type_id()), is_variable(), 201)(*this);
}
inline Tensor Tensor::expm1() const {
    return find_op<Tensor, const Tensor &>(tensorTypeIdToBackend(type_id()), is_variable(), 203)(*this);
}
inline Tensor & Tensor::expm1_() {
    return find_op<Tensor &, Tensor &>(tensorTypeIdToBackend(type_id()), is_variable(), 204)(*this);
}
inline Tensor Tensor::expand(IntArrayRef size, bool implicit) const {
    return find_op<Tensor, const Tensor &, IntArrayRef, bool>(tensorTypeIdToBackend(type_id()), is_variable(), 206)(*this, size, implicit);
}
inline Tensor Tensor::expand_as(const Tensor & other) const {
    return find_op<Tensor, const Tensor &, const Tensor &>(tensorTypeIdToBackend(type_id()), is_variable(), 207)(*this, other);
}
inline Tensor Tensor::flatten(int64_t start_dim, int64_t end_dim) const {
    return find_op<Tensor, const Tensor &, int64_t, int64_t>(tensorTypeIdToBackend(type_id()), is_variable(), 212)(*this, start_dim, end_dim);
}
inline Tensor & Tensor::fill_(Scalar value) {
    return find_op<Tensor &, Tensor &, Scalar>(tensorTypeIdToBackend(type_id()), is_variable(), 213)(*this, value);
}
inline Tensor & Tensor::fill_(const Tensor & value) {
    return find_op<Tensor &, Tensor &, const Tensor &>(tensorTypeIdToBackend(type_id()), is_variable(), 214)(*this, value);
}
inline Tensor Tensor::floor() const {
    return find_op<Tensor, const Tensor &>(tensorTypeIdToBackend(type_id()), is_variable(), 215)(*this);
}
inline Tensor & Tensor::floor_() {
    return find_op<Tensor &, Tensor &>(tensorTypeIdToBackend(type_id()), is_variable(), 216)(*this);
}
inline Tensor Tensor::frac() const {
    return find_op<Tensor, const Tensor &>(tensorTypeIdToBackend(type_id()), is_variable(), 218)(*this);
}
inline Tensor & Tensor::frac_() {
    return find_op<Tensor &, Tensor &>(tensorTypeIdToBackend(type_id()), is_variable(), 219)(*this);
}
inline Tensor Tensor::ger(const Tensor & vec2) const {
    return find_op<Tensor, const Tensor &, const Tensor &>(tensorTypeIdToBackend(type_id()), is_variable(), 238)(*this, vec2);
}
inline Tensor Tensor::fft(int64_t signal_ndim, bool normalized) const {
    return find_op<Tensor, const Tensor &, int64_t, bool>(tensorTypeIdToBackend(type_id()), is_variable(), 241)(*this, signal_ndim, normalized);
}
inline Tensor Tensor::ifft(int64_t signal_ndim, bool normalized) const {
    return find_op<Tensor, const Tensor &, int64_t, bool>(tensorTypeIdToBackend(type_id()), is_variable(), 242)(*this, signal_ndim, normalized);
}
inline Tensor Tensor::rfft(int64_t signal_ndim, bool normalized, bool onesided) const {
    return find_op<Tensor, const Tensor &, int64_t, bool, bool>(tensorTypeIdToBackend(type_id()), is_variable(), 243)(*this, signal_ndim, normalized, onesided);
}
inline Tensor Tensor::irfft(int64_t signal_ndim, bool normalized, bool onesided, IntArrayRef signal_sizes) const {
    return find_op<Tensor, const Tensor &, int64_t, bool, bool, IntArrayRef>(tensorTypeIdToBackend(type_id()), is_variable(), 244)(*this, signal_ndim, normalized, onesided, signal_sizes);
}
inline Tensor Tensor::index(TensorList indices) const {
    return find_op<Tensor, const Tensor &, TensorList>(tensorTypeIdToBackend(type_id()), is_variable(), 250)(*this, indices);
}
inline Tensor & Tensor::index_copy_(int64_t dim, const Tensor & index, const Tensor & source) {
    return find_op<Tensor &, Tensor &, int64_t, const Tensor &, const Tensor &>(tensorTypeIdToBackend(type_id()), is_variable(), 251)(*this, dim, index, source);
}
inline Tensor Tensor::index_copy(int64_t dim, const Tensor & index, const Tensor & source) const {
    return find_op<Tensor, const Tensor &, int64_t, const Tensor &, const Tensor &>(tensorTypeIdToBackend(type_id()), is_variable(), 252)(*this, dim, index, source);
}
inline Tensor & Tensor::index_put_(TensorList indices, const Tensor & values, bool accumulate) {
    return find_op<Tensor &, Tensor &, TensorList, const Tensor &, bool>(tensorTypeIdToBackend(type_id()), is_variable(), 253)(*this, indices, values, accumulate);
}
inline Tensor Tensor::index_put(TensorList indices, const Tensor & values, bool accumulate) const {
    return find_op<Tensor, const Tensor &, TensorList, const Tensor &, bool>(tensorTypeIdToBackend(type_id()), is_variable(), 254)(*this, indices, values, accumulate);
}
inline Tensor Tensor::inverse() const {
    return find_op<Tensor, const Tensor &>(tensorTypeIdToBackend(type_id()), is_variable(), 256)(*this);
}
inline Tensor Tensor::isclose(const Tensor & other, double rtol, double atol, bool equal_nan) const {
    return find_op<Tensor, const Tensor &, const Tensor &, double, double, bool>(tensorTypeIdToBackend(type_id()), is_variable(), 259)(*this, other, rtol, atol, equal_nan);
}
inline bool Tensor::is_distributed() const {
    return find_op<bool, const Tensor &>(tensorTypeIdToBackend(type_id()), is_variable(), 261)(*this);
}
inline bool Tensor::is_floating_point() const {
    return find_op<bool, const Tensor &>(tensorTypeIdToBackend(type_id()), is_variable(), 262)(*this);
}
inline bool Tensor::is_complex() const {
    return find_op<bool, const Tensor &>(tensorTypeIdToBackend(type_id()), is_variable(), 263)(*this);
}
inline bool Tensor::is_nonzero() const {
    return find_op<bool, const Tensor &>(tensorTypeIdToBackend(type_id()), is_variable(), 264)(*this);
}
inline bool Tensor::is_same_size(const Tensor & other) const {
    return find_op<bool, const Tensor &, const Tensor &>(tensorTypeIdToBackend(type_id()), is_variable(), 265)(*this, other);
}
inline bool Tensor::is_signed() const {
    return find_op<bool, const Tensor &>(tensorTypeIdToBackend(type_id()), is_variable(), 266)(*this);
}
inline std::tuple<Tensor,Tensor> Tensor::kthvalue(int64_t k, int64_t dim, bool keepdim) const {
    return find_op<std::tuple<Tensor,Tensor>, const Tensor &, int64_t, int64_t, bool>(tensorTypeIdToBackend(type_id()), is_variable(), 269)(*this, k, dim, keepdim);
}
inline Tensor Tensor::log() const {
    return find_op<Tensor, const Tensor &>(tensorTypeIdToBackend(type_id()), is_variable(), 280)(*this);
}
inline Tensor & Tensor::log_() {
    return find_op<Tensor &, Tensor &>(tensorTypeIdToBackend(type_id()), is_variable(), 281)(*this);
}
inline Tensor Tensor::log10() const {
    return find_op<Tensor, const Tensor &>(tensorTypeIdToBackend(type_id()), is_variable(), 283)(*this);
}
inline Tensor & Tensor::log10_() {
    return find_op<Tensor &, Tensor &>(tensorTypeIdToBackend(type_id()), is_variable(), 284)(*this);
}
inline Tensor Tensor::log1p() const {
    return find_op<Tensor, const Tensor &>(tensorTypeIdToBackend(type_id()), is_variable(), 286)(*this);
}
inline Tensor & Tensor::log1p_() {
    return find_op<Tensor &, Tensor &>(tensorTypeIdToBackend(type_id()), is_variable(), 287)(*this);
}
inline Tensor Tensor::log2() const {
    return find_op<Tensor, const Tensor &>(tensorTypeIdToBackend(type_id()), is_variable(), 289)(*this);
}
inline Tensor & Tensor::log2_() {
    return find_op<Tensor &, Tensor &>(tensorTypeIdToBackend(type_id()), is_variable(), 290)(*this);
}
inline Tensor Tensor::logdet() const {
    return find_op<Tensor, const Tensor &>(tensorTypeIdToBackend(type_id()), is_variable(), 292)(*this);
}
inline Tensor Tensor::log_softmax(int64_t dim, ScalarType dtype) const {
    return find_op<Tensor, const Tensor &, int64_t, ScalarType>(tensorTypeIdToBackend(type_id()), is_variable(), 295)(*this, dim, dtype);
}
inline Tensor Tensor::log_softmax(int64_t dim) const {
    return find_op<Tensor, const Tensor &, int64_t>(tensorTypeIdToBackend(type_id()), is_variable(), 296)(*this, dim);
}
inline Tensor Tensor::logsumexp(IntArrayRef dim, bool keepdim) const {
    return find_op<Tensor, const Tensor &, IntArrayRef, bool>(tensorTypeIdToBackend(type_id()), is_variable(), 299)(*this, dim, keepdim);
}
inline Tensor Tensor::matmul(const Tensor & other) const {
    return find_op<Tensor, const Tensor &, const Tensor &>(tensorTypeIdToBackend(type_id()), is_variable(), 302)(*this, other);
}
inline Tensor Tensor::matrix_power(int64_t n) const {
    return find_op<Tensor, const Tensor &, int64_t>(tensorTypeIdToBackend(type_id()), is_variable(), 306)(*this, n);
}
inline std::tuple<Tensor,Tensor> Tensor::max(int64_t dim, bool keepdim) const {
    return find_op<std::tuple<Tensor,Tensor>, const Tensor &, int64_t, bool>(tensorTypeIdToBackend(type_id()), is_variable(), 307)(*this, dim, keepdim);
}
inline Tensor Tensor::max_values(IntArrayRef dim, bool keepdim) const {
    return find_op<Tensor, const Tensor &, IntArrayRef, bool>(tensorTypeIdToBackend(type_id()), is_variable(), 309)(*this, dim, keepdim);
}
inline Tensor Tensor::mean(ScalarType dtype) const {
    return find_op<Tensor, const Tensor &, ScalarType>(tensorTypeIdToBackend(type_id()), is_variable(), 315)(*this, dtype);
}
inline Tensor Tensor::mean() const {
    return find_op<Tensor, const Tensor &>(tensorTypeIdToBackend(type_id()), is_variable(), 316)(*this);
}
inline Tensor Tensor::mean(IntArrayRef dim, bool keepdim, ScalarType dtype) const {
    return find_op<Tensor, const Tensor &, IntArrayRef, bool, ScalarType>(tensorTypeIdToBackend(type_id()), is_variable(), 317)(*this, dim, keepdim, dtype);
}
inline Tensor Tensor::mean(IntArrayRef dim, bool keepdim) const {
    return find_op<Tensor, const Tensor &, IntArrayRef, bool>(tensorTypeIdToBackend(type_id()), is_variable(), 318)(*this, dim, keepdim);
}
inline Tensor Tensor::mean(IntArrayRef dim, ScalarType dtype) const {
    return find_op<Tensor, const Tensor &, IntArrayRef, ScalarType>(tensorTypeIdToBackend(type_id()), is_variable(), 319)(*this, dim, dtype);
}
inline std::tuple<Tensor,Tensor> Tensor::median(int64_t dim, bool keepdim) const {
    return find_op<std::tuple<Tensor,Tensor>, const Tensor &, int64_t, bool>(tensorTypeIdToBackend(type_id()), is_variable(), 323)(*this, dim, keepdim);
}
inline std::tuple<Tensor,Tensor> Tensor::min(int64_t dim, bool keepdim) const {
    return find_op<std::tuple<Tensor,Tensor>, const Tensor &, int64_t, bool>(tensorTypeIdToBackend(type_id()), is_variable(), 325)(*this, dim, keepdim);
}
inline Tensor Tensor::min_values(IntArrayRef dim, bool keepdim) const {
    return find_op<Tensor, const Tensor &, IntArrayRef, bool>(tensorTypeIdToBackend(type_id()), is_variable(), 327)(*this, dim, keepdim);
}
inline Tensor Tensor::mm(const Tensor & mat2) const {
    return find_op<Tensor, const Tensor &, const Tensor &>(tensorTypeIdToBackend(type_id()), is_variable(), 347)(*this, mat2);
}
inline std::tuple<Tensor,Tensor> Tensor::mode(int64_t dim, bool keepdim) const {
    return find_op<std::tuple<Tensor,Tensor>, const Tensor &, int64_t, bool>(tensorTypeIdToBackend(type_id()), is_variable(), 350)(*this, dim, keepdim);
}
inline Tensor Tensor::mul(const Tensor & other) const {
    return find_op<Tensor, const Tensor &, const Tensor &>(tensorTypeIdToBackend(type_id()), is_variable(), 352)(*this, other);
}
inline Tensor & Tensor::mul_(const Tensor & other) {
    return find_op<Tensor &, Tensor &, const Tensor &>(tensorTypeIdToBackend(type_id()), is_variable(), 353)(*this, other);
}
inline Tensor Tensor::mul(Scalar other) const {
    return find_op<Tensor, const Tensor &, Scalar>(tensorTypeIdToBackend(type_id()), is_variable(), 355)(*this, other);
}
inline Tensor & Tensor::mul_(Scalar other) {
    return find_op<Tensor &, Tensor &, Scalar>(tensorTypeIdToBackend(type_id()), is_variable(), 356)(*this, other);
}
inline Tensor Tensor::mv(const Tensor & vec) const {
    return find_op<Tensor, const Tensor &, const Tensor &>(tensorTypeIdToBackend(type_id()), is_variable(), 357)(*this, vec);
}
inline Tensor Tensor::mvlgamma(int64_t p) const {
    return find_op<Tensor, const Tensor &, int64_t>(tensorTypeIdToBackend(type_id()), is_variable(), 359)(*this, p);
}
inline Tensor & Tensor::mvlgamma_(int64_t p) {
    return find_op<Tensor &, Tensor &, int64_t>(tensorTypeIdToBackend(type_id()), is_variable(), 360)(*this, p);
}
inline Tensor Tensor::narrow_copy(int64_t dim, int64_t start, int64_t length) const {
    return find_op<Tensor, const Tensor &, int64_t, int64_t, int64_t>(tensorTypeIdToBackend(type_id()), is_variable(), 361)(*this, dim, start, length);
}
inline Tensor Tensor::narrow(int64_t dim, int64_t start, int64_t length) const {
    return find_op<Tensor, const Tensor &, int64_t, int64_t, int64_t>(tensorTypeIdToBackend(type_id()), is_variable(), 362)(*this, dim, start, length);
}
inline Tensor Tensor::permute(IntArrayRef dims) const {
    return find_op<Tensor, const Tensor &, IntArrayRef>(tensorTypeIdToBackend(type_id()), is_variable(), 387)(*this, dims);
}
inline Tensor Tensor::pin_memory() const {
    return find_op<Tensor, const Tensor &>(tensorTypeIdToBackend(type_id()), is_variable(), 389)(*this);
}
inline Tensor Tensor::pinverse(double rcond) const {
    return find_op<Tensor, const Tensor &, double>(tensorTypeIdToBackend(type_id()), is_variable(), 390)(*this, rcond);
}
inline Tensor Tensor::reciprocal() const {
    return find_op<Tensor, const Tensor &>(tensorTypeIdToBackend(type_id()), is_variable(), 424)(*this);
}
inline Tensor & Tensor::reciprocal_() {
    return find_op<Tensor &, Tensor &>(tensorTypeIdToBackend(type_id()), is_variable(), 425)(*this);
}
inline Tensor Tensor::neg() const {
    return find_op<Tensor, const Tensor &>(tensorTypeIdToBackend(type_id()), is_variable(), 427)(*this);
}
inline Tensor & Tensor::neg_() {
    return find_op<Tensor &, Tensor &>(tensorTypeIdToBackend(type_id()), is_variable(), 428)(*this);
}
inline Tensor Tensor::repeat(IntArrayRef repeats) const {
    return find_op<Tensor, const Tensor &, IntArrayRef>(tensorTypeIdToBackend(type_id()), is_variable(), 430)(*this, repeats);
}
inline Tensor Tensor::repeat_interleave(const Tensor & repeats, c10::optional<int64_t> dim) const {
    return find_op<Tensor, const Tensor &, const Tensor &, c10::optional<int64_t>>(tensorTypeIdToBackend(type_id()), is_variable(), 432)(*this, repeats, dim);
}
inline Tensor Tensor::repeat_interleave(int64_t repeats, c10::optional<int64_t> dim) const {
    return find_op<Tensor, const Tensor &, int64_t, c10::optional<int64_t>>(tensorTypeIdToBackend(type_id()), is_variable(), 433)(*this, repeats, dim);
}
inline Tensor Tensor::reshape(IntArrayRef shape) const {
    return find_op<Tensor, const Tensor &, IntArrayRef>(tensorTypeIdToBackend(type_id()), is_variable(), 434)(*this, shape);
}
inline Tensor Tensor::reshape_as(const Tensor & other) const {
    return find_op<Tensor, const Tensor &, const Tensor &>(tensorTypeIdToBackend(type_id()), is_variable(), 436)(*this, other);
}
inline Tensor Tensor::round() const {
    return find_op<Tensor, const Tensor &>(tensorTypeIdToBackend(type_id()), is_variable(), 437)(*this);
}
inline Tensor & Tensor::round_() {
    return find_op<Tensor &, Tensor &>(tensorTypeIdToBackend(type_id()), is_variable(), 438)(*this);
}
inline Tensor Tensor::relu() const {
    return find_op<Tensor, const Tensor &>(tensorTypeIdToBackend(type_id()), is_variable(), 442)(*this);
}
inline Tensor & Tensor::relu_() {
    return find_op<Tensor &, Tensor &>(tensorTypeIdToBackend(type_id()), is_variable(), 443)(*this);
}
inline Tensor Tensor::prelu(const Tensor & weight) const {
    return find_op<Tensor, const Tensor &, const Tensor &>(tensorTypeIdToBackend(type_id()), is_variable(), 444)(*this, weight);
}
inline std::tuple<Tensor,Tensor> Tensor::prelu_backward(const Tensor & grad_output, const Tensor & weight) const {
    return find_op<std::tuple<Tensor,Tensor>, const Tensor &, const Tensor &, const Tensor &>(tensorTypeIdToBackend(type_id()), is_variable(), 445)(grad_output, *this, weight);
}
inline Tensor Tensor::hardshrink(Scalar lambd) const {
    return find_op<Tensor, const Tensor &, Scalar>(tensorTypeIdToBackend(type_id()), is_variable(), 446)(*this, lambd);
}
inline Tensor Tensor::hardshrink_backward(const Tensor & grad_out, Scalar lambd) const {
    return find_op<Tensor, const Tensor &, const Tensor &, Scalar>(tensorTypeIdToBackend(type_id()), is_variable(), 447)(grad_out, *this, lambd);
}
inline Tensor Tensor::rsqrt() const {
    return find_op<Tensor, const Tensor &>(tensorTypeIdToBackend(type_id()), is_variable(), 448)(*this);
}
inline Tensor & Tensor::rsqrt_() {
    return find_op<Tensor &, Tensor &>(tensorTypeIdToBackend(type_id()), is_variable(), 449)(*this);
}
inline Tensor Tensor::select(int64_t dim, int64_t index) const {
    return find_op<Tensor, const Tensor &, int64_t, int64_t>(tensorTypeIdToBackend(type_id()), is_variable(), 451)(*this, dim, index);
}
inline Tensor Tensor::sigmoid() const {
    return find_op<Tensor, const Tensor &>(tensorTypeIdToBackend(type_id()), is_variable(), 456)(*this);
}
inline Tensor & Tensor::sigmoid_() {
    return find_op<Tensor &, Tensor &>(tensorTypeIdToBackend(type_id()), is_variable(), 457)(*this);
}
inline Tensor Tensor::sin() const {
    return find_op<Tensor, const Tensor &>(tensorTypeIdToBackend(type_id()), is_variable(), 459)(*this);
}
inline Tensor & Tensor::sin_() {
    return find_op<Tensor &, Tensor &>(tensorTypeIdToBackend(type_id()), is_variable(), 460)(*this);
}
inline Tensor Tensor::sinh() const {
    return find_op<Tensor, const Tensor &>(tensorTypeIdToBackend(type_id()), is_variable(), 462)(*this);
}
inline Tensor & Tensor::sinh_() {
    return find_op<Tensor &, Tensor &>(tensorTypeIdToBackend(type_id()), is_variable(), 463)(*this);
}
inline Tensor Tensor::detach() const {
    return find_op<Tensor, const Tensor &>(tensorTypeIdToBackend(type_id()), is_variable(), 465)(*this);
}
inline Tensor & Tensor::detach_() {
    return find_op<Tensor &, Tensor &>(tensorTypeIdToBackend(type_id()), is_variable(), 466)(*this);
}
inline int64_t Tensor::size(int64_t dim) const {
    return find_op<int64_t, const Tensor &, int64_t>(tensorTypeIdToBackend(type_id()), is_variable(), 467)(*this, dim);
}
inline Tensor Tensor::slice(int64_t dim, int64_t start, int64_t end, int64_t step) const {
    return find_op<Tensor, const Tensor &, int64_t, int64_t, int64_t, int64_t>(tensorTypeIdToBackend(type_id()), is_variable(), 468)(*this, dim, start, end, step);
}
inline std::tuple<Tensor,Tensor> Tensor::slogdet() const {
    return find_op<std::tuple<Tensor,Tensor>, const Tensor &>(tensorTypeIdToBackend(type_id()), is_variable(), 469)(*this);
}
inline Tensor Tensor::smm(const Tensor & mat2) const {
    return find_op<Tensor, const Tensor &, const Tensor &>(tensorTypeIdToBackend(type_id()), is_variable(), 470)(*this, mat2);
}
inline Tensor Tensor::softmax(int64_t dim, ScalarType dtype) const {
    return find_op<Tensor, const Tensor &, int64_t, ScalarType>(tensorTypeIdToBackend(type_id()), is_variable(), 471)(*this, dim, dtype);
}
inline Tensor Tensor::softmax(int64_t dim) const {
    return find_op<Tensor, const Tensor &, int64_t>(tensorTypeIdToBackend(type_id()), is_variable(), 472)(*this, dim);
}
inline std::vector<Tensor> Tensor::split(int64_t split_size, int64_t dim) const {
    return find_op<std::vector<Tensor>, const Tensor &, int64_t, int64_t>(tensorTypeIdToBackend(type_id()), is_variable(), 482)(*this, split_size, dim);
}
inline std::vector<Tensor> Tensor::split_with_sizes(IntArrayRef split_sizes, int64_t dim) const {
    return find_op<std::vector<Tensor>, const Tensor &, IntArrayRef, int64_t>(tensorTypeIdToBackend(type_id()), is_variable(), 483)(*this, split_sizes, dim);
}
inline Tensor Tensor::squeeze() const {
    return find_op<Tensor, const Tensor &>(tensorTypeIdToBackend(type_id()), is_variable(), 484)(*this);
}
inline Tensor Tensor::squeeze(int64_t dim) const {
    return find_op<Tensor, const Tensor &, int64_t>(tensorTypeIdToBackend(type_id()), is_variable(), 485)(*this, dim);
}
inline Tensor & Tensor::squeeze_() {
    return find_op<Tensor &, Tensor &>(tensorTypeIdToBackend(type_id()), is_variable(), 486)(*this);
}
inline Tensor & Tensor::squeeze_(int64_t dim) {
    return find_op<Tensor &, Tensor &, int64_t>(tensorTypeIdToBackend(type_id()), is_variable(), 487)(*this, dim);
}
inline Tensor Tensor::sspaddmm(const Tensor & mat1, const Tensor & mat2, Scalar beta, Scalar alpha) const {
    return find_op<Tensor, const Tensor &, const Tensor &, const Tensor &, Scalar, Scalar>(tensorTypeIdToBackend(type_id()), is_variable(), 488)(*this, mat1, mat2, beta, alpha);
}
inline Tensor Tensor::stft(int64_t n_fft, c10::optional<int64_t> hop_length, c10::optional<int64_t> win_length, const Tensor & window, bool normalized, bool onesided) const {
    return find_op<Tensor, const Tensor &, int64_t, c10::optional<int64_t>, c10::optional<int64_t>, const Tensor &, bool, bool>(tensorTypeIdToBackend(type_id()), is_variable(), 492)(*this, n_fft, hop_length, win_length, window, normalized, onesided);
}
inline int64_t Tensor::stride(int64_t dim) const {
    return find_op<int64_t, const Tensor &, int64_t>(tensorTypeIdToBackend(type_id()), is_variable(), 493)(*this, dim);
}
inline Tensor Tensor::sum(ScalarType dtype) const {
    return find_op<Tensor, const Tensor &, ScalarType>(tensorTypeIdToBackend(type_id()), is_variable(), 494)(*this, dtype);
}
inline Tensor Tensor::sum() const {
    return find_op<Tensor, const Tensor &>(tensorTypeIdToBackend(type_id()), is_variable(), 495)(*this);
}
inline Tensor Tensor::sum(IntArrayRef dim, bool keepdim, ScalarType dtype) const {
    return find_op<Tensor, const Tensor &, IntArrayRef, bool, ScalarType>(tensorTypeIdToBackend(type_id()), is_variable(), 496)(*this, dim, keepdim, dtype);
}
inline Tensor Tensor::sum(IntArrayRef dim, bool keepdim) const {
    return find_op<Tensor, const Tensor &, IntArrayRef, bool>(tensorTypeIdToBackend(type_id()), is_variable(), 497)(*this, dim, keepdim);
}
inline Tensor Tensor::sum(IntArrayRef dim, ScalarType dtype) const {
    return find_op<Tensor, const Tensor &, IntArrayRef, ScalarType>(tensorTypeIdToBackend(type_id()), is_variable(), 498)(*this, dim, dtype);
}
inline Tensor Tensor::sum_to_size(IntArrayRef size) const {
    return find_op<Tensor, const Tensor &, IntArrayRef>(tensorTypeIdToBackend(type_id()), is_variable(), 502)(*this, size);
}
inline Tensor Tensor::sqrt() const {
    return find_op<Tensor, const Tensor &>(tensorTypeIdToBackend(type_id()), is_variable(), 503)(*this);
}
inline Tensor & Tensor::sqrt_() {
    return find_op<Tensor &, Tensor &>(tensorTypeIdToBackend(type_id()), is_variable(), 504)(*this);
}
inline Tensor Tensor::std(bool unbiased) const {
    return find_op<Tensor, const Tensor &, bool>(tensorTypeIdToBackend(type_id()), is_variable(), 506)(*this, unbiased);
}
inline Tensor Tensor::std(IntArrayRef dim, bool unbiased, bool keepdim) const {
    return find_op<Tensor, const Tensor &, IntArrayRef, bool, bool>(tensorTypeIdToBackend(type_id()), is_variable(), 507)(*this, dim, unbiased, keepdim);
}
inline Tensor Tensor::prod(ScalarType dtype) const {
    return find_op<Tensor, const Tensor &, ScalarType>(tensorTypeIdToBackend(type_id()), is_variable(), 511)(*this, dtype);
}
inline Tensor Tensor::prod() const {
    return find_op<Tensor, const Tensor &>(tensorTypeIdToBackend(type_id()), is_variable(), 512)(*this);
}
inline Tensor Tensor::prod(int64_t dim, bool keepdim, ScalarType dtype) const {
    return find_op<Tensor, const Tensor &, int64_t, bool, ScalarType>(tensorTypeIdToBackend(type_id()), is_variable(), 513)(*this, dim, keepdim, dtype);
}
inline Tensor Tensor::prod(int64_t dim, bool keepdim) const {
    return find_op<Tensor, const Tensor &, int64_t, bool>(tensorTypeIdToBackend(type_id()), is_variable(), 514)(*this, dim, keepdim);
}
inline Tensor Tensor::prod(int64_t dim, ScalarType dtype) const {
    return find_op<Tensor, const Tensor &, int64_t, ScalarType>(tensorTypeIdToBackend(type_id()), is_variable(), 515)(*this, dim, dtype);
}
inline Tensor Tensor::t() const {
    return find_op<Tensor, const Tensor &>(tensorTypeIdToBackend(type_id()), is_variable(), 519)(*this);
}
inline Tensor & Tensor::t_() {
    return find_op<Tensor &, Tensor &>(tensorTypeIdToBackend(type_id()), is_variable(), 520)(*this);
}
inline Tensor Tensor::tan() const {
    return find_op<Tensor, const Tensor &>(tensorTypeIdToBackend(type_id()), is_variable(), 521)(*this);
}
inline Tensor & Tensor::tan_() {
    return find_op<Tensor &, Tensor &>(tensorTypeIdToBackend(type_id()), is_variable(), 522)(*this);
}
inline Tensor Tensor::tanh() const {
    return find_op<Tensor, const Tensor &>(tensorTypeIdToBackend(type_id()), is_variable(), 524)(*this);
}
inline Tensor & Tensor::tanh_() {
    return find_op<Tensor &, Tensor &>(tensorTypeIdToBackend(type_id()), is_variable(), 525)(*this);
}
inline Tensor Tensor::transpose(int64_t dim0, int64_t dim1) const {
    return find_op<Tensor, const Tensor &, int64_t, int64_t>(tensorTypeIdToBackend(type_id()), is_variable(), 532)(*this, dim0, dim1);
}
inline Tensor & Tensor::transpose_(int64_t dim0, int64_t dim1) {
    return find_op<Tensor &, Tensor &, int64_t, int64_t>(tensorTypeIdToBackend(type_id()), is_variable(), 533)(*this, dim0, dim1);
}
inline Tensor Tensor::flip(IntArrayRef dims) const {
    return find_op<Tensor, const Tensor &, IntArrayRef>(tensorTypeIdToBackend(type_id()), is_variable(), 535)(*this, dims);
}
inline Tensor Tensor::roll(IntArrayRef shifts, IntArrayRef dims) const {
    return find_op<Tensor, const Tensor &, IntArrayRef, IntArrayRef>(tensorTypeIdToBackend(type_id()), is_variable(), 536)(*this, shifts, dims);
}
inline Tensor Tensor::rot90(int64_t k, IntArrayRef dims) const {
    return find_op<Tensor, const Tensor &, int64_t, IntArrayRef>(tensorTypeIdToBackend(type_id()), is_variable(), 537)(*this, k, dims);
}
inline Tensor Tensor::trunc() const {
    return find_op<Tensor, const Tensor &>(tensorTypeIdToBackend(type_id()), is_variable(), 540)(*this);
}
inline Tensor & Tensor::trunc_() {
    return find_op<Tensor &, Tensor &>(tensorTypeIdToBackend(type_id()), is_variable(), 541)(*this);
}
inline Tensor Tensor::type_as(const Tensor & other) const {
    return find_op<Tensor, const Tensor &, const Tensor &>(tensorTypeIdToBackend(type_id()), is_variable(), 543)(*this, other);
}
inline Tensor Tensor::unsqueeze(int64_t dim) const {
    return find_op<Tensor, const Tensor &, int64_t>(tensorTypeIdToBackend(type_id()), is_variable(), 550)(*this, dim);
}
inline Tensor & Tensor::unsqueeze_(int64_t dim) {
    return find_op<Tensor &, Tensor &, int64_t>(tensorTypeIdToBackend(type_id()), is_variable(), 551)(*this, dim);
}
inline Tensor Tensor::var(bool unbiased) const {
    return find_op<Tensor, const Tensor &, bool>(tensorTypeIdToBackend(type_id()), is_variable(), 552)(*this, unbiased);
}
inline Tensor Tensor::var(IntArrayRef dim, bool unbiased, bool keepdim) const {
    return find_op<Tensor, const Tensor &, IntArrayRef, bool, bool>(tensorTypeIdToBackend(type_id()), is_variable(), 553)(*this, dim, unbiased, keepdim);
}
inline Tensor Tensor::view_as(const Tensor & other) const {
    return find_op<Tensor, const Tensor &, const Tensor &>(tensorTypeIdToBackend(type_id()), is_variable(), 557)(*this, other);
}
inline Tensor Tensor::where(const Tensor & condition, const Tensor & other) const {
    return find_op<Tensor, const Tensor &, const Tensor &, const Tensor &>(tensorTypeIdToBackend(type_id()), is_variable(), 558)(condition, *this, other);
}
inline Tensor Tensor::norm(c10::optional<Scalar> p, ScalarType dtype) const {
    return find_op<Tensor, const Tensor &, c10::optional<Scalar>, ScalarType>(tensorTypeIdToBackend(type_id()), is_variable(), 579)(*this, p, dtype);
}
inline Tensor Tensor::norm(Scalar p) const {
    return find_op<Tensor, const Tensor &, Scalar>(tensorTypeIdToBackend(type_id()), is_variable(), 580)(*this, p);
}
inline Tensor Tensor::norm(c10::optional<Scalar> p, IntArrayRef dim, bool keepdim, ScalarType dtype) const {
    return find_op<Tensor, const Tensor &, c10::optional<Scalar>, IntArrayRef, bool, ScalarType>(tensorTypeIdToBackend(type_id()), is_variable(), 581)(*this, p, dim, keepdim, dtype);
}
inline Tensor Tensor::norm(c10::optional<Scalar> p, IntArrayRef dim, bool keepdim) const {
    return find_op<Tensor, const Tensor &, c10::optional<Scalar>, IntArrayRef, bool>(tensorTypeIdToBackend(type_id()), is_variable(), 582)(*this, p, dim, keepdim);
}
inline Tensor Tensor::clone() const {
    return find_op<Tensor, const Tensor &>(tensorTypeIdToBackend(type_id()), is_variable(), 590)(*this);
}
inline Tensor & Tensor::resize_as_(const Tensor & the_template) {
    return find_op<Tensor &, Tensor &, const Tensor &>(tensorTypeIdToBackend(type_id()), is_variable(), 591)(*this, the_template);
}
inline Tensor Tensor::pow(Scalar exponent) const {
    return find_op<Tensor, const Tensor &, Scalar>(tensorTypeIdToBackend(type_id()), is_variable(), 593)(*this, exponent);
}
inline Tensor & Tensor::zero_() {
    return find_op<Tensor &, Tensor &>(tensorTypeIdToBackend(type_id()), is_variable(), 594)(*this);
}
inline Tensor Tensor::sub(const Tensor & other, Scalar alpha) const {
    return find_op<Tensor, const Tensor &, const Tensor &, Scalar>(tensorTypeIdToBackend(type_id()), is_variable(), 596)(*this, other, alpha);
}
inline Tensor & Tensor::sub_(const Tensor & other, Scalar alpha) {
    return find_op<Tensor &, Tensor &, const Tensor &, Scalar>(tensorTypeIdToBackend(type_id()), is_variable(), 597)(*this, other, alpha);
}
inline Tensor Tensor::sub(Scalar other, Scalar alpha) const {
    return find_op<Tensor, const Tensor &, Scalar, Scalar>(tensorTypeIdToBackend(type_id()), is_variable(), 598)(*this, other, alpha);
}
inline Tensor & Tensor::sub_(Scalar other, Scalar alpha) {
    return find_op<Tensor &, Tensor &, Scalar, Scalar>(tensorTypeIdToBackend(type_id()), is_variable(), 599)(*this, other, alpha);
}
inline Tensor Tensor::addmm(const Tensor & mat1, const Tensor & mat2, Scalar beta, Scalar alpha) const {
    return find_op<Tensor, const Tensor &, const Tensor &, const Tensor &, Scalar, Scalar>(tensorTypeIdToBackend(type_id()), is_variable(), 607)(*this, mat1, mat2, beta, alpha);
}
inline Tensor & Tensor::addmm_(const Tensor & mat1, const Tensor & mat2, Scalar beta, Scalar alpha) {
    return find_op<Tensor &, Tensor &, const Tensor &, const Tensor &, Scalar, Scalar>(tensorTypeIdToBackend(type_id()), is_variable(), 608)(*this, mat1, mat2, beta, alpha);
}
inline Tensor & Tensor::sparse_resize_(IntArrayRef size, int64_t sparse_dim, int64_t dense_dim) {
    return find_op<Tensor &, Tensor &, IntArrayRef, int64_t, int64_t>(tensorTypeIdToBackend(type_id()), is_variable(), 615)(*this, size, sparse_dim, dense_dim);
}
inline Tensor & Tensor::sparse_resize_and_clear_(IntArrayRef size, int64_t sparse_dim, int64_t dense_dim) {
    return find_op<Tensor &, Tensor &, IntArrayRef, int64_t, int64_t>(tensorTypeIdToBackend(type_id()), is_variable(), 616)(*this, size, sparse_dim, dense_dim);
}
inline Tensor Tensor::sparse_mask(const Tensor & mask) const {
    return find_op<Tensor, const Tensor &, const Tensor &>(tensorTypeIdToBackend(type_id()), is_variable(), 617)(*this, mask);
}
inline Tensor Tensor::to_dense() const {
    return find_op<Tensor, const Tensor &>(tensorTypeIdToBackend(type_id()), is_variable(), 618)(*this);
}
inline int64_t Tensor::sparse_dim() const {
    return find_op<int64_t, const Tensor &>(tensorTypeIdToBackend(type_id()), is_variable(), 620)(*this);
}
inline int64_t Tensor::_dimI() const {
    return find_op<int64_t, const Tensor &>(tensorTypeIdToBackend(type_id()), is_variable(), 621)(*this);
}
inline int64_t Tensor::dense_dim() const {
    return find_op<int64_t, const Tensor &>(tensorTypeIdToBackend(type_id()), is_variable(), 622)(*this);
}
inline int64_t Tensor::_dimV() const {
    return find_op<int64_t, const Tensor &>(tensorTypeIdToBackend(type_id()), is_variable(), 623)(*this);
}
inline int64_t Tensor::_nnz() const {
    return find_op<int64_t, const Tensor &>(tensorTypeIdToBackend(type_id()), is_variable(), 624)(*this);
}
inline Tensor Tensor::coalesce() const {
    return find_op<Tensor, const Tensor &>(tensorTypeIdToBackend(type_id()), is_variable(), 625)(*this);
}
inline bool Tensor::is_coalesced() const {
    return find_op<bool, const Tensor &>(tensorTypeIdToBackend(type_id()), is_variable(), 626)(*this);
}
inline Tensor Tensor::_indices() const {
    return find_op<Tensor, const Tensor &>(tensorTypeIdToBackend(type_id()), is_variable(), 627)(*this);
}
inline Tensor Tensor::_values() const {
    return find_op<Tensor, const Tensor &>(tensorTypeIdToBackend(type_id()), is_variable(), 628)(*this);
}
inline Tensor & Tensor::_coalesced_(bool coalesced) {
    return find_op<Tensor &, Tensor &, bool>(tensorTypeIdToBackend(type_id()), is_variable(), 629)(*this, coalesced);
}
inline Tensor Tensor::indices() const {
    return find_op<Tensor, const Tensor &>(tensorTypeIdToBackend(type_id()), is_variable(), 630)(*this);
}
inline Tensor Tensor::values() const {
    return find_op<Tensor, const Tensor &>(tensorTypeIdToBackend(type_id()), is_variable(), 631)(*this);
}
inline int64_t Tensor::numel() const {
    return find_op<int64_t, const Tensor &>(tensorTypeIdToBackend(type_id()), is_variable(), 635)(*this);
}
inline std::vector<Tensor> Tensor::unbind(int64_t dim) const {
    return find_op<std::vector<Tensor>, const Tensor &, int64_t>(tensorTypeIdToBackend(type_id()), is_variable(), 636)(*this, dim);
}
inline Tensor Tensor::to_sparse(int64_t sparse_dim) const {
    return find_op<Tensor, const Tensor &, int64_t>(tensorTypeIdToBackend(type_id()), is_variable(), 637)(*this, sparse_dim);
}
inline Tensor Tensor::to_sparse() const {
    return find_op<Tensor, const Tensor &>(tensorTypeIdToBackend(type_id()), is_variable(), 638)(*this);
}
inline Tensor Tensor::to_mkldnn() const {
    return find_op<Tensor, const Tensor &>(tensorTypeIdToBackend(type_id()), is_variable(), 639)(*this);
}
inline Tensor Tensor::quantize_linear(double scale, int64_t zero_point, ScalarType dtype) const {
    return find_op<Tensor, const Tensor &, double, int64_t, ScalarType>(tensorTypeIdToBackend(type_id()), is_variable(), 642)(*this, scale, zero_point, dtype);
}
inline Tensor Tensor::dequantize() const {
    return find_op<Tensor, const Tensor &>(tensorTypeIdToBackend(type_id()), is_variable(), 644)(*this);
}
inline Scalar Tensor::q_scale() const {
    return find_op<Scalar, const Tensor &>(tensorTypeIdToBackend(type_id()), is_variable(), 646)(*this);
}
inline Scalar Tensor::q_zero_point() const {
    return find_op<Scalar, const Tensor &>(tensorTypeIdToBackend(type_id()), is_variable(), 647)(*this);
}
inline Tensor Tensor::int_repr() const {
    return find_op<Tensor, const Tensor &>(tensorTypeIdToBackend(type_id()), is_variable(), 648)(*this);
}
inline Tensor Tensor::to(const TensorOptions & options, bool non_blocking, bool copy) const {
    return find_op<Tensor, const Tensor &, const TensorOptions &, bool, bool>(tensorTypeIdToBackend(type_id()), is_variable(), 650)(*this, options, non_blocking, copy);
}
inline Tensor Tensor::to(Device device, ScalarType dtype, bool non_blocking, bool copy) const {
    return find_op<Tensor, const Tensor &, Device, ScalarType, bool, bool>(tensorTypeIdToBackend(type_id()), is_variable(), 651)(*this, device, dtype, non_blocking, copy);
}
inline Tensor Tensor::to(ScalarType dtype, bool non_blocking, bool copy) const {
    return find_op<Tensor, const Tensor &, ScalarType, bool, bool>(tensorTypeIdToBackend(type_id()), is_variable(), 652)(*this, dtype, non_blocking, copy);
}
inline Tensor Tensor::to(const Tensor & other, bool non_blocking, bool copy) const {
    return find_op<Tensor, const Tensor &, const Tensor &, bool, bool>(tensorTypeIdToBackend(type_id()), is_variable(), 653)(*this, other, non_blocking, copy);
}
inline Scalar Tensor::item() const {
    return find_op<Scalar, const Tensor &>(tensorTypeIdToBackend(type_id()), is_variable(), 657)(*this);
}
inline void* Tensor::data_ptr() const {
    return find_op<void*, const Tensor &>(tensorTypeIdToBackend(type_id()), is_variable(), 683)(*this);
}
inline Tensor & Tensor::set_(Storage source) {
    return find_op<Tensor &, Tensor &, Storage>(tensorTypeIdToBackend(type_id()), is_variable(), 684)(*this, source);
}
inline Tensor & Tensor::set_(Storage source, int64_t storage_offset, IntArrayRef size, IntArrayRef stride) {
    return find_op<Tensor &, Tensor &, Storage, int64_t, IntArrayRef, IntArrayRef>(tensorTypeIdToBackend(type_id()), is_variable(), 685)(*this, source, storage_offset, size, stride);
}
inline Tensor & Tensor::set_(const Tensor & source) {
    return find_op<Tensor &, Tensor &, const Tensor &>(tensorTypeIdToBackend(type_id()), is_variable(), 686)(*this, source);
}
inline Tensor & Tensor::set_() {
    return find_op<Tensor &, Tensor &>(tensorTypeIdToBackend(type_id()), is_variable(), 687)(*this);
}
inline bool Tensor::is_set_to(const Tensor & tensor) const {
    return find_op<bool, const Tensor &, const Tensor &>(tensorTypeIdToBackend(type_id()), is_variable(), 688)(*this, tensor);
}
inline Tensor & Tensor::masked_fill_(const Tensor & mask, Scalar value) {
    return find_op<Tensor &, Tensor &, const Tensor &, Scalar>(tensorTypeIdToBackend(type_id()), is_variable(), 689)(*this, mask, value);
}
inline Tensor Tensor::masked_fill(const Tensor & mask, Scalar value) const {
    return find_op<Tensor, const Tensor &, const Tensor &, Scalar>(tensorTypeIdToBackend(type_id()), is_variable(), 690)(*this, mask, value);
}
inline Tensor & Tensor::masked_fill_(const Tensor & mask, const Tensor & value) {
    return find_op<Tensor &, Tensor &, const Tensor &, const Tensor &>(tensorTypeIdToBackend(type_id()), is_variable(), 691)(*this, mask, value);
}
inline Tensor Tensor::masked_fill(const Tensor & mask, const Tensor & value) const {
    return find_op<Tensor, const Tensor &, const Tensor &, const Tensor &>(tensorTypeIdToBackend(type_id()), is_variable(), 692)(*this, mask, value);
}
inline Tensor & Tensor::masked_scatter_(const Tensor & mask, const Tensor & source) {
    return find_op<Tensor &, Tensor &, const Tensor &, const Tensor &>(tensorTypeIdToBackend(type_id()), is_variable(), 693)(*this, mask, source);
}
inline Tensor Tensor::masked_scatter(const Tensor & mask, const Tensor & source) const {
    return find_op<Tensor, const Tensor &, const Tensor &, const Tensor &>(tensorTypeIdToBackend(type_id()), is_variable(), 694)(*this, mask, source);
}
inline Tensor Tensor::view(IntArrayRef size) const {
    return find_op<Tensor, const Tensor &, IntArrayRef>(tensorTypeIdToBackend(type_id()), is_variable(), 695)(*this, size);
}
inline Tensor & Tensor::put_(const Tensor & index, const Tensor & source, bool accumulate) {
    return find_op<Tensor &, Tensor &, const Tensor &, const Tensor &, bool>(tensorTypeIdToBackend(type_id()), is_variable(), 696)(*this, index, source, accumulate);
}
inline Tensor & Tensor::index_add_(int64_t dim, const Tensor & index, const Tensor & source) {
    return find_op<Tensor &, Tensor &, int64_t, const Tensor &, const Tensor &>(tensorTypeIdToBackend(type_id()), is_variable(), 697)(*this, dim, index, source);
}
inline Tensor Tensor::index_add(int64_t dim, const Tensor & index, const Tensor & source) const {
    return find_op<Tensor, const Tensor &, int64_t, const Tensor &, const Tensor &>(tensorTypeIdToBackend(type_id()), is_variable(), 698)(*this, dim, index, source);
}
inline Tensor & Tensor::index_fill_(int64_t dim, const Tensor & index, Scalar value) {
    return find_op<Tensor &, Tensor &, int64_t, const Tensor &, Scalar>(tensorTypeIdToBackend(type_id()), is_variable(), 699)(*this, dim, index, value);
}
inline Tensor Tensor::index_fill(int64_t dim, const Tensor & index, Scalar value) const {
    return find_op<Tensor, const Tensor &, int64_t, const Tensor &, Scalar>(tensorTypeIdToBackend(type_id()), is_variable(), 700)(*this, dim, index, value);
}
inline Tensor & Tensor::index_fill_(int64_t dim, const Tensor & index, const Tensor & value) {
    return find_op<Tensor &, Tensor &, int64_t, const Tensor &, const Tensor &>(tensorTypeIdToBackend(type_id()), is_variable(), 701)(*this, dim, index, value);
}
inline Tensor Tensor::index_fill(int64_t dim, const Tensor & index, const Tensor & value) const {
    return find_op<Tensor, const Tensor &, int64_t, const Tensor &, const Tensor &>(tensorTypeIdToBackend(type_id()), is_variable(), 702)(*this, dim, index, value);
}
inline Tensor & Tensor::scatter_(int64_t dim, const Tensor & index, const Tensor & src) {
    return find_op<Tensor &, Tensor &, int64_t, const Tensor &, const Tensor &>(tensorTypeIdToBackend(type_id()), is_variable(), 703)(*this, dim, index, src);
}
inline Tensor Tensor::scatter(int64_t dim, const Tensor & index, const Tensor & src) const {
    return find_op<Tensor, const Tensor &, int64_t, const Tensor &, const Tensor &>(tensorTypeIdToBackend(type_id()), is_variable(), 704)(*this, dim, index, src);
}
inline Tensor & Tensor::scatter_(int64_t dim, const Tensor & index, Scalar value) {
    return find_op<Tensor &, Tensor &, int64_t, const Tensor &, Scalar>(tensorTypeIdToBackend(type_id()), is_variable(), 705)(*this, dim, index, value);
}
inline Tensor Tensor::scatter(int64_t dim, const Tensor & index, Scalar value) const {
    return find_op<Tensor, const Tensor &, int64_t, const Tensor &, Scalar>(tensorTypeIdToBackend(type_id()), is_variable(), 706)(*this, dim, index, value);
}
inline Tensor & Tensor::scatter_add_(int64_t dim, const Tensor & index, const Tensor & src) {
    return find_op<Tensor &, Tensor &, int64_t, const Tensor &, const Tensor &>(tensorTypeIdToBackend(type_id()), is_variable(), 707)(*this, dim, index, src);
}
inline Tensor Tensor::scatter_add(int64_t dim, const Tensor & index, const Tensor & src) const {
    return find_op<Tensor, const Tensor &, int64_t, const Tensor &, const Tensor &>(tensorTypeIdToBackend(type_id()), is_variable(), 708)(*this, dim, index, src);
}
inline Tensor & Tensor::lt_(Scalar other) {
    return find_op<Tensor &, Tensor &, Scalar>(tensorTypeIdToBackend(type_id()), is_variable(), 709)(*this, other);
}
inline Tensor & Tensor::lt_(const Tensor & other) {
    return find_op<Tensor &, Tensor &, const Tensor &>(tensorTypeIdToBackend(type_id()), is_variable(), 710)(*this, other);
}
inline Tensor & Tensor::gt_(Scalar other) {
    return find_op<Tensor &, Tensor &, Scalar>(tensorTypeIdToBackend(type_id()), is_variable(), 711)(*this, other);
}
inline Tensor & Tensor::gt_(const Tensor & other) {
    return find_op<Tensor &, Tensor &, const Tensor &>(tensorTypeIdToBackend(type_id()), is_variable(), 712)(*this, other);
}
inline Tensor & Tensor::le_(Scalar other) {
    return find_op<Tensor &, Tensor &, Scalar>(tensorTypeIdToBackend(type_id()), is_variable(), 713)(*this, other);
}
inline Tensor & Tensor::le_(const Tensor & other) {
    return find_op<Tensor &, Tensor &, const Tensor &>(tensorTypeIdToBackend(type_id()), is_variable(), 714)(*this, other);
}
inline Tensor & Tensor::ge_(Scalar other) {
    return find_op<Tensor &, Tensor &, Scalar>(tensorTypeIdToBackend(type_id()), is_variable(), 715)(*this, other);
}
inline Tensor & Tensor::ge_(const Tensor & other) {
    return find_op<Tensor &, Tensor &, const Tensor &>(tensorTypeIdToBackend(type_id()), is_variable(), 716)(*this, other);
}
inline Tensor & Tensor::eq_(Scalar other) {
    return find_op<Tensor &, Tensor &, Scalar>(tensorTypeIdToBackend(type_id()), is_variable(), 717)(*this, other);
}
inline Tensor & Tensor::eq_(const Tensor & other) {
    return find_op<Tensor &, Tensor &, const Tensor &>(tensorTypeIdToBackend(type_id()), is_variable(), 718)(*this, other);
}
inline Tensor & Tensor::ne_(Scalar other) {
    return find_op<Tensor &, Tensor &, Scalar>(tensorTypeIdToBackend(type_id()), is_variable(), 719)(*this, other);
}
inline Tensor & Tensor::ne_(const Tensor & other) {
    return find_op<Tensor &, Tensor &, const Tensor &>(tensorTypeIdToBackend(type_id()), is_variable(), 720)(*this, other);
}
inline Tensor Tensor::__and__(Scalar other) const {
    return find_op<Tensor, const Tensor &, Scalar>(tensorTypeIdToBackend(type_id()), is_variable(), 721)(*this, other);
}
inline Tensor Tensor::__and__(const Tensor & other) const {
    return find_op<Tensor, const Tensor &, const Tensor &>(tensorTypeIdToBackend(type_id()), is_variable(), 722)(*this, other);
}
inline Tensor & Tensor::__iand__(Scalar other) {
    return find_op<Tensor &, Tensor &, Scalar>(tensorTypeIdToBackend(type_id()), is_variable(), 723)(*this, other);
}
inline Tensor & Tensor::__iand__(const Tensor & other) {
    return find_op<Tensor &, Tensor &, const Tensor &>(tensorTypeIdToBackend(type_id()), is_variable(), 724)(*this, other);
}
inline Tensor Tensor::__or__(Scalar other) const {
    return find_op<Tensor, const Tensor &, Scalar>(tensorTypeIdToBackend(type_id()), is_variable(), 725)(*this, other);
}
inline Tensor Tensor::__or__(const Tensor & other) const {
    return find_op<Tensor, const Tensor &, const Tensor &>(tensorTypeIdToBackend(type_id()), is_variable(), 726)(*this, other);
}
inline Tensor & Tensor::__ior__(Scalar other) {
    return find_op<Tensor &, Tensor &, Scalar>(tensorTypeIdToBackend(type_id()), is_variable(), 727)(*this, other);
}
inline Tensor & Tensor::__ior__(const Tensor & other) {
    return find_op<Tensor &, Tensor &, const Tensor &>(tensorTypeIdToBackend(type_id()), is_variable(), 728)(*this, other);
}
inline Tensor Tensor::__xor__(Scalar other) const {
    return find_op<Tensor, const Tensor &, Scalar>(tensorTypeIdToBackend(type_id()), is_variable(), 729)(*this, other);
}
inline Tensor Tensor::__xor__(const Tensor & other) const {
    return find_op<Tensor, const Tensor &, const Tensor &>(tensorTypeIdToBackend(type_id()), is_variable(), 730)(*this, other);
}
inline Tensor & Tensor::__ixor__(Scalar other) {
    return find_op<Tensor &, Tensor &, Scalar>(tensorTypeIdToBackend(type_id()), is_variable(), 731)(*this, other);
}
inline Tensor & Tensor::__ixor__(const Tensor & other) {
    return find_op<Tensor &, Tensor &, const Tensor &>(tensorTypeIdToBackend(type_id()), is_variable(), 732)(*this, other);
}
inline Tensor Tensor::__lshift__(Scalar other) const {
    return find_op<Tensor, const Tensor &, Scalar>(tensorTypeIdToBackend(type_id()), is_variable(), 733)(*this, other);
}
inline Tensor Tensor::__lshift__(const Tensor & other) const {
    return find_op<Tensor, const Tensor &, const Tensor &>(tensorTypeIdToBackend(type_id()), is_variable(), 734)(*this, other);
}
inline Tensor & Tensor::__ilshift__(Scalar other) {
    return find_op<Tensor &, Tensor &, Scalar>(tensorTypeIdToBackend(type_id()), is_variable(), 735)(*this, other);
}
inline Tensor & Tensor::__ilshift__(const Tensor & other) {
    return find_op<Tensor &, Tensor &, const Tensor &>(tensorTypeIdToBackend(type_id()), is_variable(), 736)(*this, other);
}
inline Tensor Tensor::__rshift__(Scalar other) const {
    return find_op<Tensor, const Tensor &, Scalar>(tensorTypeIdToBackend(type_id()), is_variable(), 737)(*this, other);
}
inline Tensor Tensor::__rshift__(const Tensor & other) const {
    return find_op<Tensor, const Tensor &, const Tensor &>(tensorTypeIdToBackend(type_id()), is_variable(), 738)(*this, other);
}
inline Tensor & Tensor::__irshift__(Scalar other) {
    return find_op<Tensor &, Tensor &, Scalar>(tensorTypeIdToBackend(type_id()), is_variable(), 739)(*this, other);
}
inline Tensor & Tensor::__irshift__(const Tensor & other) {
    return find_op<Tensor &, Tensor &, const Tensor &>(tensorTypeIdToBackend(type_id()), is_variable(), 740)(*this, other);
}
inline Tensor & Tensor::lgamma_() {
    return find_op<Tensor &, Tensor &>(tensorTypeIdToBackend(type_id()), is_variable(), 741)(*this);
}
inline Tensor & Tensor::atan2_(const Tensor & other) {
    return find_op<Tensor &, Tensor &, const Tensor &>(tensorTypeIdToBackend(type_id()), is_variable(), 742)(*this, other);
}
inline Tensor & Tensor::tril_(int64_t diagonal) {
    return find_op<Tensor &, Tensor &, int64_t>(tensorTypeIdToBackend(type_id()), is_variable(), 743)(*this, diagonal);
}
inline Tensor & Tensor::triu_(int64_t diagonal) {
    return find_op<Tensor &, Tensor &, int64_t>(tensorTypeIdToBackend(type_id()), is_variable(), 744)(*this, diagonal);
}
inline Tensor & Tensor::digamma_() {
    return find_op<Tensor &, Tensor &>(tensorTypeIdToBackend(type_id()), is_variable(), 745)(*this);
}
inline Tensor & Tensor::polygamma_(int64_t n) {
    return find_op<Tensor &, Tensor &, int64_t>(tensorTypeIdToBackend(type_id()), is_variable(), 746)(*this, n);
}
inline Tensor & Tensor::erfinv_() {
    return find_op<Tensor &, Tensor &>(tensorTypeIdToBackend(type_id()), is_variable(), 747)(*this);
}
inline Tensor & Tensor::renorm_(Scalar p, int64_t dim, Scalar maxnorm) {
    return find_op<Tensor &, Tensor &, Scalar, int64_t, Scalar>(tensorTypeIdToBackend(type_id()), is_variable(), 748)(*this, p, dim, maxnorm);
}
inline Tensor & Tensor::pow_(Scalar exponent) {
    return find_op<Tensor &, Tensor &, Scalar>(tensorTypeIdToBackend(type_id()), is_variable(), 749)(*this, exponent);
}
inline Tensor & Tensor::pow_(const Tensor & exponent) {
    return find_op<Tensor &, Tensor &, const Tensor &>(tensorTypeIdToBackend(type_id()), is_variable(), 750)(*this, exponent);
}
inline Tensor & Tensor::lerp_(const Tensor & end, Scalar weight) {
    return find_op<Tensor &, Tensor &, const Tensor &, Scalar>(tensorTypeIdToBackend(type_id()), is_variable(), 751)(*this, end, weight);
}
inline Tensor & Tensor::lerp_(const Tensor & end, const Tensor & weight) {
    return find_op<Tensor &, Tensor &, const Tensor &, const Tensor &>(tensorTypeIdToBackend(type_id()), is_variable(), 752)(*this, end, weight);
}
inline Tensor & Tensor::sign_() {
    return find_op<Tensor &, Tensor &>(tensorTypeIdToBackend(type_id()), is_variable(), 753)(*this);
}
inline Tensor & Tensor::fmod_(Scalar other) {
    return find_op<Tensor &, Tensor &, Scalar>(tensorTypeIdToBackend(type_id()), is_variable(), 754)(*this, other);
}
inline Tensor & Tensor::fmod_(const Tensor & other) {
    return find_op<Tensor &, Tensor &, const Tensor &>(tensorTypeIdToBackend(type_id()), is_variable(), 755)(*this, other);
}
inline Tensor & Tensor::remainder_(Scalar other) {
    return find_op<Tensor &, Tensor &, Scalar>(tensorTypeIdToBackend(type_id()), is_variable(), 756)(*this, other);
}
inline Tensor & Tensor::remainder_(const Tensor & other) {
    return find_op<Tensor &, Tensor &, const Tensor &>(tensorTypeIdToBackend(type_id()), is_variable(), 757)(*this, other);
}
inline Tensor & Tensor::addbmm_(const Tensor & batch1, const Tensor & batch2, Scalar beta, Scalar alpha) {
    return find_op<Tensor &, Tensor &, const Tensor &, const Tensor &, Scalar, Scalar>(tensorTypeIdToBackend(type_id()), is_variable(), 758)(*this, batch1, batch2, beta, alpha);
}
inline Tensor Tensor::addbmm(const Tensor & batch1, const Tensor & batch2, Scalar beta, Scalar alpha) const {
    return find_op<Tensor, const Tensor &, const Tensor &, const Tensor &, Scalar, Scalar>(tensorTypeIdToBackend(type_id()), is_variable(), 760)(*this, batch1, batch2, beta, alpha);
}
inline Tensor & Tensor::addcmul_(const Tensor & tensor1, const Tensor & tensor2, Scalar value) {
    return find_op<Tensor &, Tensor &, const Tensor &, const Tensor &, Scalar>(tensorTypeIdToBackend(type_id()), is_variable(), 761)(*this, tensor1, tensor2, value);
}
inline Tensor & Tensor::addcdiv_(const Tensor & tensor1, const Tensor & tensor2, Scalar value) {
    return find_op<Tensor &, Tensor &, const Tensor &, const Tensor &, Scalar>(tensorTypeIdToBackend(type_id()), is_variable(), 762)(*this, tensor1, tensor2, value);
}
inline Tensor & Tensor::random_(int64_t from, int64_t to, Generator * generator) {
    return find_op<Tensor &, Tensor &, int64_t, int64_t, Generator *>(tensorTypeIdToBackend(type_id()), is_variable(), 763)(*this, from, to, generator);
}
inline Tensor & Tensor::random_(int64_t to, Generator * generator) {
    return find_op<Tensor &, Tensor &, int64_t, Generator *>(tensorTypeIdToBackend(type_id()), is_variable(), 764)(*this, to, generator);
}
inline Tensor & Tensor::random_(Generator * generator) {
    return find_op<Tensor &, Tensor &, Generator *>(tensorTypeIdToBackend(type_id()), is_variable(), 765)(*this, generator);
}
inline Tensor & Tensor::uniform_(double from, double to, Generator * generator) {
    return find_op<Tensor &, Tensor &, double, double, Generator *>(tensorTypeIdToBackend(type_id()), is_variable(), 766)(*this, from, to, generator);
}
inline Tensor & Tensor::normal_(double mean, double std, Generator * generator) {
    return find_op<Tensor &, Tensor &, double, double, Generator *>(tensorTypeIdToBackend(type_id()), is_variable(), 767)(*this, mean, std, generator);
}
inline Tensor & Tensor::cauchy_(double median, double sigma, Generator * generator) {
    return find_op<Tensor &, Tensor &, double, double, Generator *>(tensorTypeIdToBackend(type_id()), is_variable(), 768)(*this, median, sigma, generator);
}
inline Tensor & Tensor::log_normal_(double mean, double std, Generator * generator) {
    return find_op<Tensor &, Tensor &, double, double, Generator *>(tensorTypeIdToBackend(type_id()), is_variable(), 769)(*this, mean, std, generator);
}
inline Tensor & Tensor::exponential_(double lambd, Generator * generator) {
    return find_op<Tensor &, Tensor &, double, Generator *>(tensorTypeIdToBackend(type_id()), is_variable(), 770)(*this, lambd, generator);
}
inline Tensor & Tensor::geometric_(double p, Generator * generator) {
    return find_op<Tensor &, Tensor &, double, Generator *>(tensorTypeIdToBackend(type_id()), is_variable(), 771)(*this, p, generator);
}
inline Tensor Tensor::diag(int64_t diagonal) const {
    return find_op<Tensor, const Tensor &, int64_t>(tensorTypeIdToBackend(type_id()), is_variable(), 773)(*this, diagonal);
}
inline Tensor Tensor::cross(const Tensor & other, c10::optional<int64_t> dim) const {
    return find_op<Tensor, const Tensor &, const Tensor &, c10::optional<int64_t>>(tensorTypeIdToBackend(type_id()), is_variable(), 775)(*this, other, dim);
}
inline Tensor Tensor::triu(int64_t diagonal) const {
    return find_op<Tensor, const Tensor &, int64_t>(tensorTypeIdToBackend(type_id()), is_variable(), 777)(*this, diagonal);
}
inline Tensor Tensor::tril(int64_t diagonal) const {
    return find_op<Tensor, const Tensor &, int64_t>(tensorTypeIdToBackend(type_id()), is_variable(), 779)(*this, diagonal);
}
inline Tensor Tensor::trace() const {
    return find_op<Tensor, const Tensor &>(tensorTypeIdToBackend(type_id()), is_variable(), 782)(*this);
}
inline Tensor Tensor::ne(Scalar other) const {
    return find_op<Tensor, const Tensor &, Scalar>(tensorTypeIdToBackend(type_id()), is_variable(), 784)(*this, other);
}
inline Tensor Tensor::ne(const Tensor & other) const {
    return find_op<Tensor, const Tensor &, const Tensor &>(tensorTypeIdToBackend(type_id()), is_variable(), 786)(*this, other);
}
inline Tensor Tensor::eq(Scalar other) const {
    return find_op<Tensor, const Tensor &, Scalar>(tensorTypeIdToBackend(type_id()), is_variable(), 788)(*this, other);
}
inline Tensor Tensor::eq(const Tensor & other) const {
    return find_op<Tensor, const Tensor &, const Tensor &>(tensorTypeIdToBackend(type_id()), is_variable(), 790)(*this, other);
}
inline Tensor Tensor::ge(Scalar other) const {
    return find_op<Tensor, const Tensor &, Scalar>(tensorTypeIdToBackend(type_id()), is_variable(), 792)(*this, other);
}
inline Tensor Tensor::ge(const Tensor & other) const {
    return find_op<Tensor, const Tensor &, const Tensor &>(tensorTypeIdToBackend(type_id()), is_variable(), 794)(*this, other);
}
inline Tensor Tensor::le(Scalar other) const {
    return find_op<Tensor, const Tensor &, Scalar>(tensorTypeIdToBackend(type_id()), is_variable(), 796)(*this, other);
}
inline Tensor Tensor::le(const Tensor & other) const {
    return find_op<Tensor, const Tensor &, const Tensor &>(tensorTypeIdToBackend(type_id()), is_variable(), 798)(*this, other);
}
inline Tensor Tensor::gt(Scalar other) const {
    return find_op<Tensor, const Tensor &, Scalar>(tensorTypeIdToBackend(type_id()), is_variable(), 800)(*this, other);
}
inline Tensor Tensor::gt(const Tensor & other) const {
    return find_op<Tensor, const Tensor &, const Tensor &>(tensorTypeIdToBackend(type_id()), is_variable(), 802)(*this, other);
}
inline Tensor Tensor::lt(Scalar other) const {
    return find_op<Tensor, const Tensor &, Scalar>(tensorTypeIdToBackend(type_id()), is_variable(), 804)(*this, other);
}
inline Tensor Tensor::lt(const Tensor & other) const {
    return find_op<Tensor, const Tensor &, const Tensor &>(tensorTypeIdToBackend(type_id()), is_variable(), 806)(*this, other);
}
inline Tensor Tensor::take(const Tensor & index) const {
    return find_op<Tensor, const Tensor &, const Tensor &>(tensorTypeIdToBackend(type_id()), is_variable(), 808)(*this, index);
}
inline Tensor Tensor::index_select(int64_t dim, const Tensor & index) const {
    return find_op<Tensor, const Tensor &, int64_t, const Tensor &>(tensorTypeIdToBackend(type_id()), is_variable(), 810)(*this, dim, index);
}
inline Tensor Tensor::masked_select(const Tensor & mask) const {
    return find_op<Tensor, const Tensor &, const Tensor &>(tensorTypeIdToBackend(type_id()), is_variable(), 812)(*this, mask);
}
inline Tensor Tensor::nonzero() const {
    return find_op<Tensor, const Tensor &>(tensorTypeIdToBackend(type_id()), is_variable(), 814)(*this);
}
inline Tensor Tensor::gather(int64_t dim, const Tensor & index, bool sparse_grad) const {
    return find_op<Tensor, const Tensor &, int64_t, const Tensor &, bool>(tensorTypeIdToBackend(type_id()), is_variable(), 816)(*this, dim, index, sparse_grad);
}
inline Tensor Tensor::addcmul(const Tensor & tensor1, const Tensor & tensor2, Scalar value) const {
    return find_op<Tensor, const Tensor &, const Tensor &, const Tensor &, Scalar>(tensorTypeIdToBackend(type_id()), is_variable(), 819)(*this, tensor1, tensor2, value);
}
inline Tensor Tensor::addcdiv(const Tensor & tensor1, const Tensor & tensor2, Scalar value) const {
    return find_op<Tensor, const Tensor &, const Tensor &, const Tensor &, Scalar>(tensorTypeIdToBackend(type_id()), is_variable(), 821)(*this, tensor1, tensor2, value);
}
inline std::tuple<Tensor,Tensor> Tensor::gels(const Tensor & A) const {
    return find_op<std::tuple<Tensor,Tensor>, const Tensor &, const Tensor &>(tensorTypeIdToBackend(type_id()), is_variable(), 823)(*this, A);
}
inline std::tuple<Tensor,Tensor> Tensor::triangular_solve(const Tensor & A, bool upper, bool transpose, bool unitriangular) const {
    return find_op<std::tuple<Tensor,Tensor>, const Tensor &, const Tensor &, bool, bool, bool>(tensorTypeIdToBackend(type_id()), is_variable(), 825)(*this, A, upper, transpose, unitriangular);
}
inline std::tuple<Tensor,Tensor> Tensor::symeig(bool eigenvectors, bool upper) const {
    return find_op<std::tuple<Tensor,Tensor>, const Tensor &, bool, bool>(tensorTypeIdToBackend(type_id()), is_variable(), 828)(*this, eigenvectors, upper);
}
inline std::tuple<Tensor,Tensor> Tensor::eig(bool eigenvectors) const {
    return find_op<std::tuple<Tensor,Tensor>, const Tensor &, bool>(tensorTypeIdToBackend(type_id()), is_variable(), 830)(*this, eigenvectors);
}
inline std::tuple<Tensor,Tensor,Tensor> Tensor::svd(bool some, bool compute_uv) const {
    return find_op<std::tuple<Tensor,Tensor,Tensor>, const Tensor &, bool, bool>(tensorTypeIdToBackend(type_id()), is_variable(), 832)(*this, some, compute_uv);
}
inline Tensor Tensor::cholesky(bool upper) const {
    return find_op<Tensor, const Tensor &, bool>(tensorTypeIdToBackend(type_id()), is_variable(), 834)(*this, upper);
}
inline Tensor Tensor::cholesky_solve(const Tensor & input2, bool upper) const {
    return find_op<Tensor, const Tensor &, const Tensor &, bool>(tensorTypeIdToBackend(type_id()), is_variable(), 837)(*this, input2, upper);
}
inline std::tuple<Tensor,Tensor> Tensor::solve(const Tensor & A) const {
    return find_op<std::tuple<Tensor,Tensor>, const Tensor &, const Tensor &>(tensorTypeIdToBackend(type_id()), is_variable(), 839)(*this, A);
}
inline Tensor Tensor::cholesky_inverse(bool upper) const {
    return find_op<Tensor, const Tensor &, bool>(tensorTypeIdToBackend(type_id()), is_variable(), 843)(*this, upper);
}
inline std::tuple<Tensor,Tensor> Tensor::pstrf(bool upper, Scalar tol) const {
    return find_op<std::tuple<Tensor,Tensor>, const Tensor &, bool, Scalar>(tensorTypeIdToBackend(type_id()), is_variable(), 845)(*this, upper, tol);
}
inline std::tuple<Tensor,Tensor> Tensor::qr(bool some) const {
    return find_op<std::tuple<Tensor,Tensor>, const Tensor &, bool>(tensorTypeIdToBackend(type_id()), is_variable(), 847)(*this, some);
}
inline std::tuple<Tensor,Tensor> Tensor::geqrf() const {
    return find_op<std::tuple<Tensor,Tensor>, const Tensor &>(tensorTypeIdToBackend(type_id()), is_variable(), 850)(*this);
}
inline Tensor Tensor::orgqr(const Tensor & input2) const {
    return find_op<Tensor, const Tensor &, const Tensor &>(tensorTypeIdToBackend(type_id()), is_variable(), 852)(*this, input2);
}
inline Tensor Tensor::ormqr(const Tensor & input2, const Tensor & input3, bool left, bool transpose) const {
    return find_op<Tensor, const Tensor &, const Tensor &, const Tensor &, bool, bool>(tensorTypeIdToBackend(type_id()), is_variable(), 854)(*this, input2, input3, left, transpose);
}
inline Tensor Tensor::lu_solve(const Tensor & LU_data, const Tensor & LU_pivots) const {
    return find_op<Tensor, const Tensor &, const Tensor &, const Tensor &>(tensorTypeIdToBackend(type_id()), is_variable(), 857)(*this, LU_data, LU_pivots);
}
inline Tensor Tensor::multinomial(int64_t num_samples, bool replacement, Generator * generator) const {
    return find_op<Tensor, const Tensor &, int64_t, bool, Generator *>(tensorTypeIdToBackend(type_id()), is_variable(), 859)(*this, num_samples, replacement, generator);
}
inline Tensor Tensor::lgamma() const {
    return find_op<Tensor, const Tensor &>(tensorTypeIdToBackend(type_id()), is_variable(), 863)(*this);
}
inline Tensor Tensor::digamma() const {
    return find_op<Tensor, const Tensor &>(tensorTypeIdToBackend(type_id()), is_variable(), 865)(*this);
}
inline Tensor Tensor::polygamma(int64_t n) const {
    return find_op<Tensor, int64_t, const Tensor &>(tensorTypeIdToBackend(type_id()), is_variable(), 867)(n, *this);
}
inline Tensor Tensor::erfinv() const {
    return find_op<Tensor, const Tensor &>(tensorTypeIdToBackend(type_id()), is_variable(), 869)(*this);
}
inline Tensor Tensor::dist(const Tensor & other, Scalar p) const {
    return find_op<Tensor, const Tensor &, const Tensor &, Scalar>(tensorTypeIdToBackend(type_id()), is_variable(), 870)(*this, other, p);
}
inline Tensor Tensor::atan2(const Tensor & other) const {
    return find_op<Tensor, const Tensor &, const Tensor &>(tensorTypeIdToBackend(type_id()), is_variable(), 872)(*this, other);
}
inline Tensor Tensor::lerp(const Tensor & end, Scalar weight) const {
    return find_op<Tensor, const Tensor &, const Tensor &, Scalar>(tensorTypeIdToBackend(type_id()), is_variable(), 875)(*this, end, weight);
}
inline Tensor Tensor::lerp(const Tensor & end, const Tensor & weight) const {
    return find_op<Tensor, const Tensor &, const Tensor &, const Tensor &>(tensorTypeIdToBackend(type_id()), is_variable(), 876)(*this, end, weight);
}
inline Tensor Tensor::histc(int64_t bins, Scalar min, Scalar max) const {
    return find_op<Tensor, const Tensor &, int64_t, Scalar, Scalar>(tensorTypeIdToBackend(type_id()), is_variable(), 878)(*this, bins, min, max);
}
inline Tensor Tensor::sign() const {
    return find_op<Tensor, const Tensor &>(tensorTypeIdToBackend(type_id()), is_variable(), 880)(*this);
}
inline Tensor Tensor::fmod(Scalar other) const {
    return find_op<Tensor, const Tensor &, Scalar>(tensorTypeIdToBackend(type_id()), is_variable(), 882)(*this, other);
}
inline Tensor Tensor::fmod(const Tensor & other) const {
    return find_op<Tensor, const Tensor &, const Tensor &>(tensorTypeIdToBackend(type_id()), is_variable(), 884)(*this, other);
}
inline Tensor Tensor::remainder(Scalar other) const {
    return find_op<Tensor, const Tensor &, Scalar>(tensorTypeIdToBackend(type_id()), is_variable(), 886)(*this, other);
}
inline Tensor Tensor::remainder(const Tensor & other) const {
    return find_op<Tensor, const Tensor &, const Tensor &>(tensorTypeIdToBackend(type_id()), is_variable(), 888)(*this, other);
}
inline Tensor Tensor::min(const Tensor & other) const {
    return find_op<Tensor, const Tensor &, const Tensor &>(tensorTypeIdToBackend(type_id()), is_variable(), 890)(*this, other);
}
inline Tensor Tensor::min() const {
    return find_op<Tensor, const Tensor &>(tensorTypeIdToBackend(type_id()), is_variable(), 891)(*this);
}
inline Tensor Tensor::max(const Tensor & other) const {
    return find_op<Tensor, const Tensor &, const Tensor &>(tensorTypeIdToBackend(type_id()), is_variable(), 893)(*this, other);
}
inline Tensor Tensor::max() const {
    return find_op<Tensor, const Tensor &>(tensorTypeIdToBackend(type_id()), is_variable(), 894)(*this);
}
inline Tensor Tensor::median() const {
    return find_op<Tensor, const Tensor &>(tensorTypeIdToBackend(type_id()), is_variable(), 895)(*this);
}
inline std::tuple<Tensor,Tensor> Tensor::sort(int64_t dim, bool descending) const {
    return find_op<std::tuple<Tensor,Tensor>, const Tensor &, int64_t, bool>(tensorTypeIdToBackend(type_id()), is_variable(), 897)(*this, dim, descending);
}
inline Tensor Tensor::argsort(int64_t dim, bool descending) const {
    return find_op<Tensor, const Tensor &, int64_t, bool>(tensorTypeIdToBackend(type_id()), is_variable(), 898)(*this, dim, descending);
}
inline std::tuple<Tensor,Tensor> Tensor::topk(int64_t k, int64_t dim, bool largest, bool sorted) const {
    return find_op<std::tuple<Tensor,Tensor>, const Tensor &, int64_t, int64_t, bool, bool>(tensorTypeIdToBackend(type_id()), is_variable(), 900)(*this, k, dim, largest, sorted);
}
inline Tensor Tensor::all() const {
    return find_op<Tensor, const Tensor &>(tensorTypeIdToBackend(type_id()), is_variable(), 901)(*this);
}
inline Tensor Tensor::any() const {
    return find_op<Tensor, const Tensor &>(tensorTypeIdToBackend(type_id()), is_variable(), 902)(*this);
}
inline Tensor Tensor::renorm(Scalar p, int64_t dim, Scalar maxnorm) const {
    return find_op<Tensor, const Tensor &, Scalar, int64_t, Scalar>(tensorTypeIdToBackend(type_id()), is_variable(), 904)(*this, p, dim, maxnorm);
}
inline Tensor Tensor::unfold(int64_t dimension, int64_t size, int64_t step) const {
    return find_op<Tensor, const Tensor &, int64_t, int64_t, int64_t>(tensorTypeIdToBackend(type_id()), is_variable(), 905)(*this, dimension, size, step);
}
inline bool Tensor::equal(const Tensor & other) const {
    return find_op<bool, const Tensor &, const Tensor &>(tensorTypeIdToBackend(type_id()), is_variable(), 906)(*this, other);
}
inline Tensor Tensor::pow(const Tensor & exponent) const {
    return find_op<Tensor, const Tensor &, const Tensor &>(tensorTypeIdToBackend(type_id()), is_variable(), 908)(*this, exponent);
}
inline Tensor Tensor::alias() const {
    return find_op<Tensor, const Tensor &>(tensorTypeIdToBackend(type_id()), is_variable(), 917)(*this);
}

inline bool Tensor::is_variable() const noexcept {
  return impl_->is_variable();
}

inline caffe2::TypeMeta Tensor::dtype() const noexcept {
  return impl_->dtype();
}

inline Layout Tensor::layout() const noexcept {
  return impl_->layout();
}

inline Device Tensor::device() const {
  return impl_->device();
}

inline int64_t Tensor::get_device() const {
  // NB: this is not a native function to avoid dispatching overhead.
  return impl_->get_device();
}

inline int64_t get_device(Tensor self) {
  return self.get_device();
}

inline bool Tensor::is_cuda() const {
  // NB: this is not a native function to avoid dispatching overhead.
  return impl_->is_cuda();
}

inline bool is_cuda(Tensor self) {
  return self.is_cuda();
}

inline bool Tensor::is_hip() const {
  // NB: this is not a native function to avoid dispatching overhead.
  return impl_->is_hip();
}

inline bool is_hip(Tensor self) {
  return self.is_hip();
}

inline bool Tensor::is_sparse() const {
  // NB: this is not a native function to avoid dispatching overhead.
  return impl_->is_sparse();
}

inline bool is_sparse(Tensor self) {
  return self.is_sparse();
}

inline bool Tensor::is_mkldnn() const {
  // NB: this is not a native function to avoid dispatching overhead.
  return impl_->is_mkldnn();
}

inline bool is_mkldnn(Tensor self) {
  return self.is_mkldnn();
}

inline bool Tensor::is_quantized() const {
  // NB: this is not a native function to avoid dispatching overhead.
  return impl_->is_quantized();
}

inline bool is_quantized(Tensor self) {
  return self.is_quantized();
}

#define DEFINE_CAST(T, name, _)                  \
  template <>                                    \
  inline T* Tensor::data() const {               \
    TORCH_CHECK(                                    \
        scalar_type() == ScalarType::name,       \
        "expected scalar type ",                 \
        #name,                                   \
        " but found ",                           \
        c10::toString(scalar_type()));           \
    return static_cast<T*>(this->data_ptr());    \
  }

AT_FORALL_SCALAR_TYPES_WITH_COMPLEX_EXCEPT_COMPLEX_HALF(DEFINE_CAST)
#undef DEFINE_CAST

#define DEFINE_ITEM(T, name, _)   \
  template <>                     \
  inline T Tensor::item() const { \
    return item().to##name();     \
  }

AT_FORALL_SCALAR_TYPES_WITH_COMPLEX_EXCEPT_COMPLEX_HALF_AND_QINT(DEFINE_ITEM)
#undef DEFINE_ITEM

} //namespace at
