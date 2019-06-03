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
    return globalATenDispatch().getOp<Tensor (const Tensor &)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 30, "abs")(*this);
}
inline Tensor & Tensor::abs_() {
    return globalATenDispatch().getOp<Tensor & (Tensor &)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 31, "abs_")(*this);
}
inline Tensor Tensor::acos() const {
    return globalATenDispatch().getOp<Tensor (const Tensor &)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 33, "acos")(*this);
}
inline Tensor & Tensor::acos_() {
    return globalATenDispatch().getOp<Tensor & (Tensor &)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 34, "acos_")(*this);
}
inline Tensor Tensor::add(const Tensor & other, Scalar alpha) const {
    return globalATenDispatch().getOp<Tensor (const Tensor &, const Tensor &, Scalar)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 39, "add")(*this, other, alpha);
}
inline Tensor & Tensor::add_(const Tensor & other, Scalar alpha) {
    return globalATenDispatch().getOp<Tensor & (Tensor &, const Tensor &, Scalar)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 40, "add_")(*this, other, alpha);
}
inline Tensor Tensor::add(Scalar other, Scalar alpha) const {
    return globalATenDispatch().getOp<Tensor (const Tensor &, Scalar, Scalar)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 42, "add")(*this, other, alpha);
}
inline Tensor & Tensor::add_(Scalar other, Scalar alpha) {
    return globalATenDispatch().getOp<Tensor & (Tensor &, Scalar, Scalar)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 43, "add_")(*this, other, alpha);
}
inline Tensor Tensor::addmv(const Tensor & mat, const Tensor & vec, Scalar beta, Scalar alpha) const {
    return globalATenDispatch().getOp<Tensor (const Tensor &, const Tensor &, const Tensor &, Scalar, Scalar)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 44, "addmv")(*this, mat, vec, beta, alpha);
}
inline Tensor & Tensor::addmv_(const Tensor & mat, const Tensor & vec, Scalar beta, Scalar alpha) {
    return globalATenDispatch().getOp<Tensor & (Tensor &, const Tensor &, const Tensor &, Scalar, Scalar)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 45, "addmv_")(*this, mat, vec, beta, alpha);
}
inline Tensor Tensor::addr(const Tensor & vec1, const Tensor & vec2, Scalar beta, Scalar alpha) const {
    return globalATenDispatch().getOp<Tensor (const Tensor &, const Tensor &, const Tensor &, Scalar, Scalar)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 47, "addr")(*this, vec1, vec2, beta, alpha);
}
inline Tensor & Tensor::addr_(const Tensor & vec1, const Tensor & vec2, Scalar beta, Scalar alpha) {
    return globalATenDispatch().getOp<Tensor & (Tensor &, const Tensor &, const Tensor &, Scalar, Scalar)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 48, "addr_")(*this, vec1, vec2, beta, alpha);
}
inline Tensor Tensor::all(int64_t dim, bool keepdim) const {
    return globalATenDispatch().getOp<Tensor (const Tensor &, int64_t, bool)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 52, "all")(*this, dim, keepdim);
}
inline bool Tensor::allclose(const Tensor & other, double rtol, double atol, bool equal_nan) const {
    return globalATenDispatch().getOp<bool (const Tensor &, const Tensor &, double, double, bool)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 54, "allclose")(*this, other, rtol, atol, equal_nan);
}
inline Tensor Tensor::any(int64_t dim, bool keepdim) const {
    return globalATenDispatch().getOp<Tensor (const Tensor &, int64_t, bool)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 55, "any")(*this, dim, keepdim);
}
inline Tensor Tensor::argmax(c10::optional<int64_t> dim, bool keepdim) const {
    return globalATenDispatch().getOp<Tensor (const Tensor &, c10::optional<int64_t>, bool)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 63, "argmax")(*this, dim, keepdim);
}
inline Tensor Tensor::argmin(c10::optional<int64_t> dim, bool keepdim) const {
    return globalATenDispatch().getOp<Tensor (const Tensor &, c10::optional<int64_t>, bool)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 64, "argmin")(*this, dim, keepdim);
}
inline Tensor Tensor::as_strided(IntArrayRef size, IntArrayRef stride, c10::optional<int64_t> storage_offset) const {
    return globalATenDispatch().getOp<Tensor (const Tensor &, IntArrayRef, IntArrayRef, c10::optional<int64_t>)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 65, "as_strided")(*this, size, stride, storage_offset);
}
inline Tensor & Tensor::as_strided_(IntArrayRef size, IntArrayRef stride, c10::optional<int64_t> storage_offset) {
    return globalATenDispatch().getOp<Tensor & (Tensor &, IntArrayRef, IntArrayRef, c10::optional<int64_t>)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 66, "as_strided_")(*this, size, stride, storage_offset);
}
inline Tensor Tensor::asin() const {
    return globalATenDispatch().getOp<Tensor (const Tensor &)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 67, "asin")(*this);
}
inline Tensor & Tensor::asin_() {
    return globalATenDispatch().getOp<Tensor & (Tensor &)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 68, "asin_")(*this);
}
inline Tensor Tensor::atan() const {
    return globalATenDispatch().getOp<Tensor (const Tensor &)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 70, "atan")(*this);
}
inline Tensor & Tensor::atan_() {
    return globalATenDispatch().getOp<Tensor & (Tensor &)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 71, "atan_")(*this);
}
inline Tensor Tensor::baddbmm(const Tensor & batch1, const Tensor & batch2, Scalar beta, Scalar alpha) const {
    return globalATenDispatch().getOp<Tensor (const Tensor &, const Tensor &, const Tensor &, Scalar, Scalar)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 73, "baddbmm")(*this, batch1, batch2, beta, alpha);
}
inline Tensor & Tensor::baddbmm_(const Tensor & batch1, const Tensor & batch2, Scalar beta, Scalar alpha) {
    return globalATenDispatch().getOp<Tensor & (Tensor &, const Tensor &, const Tensor &, Scalar, Scalar)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 74, "baddbmm_")(*this, batch1, batch2, beta, alpha);
}
inline Tensor Tensor::bernoulli(Generator * generator) const {
    return globalATenDispatch().getOp<Tensor (const Tensor &, Generator *)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 82, "bernoulli")(*this, generator);
}
inline Tensor & Tensor::bernoulli_(const Tensor & p, Generator * generator) {
    return globalATenDispatch().getOp<Tensor & (Tensor &, const Tensor &, Generator *)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 84, "bernoulli_")(*this, p, generator);
}
inline Tensor & Tensor::bernoulli_(double p, Generator * generator) {
    return globalATenDispatch().getOp<Tensor & (Tensor &, double, Generator *)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 85, "bernoulli_")(*this, p, generator);
}
inline Tensor Tensor::bernoulli(double p, Generator * generator) const {
    return globalATenDispatch().getOp<Tensor (const Tensor &, double, Generator *)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 86, "bernoulli")(*this, p, generator);
}
inline Tensor Tensor::bincount(const Tensor & weights, int64_t minlength) const {
    return globalATenDispatch().getOp<Tensor (const Tensor &, const Tensor &, int64_t)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 90, "bincount")(*this, weights, minlength);
}
inline Tensor Tensor::bmm(const Tensor & mat2) const {
    return globalATenDispatch().getOp<Tensor (const Tensor &, const Tensor &)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 93, "bmm")(*this, mat2);
}
inline Tensor Tensor::ceil() const {
    return globalATenDispatch().getOp<Tensor (const Tensor &)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 98, "ceil")(*this);
}
inline Tensor & Tensor::ceil_() {
    return globalATenDispatch().getOp<Tensor & (Tensor &)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 99, "ceil_")(*this);
}
inline std::vector<Tensor> Tensor::chunk(int64_t chunks, int64_t dim) const {
    return globalATenDispatch().getOp<std::vector<Tensor> (const Tensor &, int64_t, int64_t)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 102, "chunk")(*this, chunks, dim);
}
inline Tensor Tensor::clamp(c10::optional<Scalar> min, c10::optional<Scalar> max) const {
    return globalATenDispatch().getOp<Tensor (const Tensor &, c10::optional<Scalar>, c10::optional<Scalar>)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 103, "clamp")(*this, min, max);
}
inline Tensor & Tensor::clamp_(c10::optional<Scalar> min, c10::optional<Scalar> max) {
    return globalATenDispatch().getOp<Tensor & (Tensor &, c10::optional<Scalar>, c10::optional<Scalar>)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 104, "clamp_")(*this, min, max);
}
inline Tensor Tensor::clamp_max(Scalar max) const {
    return globalATenDispatch().getOp<Tensor (const Tensor &, Scalar)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 106, "clamp_max")(*this, max);
}
inline Tensor & Tensor::clamp_max_(Scalar max) {
    return globalATenDispatch().getOp<Tensor & (Tensor &, Scalar)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 107, "clamp_max_")(*this, max);
}
inline Tensor Tensor::clamp_min(Scalar min) const {
    return globalATenDispatch().getOp<Tensor (const Tensor &, Scalar)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 109, "clamp_min")(*this, min);
}
inline Tensor & Tensor::clamp_min_(Scalar min) {
    return globalATenDispatch().getOp<Tensor & (Tensor &, Scalar)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 110, "clamp_min_")(*this, min);
}
inline Tensor Tensor::contiguous(MemoryFormat memory_format) const {
    return globalATenDispatch().getOp<Tensor (const Tensor &, MemoryFormat)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 114, "contiguous")(*this, memory_format);
}
inline Tensor & Tensor::copy_(const Tensor & src, bool non_blocking) {
    return globalATenDispatch().getOp<Tensor & (Tensor &, const Tensor &, bool)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 127, "copy_")(*this, src, non_blocking);
}
inline Tensor Tensor::cos() const {
    return globalATenDispatch().getOp<Tensor (const Tensor &)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 129, "cos")(*this);
}
inline Tensor & Tensor::cos_() {
    return globalATenDispatch().getOp<Tensor & (Tensor &)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 130, "cos_")(*this);
}
inline Tensor Tensor::cosh() const {
    return globalATenDispatch().getOp<Tensor (const Tensor &)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 132, "cosh")(*this);
}
inline Tensor & Tensor::cosh_() {
    return globalATenDispatch().getOp<Tensor & (Tensor &)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 133, "cosh_")(*this);
}
inline Tensor Tensor::cumsum(int64_t dim, ScalarType dtype) const {
    return globalATenDispatch().getOp<Tensor (const Tensor &, int64_t, ScalarType)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 152, "cumsum")(*this, dim, dtype);
}
inline Tensor Tensor::cumsum(int64_t dim) const {
    return globalATenDispatch().getOp<Tensor (const Tensor &, int64_t)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 153, "cumsum")(*this, dim);
}
inline Tensor Tensor::cumprod(int64_t dim, ScalarType dtype) const {
    return globalATenDispatch().getOp<Tensor (const Tensor &, int64_t, ScalarType)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 156, "cumprod")(*this, dim, dtype);
}
inline Tensor Tensor::cumprod(int64_t dim) const {
    return globalATenDispatch().getOp<Tensor (const Tensor &, int64_t)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 157, "cumprod")(*this, dim);
}
inline Tensor Tensor::det() const {
    return globalATenDispatch().getOp<Tensor (const Tensor &)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 164, "det")(*this);
}
inline Tensor Tensor::diag_embed(int64_t offset, int64_t dim1, int64_t dim2) const {
    return globalATenDispatch().getOp<Tensor (const Tensor &, int64_t, int64_t, int64_t)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 165, "diag_embed")(*this, offset, dim1, dim2);
}
inline Tensor Tensor::diagflat(int64_t offset) const {
    return globalATenDispatch().getOp<Tensor (const Tensor &, int64_t)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 166, "diagflat")(*this, offset);
}
inline Tensor Tensor::diagonal(int64_t offset, int64_t dim1, int64_t dim2) const {
    return globalATenDispatch().getOp<Tensor (const Tensor &, int64_t, int64_t, int64_t)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 167, "diagonal")(*this, offset, dim1, dim2);
}
inline Tensor Tensor::div(const Tensor & other) const {
    return globalATenDispatch().getOp<Tensor (const Tensor &, const Tensor &)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 168, "div")(*this, other);
}
inline Tensor & Tensor::div_(const Tensor & other) {
    return globalATenDispatch().getOp<Tensor & (Tensor &, const Tensor &)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 169, "div_")(*this, other);
}
inline Tensor Tensor::div(Scalar other) const {
    return globalATenDispatch().getOp<Tensor (const Tensor &, Scalar)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 171, "div")(*this, other);
}
inline Tensor & Tensor::div_(Scalar other) {
    return globalATenDispatch().getOp<Tensor & (Tensor &, Scalar)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 172, "div_")(*this, other);
}
inline Tensor Tensor::dot(const Tensor & tensor) const {
    return globalATenDispatch().getOp<Tensor (const Tensor &, const Tensor &)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 173, "dot")(*this, tensor);
}
inline Tensor & Tensor::resize_(IntArrayRef size) {
    return globalATenDispatch().getOp<Tensor & (Tensor &, IntArrayRef)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 189, "resize_")(*this, size);
}
inline Tensor Tensor::erf() const {
    return globalATenDispatch().getOp<Tensor (const Tensor &)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 194, "erf")(*this);
}
inline Tensor & Tensor::erf_() {
    return globalATenDispatch().getOp<Tensor & (Tensor &)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 195, "erf_")(*this);
}
inline Tensor Tensor::erfc() const {
    return globalATenDispatch().getOp<Tensor (const Tensor &)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 197, "erfc")(*this);
}
inline Tensor & Tensor::erfc_() {
    return globalATenDispatch().getOp<Tensor & (Tensor &)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 198, "erfc_")(*this);
}
inline Tensor Tensor::exp() const {
    return globalATenDispatch().getOp<Tensor (const Tensor &)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 200, "exp")(*this);
}
inline Tensor & Tensor::exp_() {
    return globalATenDispatch().getOp<Tensor & (Tensor &)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 201, "exp_")(*this);
}
inline Tensor Tensor::expm1() const {
    return globalATenDispatch().getOp<Tensor (const Tensor &)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 203, "expm1")(*this);
}
inline Tensor & Tensor::expm1_() {
    return globalATenDispatch().getOp<Tensor & (Tensor &)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 204, "expm1_")(*this);
}
inline Tensor Tensor::expand(IntArrayRef size, bool implicit) const {
    return globalATenDispatch().getOp<Tensor (const Tensor &, IntArrayRef, bool)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 206, "expand")(*this, size, implicit);
}
inline Tensor Tensor::expand_as(const Tensor & other) const {
    return globalATenDispatch().getOp<Tensor (const Tensor &, const Tensor &)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 207, "expand_as")(*this, other);
}
inline Tensor Tensor::flatten(int64_t start_dim, int64_t end_dim) const {
    return globalATenDispatch().getOp<Tensor (const Tensor &, int64_t, int64_t)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 212, "flatten")(*this, start_dim, end_dim);
}
inline Tensor & Tensor::fill_(Scalar value) {
    return globalATenDispatch().getOp<Tensor & (Tensor &, Scalar)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 213, "fill_")(*this, value);
}
inline Tensor & Tensor::fill_(const Tensor & value) {
    return globalATenDispatch().getOp<Tensor & (Tensor &, const Tensor &)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 214, "fill_")(*this, value);
}
inline Tensor Tensor::floor() const {
    return globalATenDispatch().getOp<Tensor (const Tensor &)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 215, "floor")(*this);
}
inline Tensor & Tensor::floor_() {
    return globalATenDispatch().getOp<Tensor & (Tensor &)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 216, "floor_")(*this);
}
inline Tensor Tensor::frac() const {
    return globalATenDispatch().getOp<Tensor (const Tensor &)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 218, "frac")(*this);
}
inline Tensor & Tensor::frac_() {
    return globalATenDispatch().getOp<Tensor & (Tensor &)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 219, "frac_")(*this);
}
inline Tensor Tensor::ger(const Tensor & vec2) const {
    return globalATenDispatch().getOp<Tensor (const Tensor &, const Tensor &)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 238, "ger")(*this, vec2);
}
inline Tensor Tensor::fft(int64_t signal_ndim, bool normalized) const {
    return globalATenDispatch().getOp<Tensor (const Tensor &, int64_t, bool)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 241, "fft")(*this, signal_ndim, normalized);
}
inline Tensor Tensor::ifft(int64_t signal_ndim, bool normalized) const {
    return globalATenDispatch().getOp<Tensor (const Tensor &, int64_t, bool)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 242, "ifft")(*this, signal_ndim, normalized);
}
inline Tensor Tensor::rfft(int64_t signal_ndim, bool normalized, bool onesided) const {
    return globalATenDispatch().getOp<Tensor (const Tensor &, int64_t, bool, bool)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 243, "rfft")(*this, signal_ndim, normalized, onesided);
}
inline Tensor Tensor::irfft(int64_t signal_ndim, bool normalized, bool onesided, IntArrayRef signal_sizes) const {
    return globalATenDispatch().getOp<Tensor (const Tensor &, int64_t, bool, bool, IntArrayRef)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 244, "irfft")(*this, signal_ndim, normalized, onesided, signal_sizes);
}
inline Tensor Tensor::index(TensorList indices) const {
    return globalATenDispatch().getOp<Tensor (const Tensor &, TensorList)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 250, "index")(*this, indices);
}
inline Tensor & Tensor::index_copy_(int64_t dim, const Tensor & index, const Tensor & source) {
    return globalATenDispatch().getOp<Tensor & (Tensor &, int64_t, const Tensor &, const Tensor &)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 251, "index_copy_")(*this, dim, index, source);
}
inline Tensor Tensor::index_copy(int64_t dim, const Tensor & index, const Tensor & source) const {
    return globalATenDispatch().getOp<Tensor (const Tensor &, int64_t, const Tensor &, const Tensor &)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 252, "index_copy")(*this, dim, index, source);
}
inline Tensor & Tensor::index_put_(TensorList indices, const Tensor & values, bool accumulate) {
    return globalATenDispatch().getOp<Tensor & (Tensor &, TensorList, const Tensor &, bool)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 253, "index_put_")(*this, indices, values, accumulate);
}
inline Tensor Tensor::index_put(TensorList indices, const Tensor & values, bool accumulate) const {
    return globalATenDispatch().getOp<Tensor (const Tensor &, TensorList, const Tensor &, bool)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 254, "index_put")(*this, indices, values, accumulate);
}
inline Tensor Tensor::inverse() const {
    return globalATenDispatch().getOp<Tensor (const Tensor &)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 257, "inverse")(*this);
}
inline Tensor Tensor::isclose(const Tensor & other, double rtol, double atol, bool equal_nan) const {
    return globalATenDispatch().getOp<Tensor (const Tensor &, const Tensor &, double, double, bool)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 260, "isclose")(*this, other, rtol, atol, equal_nan);
}
inline bool Tensor::is_distributed() const {
    return globalATenDispatch().getOp<bool (const Tensor &)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 262, "is_distributed")(*this);
}
inline bool Tensor::is_floating_point() const {
    return globalATenDispatch().getOp<bool (const Tensor &)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 263, "is_floating_point")(*this);
}
inline bool Tensor::is_complex() const {
    return globalATenDispatch().getOp<bool (const Tensor &)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 264, "is_complex")(*this);
}
inline bool Tensor::is_nonzero() const {
    return globalATenDispatch().getOp<bool (const Tensor &)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 265, "is_nonzero")(*this);
}
inline bool Tensor::is_same_size(const Tensor & other) const {
    return globalATenDispatch().getOp<bool (const Tensor &, const Tensor &)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 266, "is_same_size")(*this, other);
}
inline bool Tensor::is_signed() const {
    return globalATenDispatch().getOp<bool (const Tensor &)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 267, "is_signed")(*this);
}
inline std::tuple<Tensor,Tensor> Tensor::kthvalue(int64_t k, int64_t dim, bool keepdim) const {
    return globalATenDispatch().getOp<std::tuple<Tensor,Tensor> (const Tensor &, int64_t, int64_t, bool)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 270, "kthvalue")(*this, k, dim, keepdim);
}
inline Tensor Tensor::log() const {
    return globalATenDispatch().getOp<Tensor (const Tensor &)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 284, "log")(*this);
}
inline Tensor & Tensor::log_() {
    return globalATenDispatch().getOp<Tensor & (Tensor &)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 285, "log_")(*this);
}
inline Tensor Tensor::log10() const {
    return globalATenDispatch().getOp<Tensor (const Tensor &)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 287, "log10")(*this);
}
inline Tensor & Tensor::log10_() {
    return globalATenDispatch().getOp<Tensor & (Tensor &)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 288, "log10_")(*this);
}
inline Tensor Tensor::log1p() const {
    return globalATenDispatch().getOp<Tensor (const Tensor &)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 290, "log1p")(*this);
}
inline Tensor & Tensor::log1p_() {
    return globalATenDispatch().getOp<Tensor & (Tensor &)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 291, "log1p_")(*this);
}
inline Tensor Tensor::log2() const {
    return globalATenDispatch().getOp<Tensor (const Tensor &)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 293, "log2")(*this);
}
inline Tensor & Tensor::log2_() {
    return globalATenDispatch().getOp<Tensor & (Tensor &)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 294, "log2_")(*this);
}
inline Tensor Tensor::logdet() const {
    return globalATenDispatch().getOp<Tensor (const Tensor &)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 296, "logdet")(*this);
}
inline Tensor Tensor::log_softmax(int64_t dim, ScalarType dtype) const {
    return globalATenDispatch().getOp<Tensor (const Tensor &, int64_t, ScalarType)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 299, "log_softmax")(*this, dim, dtype);
}
inline Tensor Tensor::log_softmax(int64_t dim) const {
    return globalATenDispatch().getOp<Tensor (const Tensor &, int64_t)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 300, "log_softmax")(*this, dim);
}
inline Tensor Tensor::logsumexp(IntArrayRef dim, bool keepdim) const {
    return globalATenDispatch().getOp<Tensor (const Tensor &, IntArrayRef, bool)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 303, "logsumexp")(*this, dim, keepdim);
}
inline Tensor Tensor::matmul(const Tensor & other) const {
    return globalATenDispatch().getOp<Tensor (const Tensor &, const Tensor &)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 306, "matmul")(*this, other);
}
inline Tensor Tensor::matrix_power(int64_t n) const {
    return globalATenDispatch().getOp<Tensor (const Tensor &, int64_t)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 310, "matrix_power")(*this, n);
}
inline std::tuple<Tensor,Tensor> Tensor::max(int64_t dim, bool keepdim) const {
    return globalATenDispatch().getOp<std::tuple<Tensor,Tensor> (const Tensor &, int64_t, bool)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 311, "max")(*this, dim, keepdim);
}
inline Tensor Tensor::max_values(IntArrayRef dim, bool keepdim) const {
    return globalATenDispatch().getOp<Tensor (const Tensor &, IntArrayRef, bool)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 313, "max_values")(*this, dim, keepdim);
}
inline Tensor Tensor::mean(ScalarType dtype) const {
    return globalATenDispatch().getOp<Tensor (const Tensor &, ScalarType)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 319, "mean")(*this, dtype);
}
inline Tensor Tensor::mean() const {
    return globalATenDispatch().getOp<Tensor (const Tensor &)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 320, "mean")(*this);
}
inline Tensor Tensor::mean(IntArrayRef dim, bool keepdim, ScalarType dtype) const {
    return globalATenDispatch().getOp<Tensor (const Tensor &, IntArrayRef, bool, ScalarType)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 321, "mean")(*this, dim, keepdim, dtype);
}
inline Tensor Tensor::mean(IntArrayRef dim, bool keepdim) const {
    return globalATenDispatch().getOp<Tensor (const Tensor &, IntArrayRef, bool)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 322, "mean")(*this, dim, keepdim);
}
inline Tensor Tensor::mean(IntArrayRef dim, ScalarType dtype) const {
    return globalATenDispatch().getOp<Tensor (const Tensor &, IntArrayRef, ScalarType)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 323, "mean")(*this, dim, dtype);
}
inline std::tuple<Tensor,Tensor> Tensor::median(int64_t dim, bool keepdim) const {
    return globalATenDispatch().getOp<std::tuple<Tensor,Tensor> (const Tensor &, int64_t, bool)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 327, "median")(*this, dim, keepdim);
}
inline std::tuple<Tensor,Tensor> Tensor::min(int64_t dim, bool keepdim) const {
    return globalATenDispatch().getOp<std::tuple<Tensor,Tensor> (const Tensor &, int64_t, bool)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 329, "min")(*this, dim, keepdim);
}
inline Tensor Tensor::min_values(IntArrayRef dim, bool keepdim) const {
    return globalATenDispatch().getOp<Tensor (const Tensor &, IntArrayRef, bool)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 331, "min_values")(*this, dim, keepdim);
}
inline Tensor Tensor::mm(const Tensor & mat2) const {
    return globalATenDispatch().getOp<Tensor (const Tensor &, const Tensor &)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 351, "mm")(*this, mat2);
}
inline std::tuple<Tensor,Tensor> Tensor::mode(int64_t dim, bool keepdim) const {
    return globalATenDispatch().getOp<std::tuple<Tensor,Tensor> (const Tensor &, int64_t, bool)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 354, "mode")(*this, dim, keepdim);
}
inline Tensor Tensor::mul(const Tensor & other) const {
    return globalATenDispatch().getOp<Tensor (const Tensor &, const Tensor &)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 356, "mul")(*this, other);
}
inline Tensor & Tensor::mul_(const Tensor & other) {
    return globalATenDispatch().getOp<Tensor & (Tensor &, const Tensor &)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 357, "mul_")(*this, other);
}
inline Tensor Tensor::mul(Scalar other) const {
    return globalATenDispatch().getOp<Tensor (const Tensor &, Scalar)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 359, "mul")(*this, other);
}
inline Tensor & Tensor::mul_(Scalar other) {
    return globalATenDispatch().getOp<Tensor & (Tensor &, Scalar)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 360, "mul_")(*this, other);
}
inline Tensor Tensor::mv(const Tensor & vec) const {
    return globalATenDispatch().getOp<Tensor (const Tensor &, const Tensor &)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 361, "mv")(*this, vec);
}
inline Tensor Tensor::mvlgamma(int64_t p) const {
    return globalATenDispatch().getOp<Tensor (const Tensor &, int64_t)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 363, "mvlgamma")(*this, p);
}
inline Tensor & Tensor::mvlgamma_(int64_t p) {
    return globalATenDispatch().getOp<Tensor & (Tensor &, int64_t)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 364, "mvlgamma_")(*this, p);
}
inline Tensor Tensor::narrow_copy(int64_t dim, int64_t start, int64_t length) const {
    return globalATenDispatch().getOp<Tensor (const Tensor &, int64_t, int64_t, int64_t)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 365, "narrow_copy")(*this, dim, start, length);
}
inline Tensor Tensor::narrow(int64_t dim, int64_t start, int64_t length) const {
    return globalATenDispatch().getOp<Tensor (const Tensor &, int64_t, int64_t, int64_t)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 366, "narrow")(*this, dim, start, length);
}
inline Tensor Tensor::permute(IntArrayRef dims) const {
    return globalATenDispatch().getOp<Tensor (const Tensor &, IntArrayRef)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 391, "permute")(*this, dims);
}
inline Tensor Tensor::pin_memory() const {
    return globalATenDispatch().getOp<Tensor (const Tensor &)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 393, "pin_memory")(*this);
}
inline Tensor Tensor::pinverse(double rcond) const {
    return globalATenDispatch().getOp<Tensor (const Tensor &, double)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 394, "pinverse")(*this, rcond);
}
inline Tensor Tensor::reciprocal() const {
    return globalATenDispatch().getOp<Tensor (const Tensor &)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 428, "reciprocal")(*this);
}
inline Tensor & Tensor::reciprocal_() {
    return globalATenDispatch().getOp<Tensor & (Tensor &)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 429, "reciprocal_")(*this);
}
inline Tensor Tensor::neg() const {
    return globalATenDispatch().getOp<Tensor (const Tensor &)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 431, "neg")(*this);
}
inline Tensor & Tensor::neg_() {
    return globalATenDispatch().getOp<Tensor & (Tensor &)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 432, "neg_")(*this);
}
inline Tensor Tensor::repeat(IntArrayRef repeats) const {
    return globalATenDispatch().getOp<Tensor (const Tensor &, IntArrayRef)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 434, "repeat")(*this, repeats);
}
inline Tensor Tensor::repeat_interleave(const Tensor & repeats, c10::optional<int64_t> dim) const {
    return globalATenDispatch().getOp<Tensor (const Tensor &, const Tensor &, c10::optional<int64_t>)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 436, "repeat_interleave")(*this, repeats, dim);
}
inline Tensor Tensor::repeat_interleave(int64_t repeats, c10::optional<int64_t> dim) const {
    return globalATenDispatch().getOp<Tensor (const Tensor &, int64_t, c10::optional<int64_t>)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 437, "repeat_interleave")(*this, repeats, dim);
}
inline Tensor Tensor::reshape(IntArrayRef shape) const {
    return globalATenDispatch().getOp<Tensor (const Tensor &, IntArrayRef)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 438, "reshape")(*this, shape);
}
inline Tensor Tensor::reshape_as(const Tensor & other) const {
    return globalATenDispatch().getOp<Tensor (const Tensor &, const Tensor &)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 440, "reshape_as")(*this, other);
}
inline Tensor Tensor::round() const {
    return globalATenDispatch().getOp<Tensor (const Tensor &)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 441, "round")(*this);
}
inline Tensor & Tensor::round_() {
    return globalATenDispatch().getOp<Tensor & (Tensor &)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 442, "round_")(*this);
}
inline Tensor Tensor::relu() const {
    return globalATenDispatch().getOp<Tensor (const Tensor &)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 446, "relu")(*this);
}
inline Tensor & Tensor::relu_() {
    return globalATenDispatch().getOp<Tensor & (Tensor &)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 447, "relu_")(*this);
}
inline Tensor Tensor::prelu(const Tensor & weight) const {
    return globalATenDispatch().getOp<Tensor (const Tensor &, const Tensor &)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 448, "prelu")(*this, weight);
}
inline std::tuple<Tensor,Tensor> Tensor::prelu_backward(const Tensor & grad_output, const Tensor & weight) const {
    return globalATenDispatch().getOp<std::tuple<Tensor,Tensor> (const Tensor &, const Tensor &, const Tensor &)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 449, "prelu_backward")(grad_output, *this, weight);
}
inline Tensor Tensor::hardshrink(Scalar lambd) const {
    return globalATenDispatch().getOp<Tensor (const Tensor &, Scalar)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 452, "hardshrink")(*this, lambd);
}
inline Tensor Tensor::hardshrink_backward(const Tensor & grad_out, Scalar lambd) const {
    return globalATenDispatch().getOp<Tensor (const Tensor &, const Tensor &, Scalar)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 453, "hardshrink_backward")(grad_out, *this, lambd);
}
inline Tensor Tensor::rsqrt() const {
    return globalATenDispatch().getOp<Tensor (const Tensor &)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 454, "rsqrt")(*this);
}
inline Tensor & Tensor::rsqrt_() {
    return globalATenDispatch().getOp<Tensor & (Tensor &)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 455, "rsqrt_")(*this);
}
inline Tensor Tensor::select(int64_t dim, int64_t index) const {
    return globalATenDispatch().getOp<Tensor (const Tensor &, int64_t, int64_t)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 457, "select")(*this, dim, index);
}
inline Tensor Tensor::sigmoid() const {
    return globalATenDispatch().getOp<Tensor (const Tensor &)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 462, "sigmoid")(*this);
}
inline Tensor & Tensor::sigmoid_() {
    return globalATenDispatch().getOp<Tensor & (Tensor &)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 463, "sigmoid_")(*this);
}
inline Tensor Tensor::sin() const {
    return globalATenDispatch().getOp<Tensor (const Tensor &)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 465, "sin")(*this);
}
inline Tensor & Tensor::sin_() {
    return globalATenDispatch().getOp<Tensor & (Tensor &)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 466, "sin_")(*this);
}
inline Tensor Tensor::sinh() const {
    return globalATenDispatch().getOp<Tensor (const Tensor &)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 468, "sinh")(*this);
}
inline Tensor & Tensor::sinh_() {
    return globalATenDispatch().getOp<Tensor & (Tensor &)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 469, "sinh_")(*this);
}
inline Tensor Tensor::detach() const {
    return globalATenDispatch().getOp<Tensor (const Tensor &)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 471, "detach")(*this);
}
inline Tensor & Tensor::detach_() {
    return globalATenDispatch().getOp<Tensor & (Tensor &)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 472, "detach_")(*this);
}
inline int64_t Tensor::size(int64_t dim) const {
    return globalATenDispatch().getOp<int64_t (const Tensor &, int64_t)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 473, "size")(*this, dim);
}
inline Tensor Tensor::slice(int64_t dim, int64_t start, int64_t end, int64_t step) const {
    return globalATenDispatch().getOp<Tensor (const Tensor &, int64_t, int64_t, int64_t, int64_t)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 474, "slice")(*this, dim, start, end, step);
}
inline std::tuple<Tensor,Tensor> Tensor::slogdet() const {
    return globalATenDispatch().getOp<std::tuple<Tensor,Tensor> (const Tensor &)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 475, "slogdet")(*this);
}
inline Tensor Tensor::smm(const Tensor & mat2) const {
    return globalATenDispatch().getOp<Tensor (const Tensor &, const Tensor &)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 476, "smm")(*this, mat2);
}
inline Tensor Tensor::softmax(int64_t dim, ScalarType dtype) const {
    return globalATenDispatch().getOp<Tensor (const Tensor &, int64_t, ScalarType)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 477, "softmax")(*this, dim, dtype);
}
inline Tensor Tensor::softmax(int64_t dim) const {
    return globalATenDispatch().getOp<Tensor (const Tensor &, int64_t)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 478, "softmax")(*this, dim);
}
inline std::vector<Tensor> Tensor::split(int64_t split_size, int64_t dim) const {
    return globalATenDispatch().getOp<std::vector<Tensor> (const Tensor &, int64_t, int64_t)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 488, "split")(*this, split_size, dim);
}
inline std::vector<Tensor> Tensor::split_with_sizes(IntArrayRef split_sizes, int64_t dim) const {
    return globalATenDispatch().getOp<std::vector<Tensor> (const Tensor &, IntArrayRef, int64_t)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 489, "split_with_sizes")(*this, split_sizes, dim);
}
inline Tensor Tensor::squeeze() const {
    return globalATenDispatch().getOp<Tensor (const Tensor &)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 490, "squeeze")(*this);
}
inline Tensor Tensor::squeeze(int64_t dim) const {
    return globalATenDispatch().getOp<Tensor (const Tensor &, int64_t)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 491, "squeeze")(*this, dim);
}
inline Tensor & Tensor::squeeze_() {
    return globalATenDispatch().getOp<Tensor & (Tensor &)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 492, "squeeze_")(*this);
}
inline Tensor & Tensor::squeeze_(int64_t dim) {
    return globalATenDispatch().getOp<Tensor & (Tensor &, int64_t)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 493, "squeeze_")(*this, dim);
}
inline Tensor Tensor::sspaddmm(const Tensor & mat1, const Tensor & mat2, Scalar beta, Scalar alpha) const {
    return globalATenDispatch().getOp<Tensor (const Tensor &, const Tensor &, const Tensor &, Scalar, Scalar)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 494, "sspaddmm")(*this, mat1, mat2, beta, alpha);
}
inline Tensor Tensor::stft(int64_t n_fft, c10::optional<int64_t> hop_length, c10::optional<int64_t> win_length, const Tensor & window, bool normalized, bool onesided) const {
    return globalATenDispatch().getOp<Tensor (const Tensor &, int64_t, c10::optional<int64_t>, c10::optional<int64_t>, const Tensor &, bool, bool)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 498, "stft")(*this, n_fft, hop_length, win_length, window, normalized, onesided);
}
inline int64_t Tensor::stride(int64_t dim) const {
    return globalATenDispatch().getOp<int64_t (const Tensor &, int64_t)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 499, "stride")(*this, dim);
}
inline Tensor Tensor::sum(ScalarType dtype) const {
    return globalATenDispatch().getOp<Tensor (const Tensor &, ScalarType)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 500, "sum")(*this, dtype);
}
inline Tensor Tensor::sum() const {
    return globalATenDispatch().getOp<Tensor (const Tensor &)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 501, "sum")(*this);
}
inline Tensor Tensor::sum(IntArrayRef dim, bool keepdim, ScalarType dtype) const {
    return globalATenDispatch().getOp<Tensor (const Tensor &, IntArrayRef, bool, ScalarType)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 502, "sum")(*this, dim, keepdim, dtype);
}
inline Tensor Tensor::sum(IntArrayRef dim, bool keepdim) const {
    return globalATenDispatch().getOp<Tensor (const Tensor &, IntArrayRef, bool)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 503, "sum")(*this, dim, keepdim);
}
inline Tensor Tensor::sum(IntArrayRef dim, ScalarType dtype) const {
    return globalATenDispatch().getOp<Tensor (const Tensor &, IntArrayRef, ScalarType)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 504, "sum")(*this, dim, dtype);
}
inline Tensor Tensor::sum_to_size(IntArrayRef size) const {
    return globalATenDispatch().getOp<Tensor (const Tensor &, IntArrayRef)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 508, "sum_to_size")(*this, size);
}
inline Tensor Tensor::sqrt() const {
    return globalATenDispatch().getOp<Tensor (const Tensor &)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 509, "sqrt")(*this);
}
inline Tensor & Tensor::sqrt_() {
    return globalATenDispatch().getOp<Tensor & (Tensor &)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 510, "sqrt_")(*this);
}
inline Tensor Tensor::std(bool unbiased) const {
    return globalATenDispatch().getOp<Tensor (const Tensor &, bool)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 512, "std")(*this, unbiased);
}
inline Tensor Tensor::std(IntArrayRef dim, bool unbiased, bool keepdim) const {
    return globalATenDispatch().getOp<Tensor (const Tensor &, IntArrayRef, bool, bool)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 513, "std")(*this, dim, unbiased, keepdim);
}
inline Tensor Tensor::prod(ScalarType dtype) const {
    return globalATenDispatch().getOp<Tensor (const Tensor &, ScalarType)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 517, "prod")(*this, dtype);
}
inline Tensor Tensor::prod() const {
    return globalATenDispatch().getOp<Tensor (const Tensor &)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 518, "prod")(*this);
}
inline Tensor Tensor::prod(int64_t dim, bool keepdim, ScalarType dtype) const {
    return globalATenDispatch().getOp<Tensor (const Tensor &, int64_t, bool, ScalarType)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 519, "prod")(*this, dim, keepdim, dtype);
}
inline Tensor Tensor::prod(int64_t dim, bool keepdim) const {
    return globalATenDispatch().getOp<Tensor (const Tensor &, int64_t, bool)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 520, "prod")(*this, dim, keepdim);
}
inline Tensor Tensor::prod(int64_t dim, ScalarType dtype) const {
    return globalATenDispatch().getOp<Tensor (const Tensor &, int64_t, ScalarType)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 521, "prod")(*this, dim, dtype);
}
inline Tensor Tensor::t() const {
    return globalATenDispatch().getOp<Tensor (const Tensor &)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 525, "t")(*this);
}
inline Tensor & Tensor::t_() {
    return globalATenDispatch().getOp<Tensor & (Tensor &)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 526, "t_")(*this);
}
inline Tensor Tensor::tan() const {
    return globalATenDispatch().getOp<Tensor (const Tensor &)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 527, "tan")(*this);
}
inline Tensor & Tensor::tan_() {
    return globalATenDispatch().getOp<Tensor & (Tensor &)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 528, "tan_")(*this);
}
inline Tensor Tensor::tanh() const {
    return globalATenDispatch().getOp<Tensor (const Tensor &)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 530, "tanh")(*this);
}
inline Tensor & Tensor::tanh_() {
    return globalATenDispatch().getOp<Tensor & (Tensor &)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 531, "tanh_")(*this);
}
inline Tensor Tensor::transpose(int64_t dim0, int64_t dim1) const {
    return globalATenDispatch().getOp<Tensor (const Tensor &, int64_t, int64_t)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 538, "transpose")(*this, dim0, dim1);
}
inline Tensor & Tensor::transpose_(int64_t dim0, int64_t dim1) {
    return globalATenDispatch().getOp<Tensor & (Tensor &, int64_t, int64_t)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 539, "transpose_")(*this, dim0, dim1);
}
inline Tensor Tensor::flip(IntArrayRef dims) const {
    return globalATenDispatch().getOp<Tensor (const Tensor &, IntArrayRef)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 541, "flip")(*this, dims);
}
inline Tensor Tensor::roll(IntArrayRef shifts, IntArrayRef dims) const {
    return globalATenDispatch().getOp<Tensor (const Tensor &, IntArrayRef, IntArrayRef)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 542, "roll")(*this, shifts, dims);
}
inline Tensor Tensor::rot90(int64_t k, IntArrayRef dims) const {
    return globalATenDispatch().getOp<Tensor (const Tensor &, int64_t, IntArrayRef)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 543, "rot90")(*this, k, dims);
}
inline Tensor Tensor::trunc() const {
    return globalATenDispatch().getOp<Tensor (const Tensor &)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 546, "trunc")(*this);
}
inline Tensor & Tensor::trunc_() {
    return globalATenDispatch().getOp<Tensor & (Tensor &)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 547, "trunc_")(*this);
}
inline Tensor Tensor::type_as(const Tensor & other) const {
    return globalATenDispatch().getOp<Tensor (const Tensor &, const Tensor &)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 549, "type_as")(*this, other);
}
inline Tensor Tensor::unsqueeze(int64_t dim) const {
    return globalATenDispatch().getOp<Tensor (const Tensor &, int64_t)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 556, "unsqueeze")(*this, dim);
}
inline Tensor & Tensor::unsqueeze_(int64_t dim) {
    return globalATenDispatch().getOp<Tensor & (Tensor &, int64_t)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 557, "unsqueeze_")(*this, dim);
}
inline Tensor Tensor::var(bool unbiased) const {
    return globalATenDispatch().getOp<Tensor (const Tensor &, bool)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 558, "var")(*this, unbiased);
}
inline Tensor Tensor::var(IntArrayRef dim, bool unbiased, bool keepdim) const {
    return globalATenDispatch().getOp<Tensor (const Tensor &, IntArrayRef, bool, bool)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 559, "var")(*this, dim, unbiased, keepdim);
}
inline Tensor Tensor::view_as(const Tensor & other) const {
    return globalATenDispatch().getOp<Tensor (const Tensor &, const Tensor &)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 563, "view_as")(*this, other);
}
inline Tensor Tensor::where(const Tensor & condition, const Tensor & other) const {
    return globalATenDispatch().getOp<Tensor (const Tensor &, const Tensor &, const Tensor &)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 564, "where")(condition, *this, other);
}
inline Tensor Tensor::norm(c10::optional<Scalar> p, ScalarType dtype) const {
    return globalATenDispatch().getOp<Tensor (const Tensor &, c10::optional<Scalar>, ScalarType)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 585, "norm")(*this, p, dtype);
}
inline Tensor Tensor::norm(Scalar p) const {
    return globalATenDispatch().getOp<Tensor (const Tensor &, Scalar)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 586, "norm")(*this, p);
}
inline Tensor Tensor::norm(c10::optional<Scalar> p, IntArrayRef dim, bool keepdim, ScalarType dtype) const {
    return globalATenDispatch().getOp<Tensor (const Tensor &, c10::optional<Scalar>, IntArrayRef, bool, ScalarType)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 587, "norm")(*this, p, dim, keepdim, dtype);
}
inline Tensor Tensor::norm(c10::optional<Scalar> p, IntArrayRef dim, bool keepdim) const {
    return globalATenDispatch().getOp<Tensor (const Tensor &, c10::optional<Scalar>, IntArrayRef, bool)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 588, "norm")(*this, p, dim, keepdim);
}
inline Tensor Tensor::clone() const {
    return globalATenDispatch().getOp<Tensor (const Tensor &)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 596, "clone")(*this);
}
inline Tensor & Tensor::resize_as_(const Tensor & the_template) {
    return globalATenDispatch().getOp<Tensor & (Tensor &, const Tensor &)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 597, "resize_as_")(*this, the_template);
}
inline Tensor Tensor::pow(Scalar exponent) const {
    return globalATenDispatch().getOp<Tensor (const Tensor &, Scalar)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 599, "pow")(*this, exponent);
}
inline Tensor & Tensor::zero_() {
    return globalATenDispatch().getOp<Tensor & (Tensor &)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 600, "zero_")(*this);
}
inline Tensor Tensor::sub(const Tensor & other, Scalar alpha) const {
    return globalATenDispatch().getOp<Tensor (const Tensor &, const Tensor &, Scalar)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 602, "sub")(*this, other, alpha);
}
inline Tensor & Tensor::sub_(const Tensor & other, Scalar alpha) {
    return globalATenDispatch().getOp<Tensor & (Tensor &, const Tensor &, Scalar)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 603, "sub_")(*this, other, alpha);
}
inline Tensor Tensor::sub(Scalar other, Scalar alpha) const {
    return globalATenDispatch().getOp<Tensor (const Tensor &, Scalar, Scalar)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 604, "sub")(*this, other, alpha);
}
inline Tensor & Tensor::sub_(Scalar other, Scalar alpha) {
    return globalATenDispatch().getOp<Tensor & (Tensor &, Scalar, Scalar)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 605, "sub_")(*this, other, alpha);
}
inline Tensor Tensor::addmm(const Tensor & mat1, const Tensor & mat2, Scalar beta, Scalar alpha) const {
    return globalATenDispatch().getOp<Tensor (const Tensor &, const Tensor &, const Tensor &, Scalar, Scalar)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 613, "addmm")(*this, mat1, mat2, beta, alpha);
}
inline Tensor & Tensor::addmm_(const Tensor & mat1, const Tensor & mat2, Scalar beta, Scalar alpha) {
    return globalATenDispatch().getOp<Tensor & (Tensor &, const Tensor &, const Tensor &, Scalar, Scalar)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 614, "addmm_")(*this, mat1, mat2, beta, alpha);
}
inline Tensor & Tensor::sparse_resize_(IntArrayRef size, int64_t sparse_dim, int64_t dense_dim) {
    return globalATenDispatch().getOp<Tensor & (Tensor &, IntArrayRef, int64_t, int64_t)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 621, "sparse_resize_")(*this, size, sparse_dim, dense_dim);
}
inline Tensor & Tensor::sparse_resize_and_clear_(IntArrayRef size, int64_t sparse_dim, int64_t dense_dim) {
    return globalATenDispatch().getOp<Tensor & (Tensor &, IntArrayRef, int64_t, int64_t)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 622, "sparse_resize_and_clear_")(*this, size, sparse_dim, dense_dim);
}
inline Tensor Tensor::sparse_mask(const Tensor & mask) const {
    return globalATenDispatch().getOp<Tensor (const Tensor &, const Tensor &)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 623, "sparse_mask")(*this, mask);
}
inline Tensor Tensor::to_dense() const {
    return globalATenDispatch().getOp<Tensor (const Tensor &)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 624, "to_dense")(*this);
}
inline int64_t Tensor::sparse_dim() const {
    return globalATenDispatch().getOp<int64_t (const Tensor &)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 626, "sparse_dim")(*this);
}
inline int64_t Tensor::_dimI() const {
    return globalATenDispatch().getOp<int64_t (const Tensor &)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 627, "_dimI")(*this);
}
inline int64_t Tensor::dense_dim() const {
    return globalATenDispatch().getOp<int64_t (const Tensor &)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 628, "dense_dim")(*this);
}
inline int64_t Tensor::_dimV() const {
    return globalATenDispatch().getOp<int64_t (const Tensor &)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 629, "_dimV")(*this);
}
inline int64_t Tensor::_nnz() const {
    return globalATenDispatch().getOp<int64_t (const Tensor &)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 630, "_nnz")(*this);
}
inline Tensor Tensor::coalesce() const {
    return globalATenDispatch().getOp<Tensor (const Tensor &)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 631, "coalesce")(*this);
}
inline bool Tensor::is_coalesced() const {
    return globalATenDispatch().getOp<bool (const Tensor &)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 632, "is_coalesced")(*this);
}
inline Tensor Tensor::_indices() const {
    return globalATenDispatch().getOp<Tensor (const Tensor &)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 633, "_indices")(*this);
}
inline Tensor Tensor::_values() const {
    return globalATenDispatch().getOp<Tensor (const Tensor &)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 634, "_values")(*this);
}
inline Tensor & Tensor::_coalesced_(bool coalesced) {
    return globalATenDispatch().getOp<Tensor & (Tensor &, bool)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 635, "_coalesced_")(*this, coalesced);
}
inline Tensor Tensor::indices() const {
    return globalATenDispatch().getOp<Tensor (const Tensor &)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 636, "indices")(*this);
}
inline Tensor Tensor::values() const {
    return globalATenDispatch().getOp<Tensor (const Tensor &)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 637, "values")(*this);
}
inline int64_t Tensor::numel() const {
    return globalATenDispatch().getOp<int64_t (const Tensor &)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 641, "numel")(*this);
}
inline std::vector<Tensor> Tensor::unbind(int64_t dim) const {
    return globalATenDispatch().getOp<std::vector<Tensor> (const Tensor &, int64_t)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 642, "unbind")(*this, dim);
}
inline Tensor Tensor::to_sparse(int64_t sparse_dim) const {
    return globalATenDispatch().getOp<Tensor (const Tensor &, int64_t)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 643, "to_sparse")(*this, sparse_dim);
}
inline Tensor Tensor::to_sparse() const {
    return globalATenDispatch().getOp<Tensor (const Tensor &)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 644, "to_sparse")(*this);
}
inline Tensor Tensor::to_mkldnn() const {
    return globalATenDispatch().getOp<Tensor (const Tensor &)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 645, "to_mkldnn")(*this);
}
inline Tensor Tensor::dequantize() const {
    return globalATenDispatch().getOp<Tensor (const Tensor &)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 650, "dequantize")(*this);
}
inline Scalar Tensor::q_scale() const {
    return globalATenDispatch().getOp<Scalar (const Tensor &)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 652, "q_scale")(*this);
}
inline Scalar Tensor::q_zero_point() const {
    return globalATenDispatch().getOp<Scalar (const Tensor &)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 653, "q_zero_point")(*this);
}
inline Tensor Tensor::int_repr() const {
    return globalATenDispatch().getOp<Tensor (const Tensor &)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 654, "int_repr")(*this);
}
inline Tensor Tensor::to(const TensorOptions & options, bool non_blocking, bool copy) const {
    return globalATenDispatch().getOp<Tensor (const Tensor &, const TensorOptions &, bool, bool)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 656, "to")(*this, options, non_blocking, copy);
}
inline Tensor Tensor::to(Device device, ScalarType dtype, bool non_blocking, bool copy) const {
    return globalATenDispatch().getOp<Tensor (const Tensor &, Device, ScalarType, bool, bool)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 657, "to")(*this, device, dtype, non_blocking, copy);
}
inline Tensor Tensor::to(ScalarType dtype, bool non_blocking, bool copy) const {
    return globalATenDispatch().getOp<Tensor (const Tensor &, ScalarType, bool, bool)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 658, "to")(*this, dtype, non_blocking, copy);
}
inline Tensor Tensor::to(const Tensor & other, bool non_blocking, bool copy) const {
    return globalATenDispatch().getOp<Tensor (const Tensor &, const Tensor &, bool, bool)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 659, "to")(*this, other, non_blocking, copy);
}
inline Scalar Tensor::item() const {
    return globalATenDispatch().getOp<Scalar (const Tensor &)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 663, "item")(*this);
}
inline Tensor & Tensor::set_(Storage source) {
    return globalATenDispatch().getOp<Tensor & (Tensor &, Storage)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 689, "set_")(*this, source);
}
inline Tensor & Tensor::set_(Storage source, int64_t storage_offset, IntArrayRef size, IntArrayRef stride) {
    return globalATenDispatch().getOp<Tensor & (Tensor &, Storage, int64_t, IntArrayRef, IntArrayRef)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 690, "set_")(*this, source, storage_offset, size, stride);
}
inline Tensor & Tensor::set_(const Tensor & source) {
    return globalATenDispatch().getOp<Tensor & (Tensor &, const Tensor &)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 691, "set_")(*this, source);
}
inline Tensor & Tensor::set_() {
    return globalATenDispatch().getOp<Tensor & (Tensor &)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 692, "set_")(*this);
}
inline bool Tensor::is_set_to(const Tensor & tensor) const {
    return globalATenDispatch().getOp<bool (const Tensor &, const Tensor &)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 693, "is_set_to")(*this, tensor);
}
inline Tensor & Tensor::masked_fill_(const Tensor & mask, Scalar value) {
    return globalATenDispatch().getOp<Tensor & (Tensor &, const Tensor &, Scalar)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 694, "masked_fill_")(*this, mask, value);
}
inline Tensor Tensor::masked_fill(const Tensor & mask, Scalar value) const {
    return globalATenDispatch().getOp<Tensor (const Tensor &, const Tensor &, Scalar)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 695, "masked_fill")(*this, mask, value);
}
inline Tensor & Tensor::masked_fill_(const Tensor & mask, const Tensor & value) {
    return globalATenDispatch().getOp<Tensor & (Tensor &, const Tensor &, const Tensor &)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 696, "masked_fill_")(*this, mask, value);
}
inline Tensor Tensor::masked_fill(const Tensor & mask, const Tensor & value) const {
    return globalATenDispatch().getOp<Tensor (const Tensor &, const Tensor &, const Tensor &)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 697, "masked_fill")(*this, mask, value);
}
inline Tensor & Tensor::masked_scatter_(const Tensor & mask, const Tensor & source) {
    return globalATenDispatch().getOp<Tensor & (Tensor &, const Tensor &, const Tensor &)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 698, "masked_scatter_")(*this, mask, source);
}
inline Tensor Tensor::masked_scatter(const Tensor & mask, const Tensor & source) const {
    return globalATenDispatch().getOp<Tensor (const Tensor &, const Tensor &, const Tensor &)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 699, "masked_scatter")(*this, mask, source);
}
inline Tensor Tensor::view(IntArrayRef size) const {
    return globalATenDispatch().getOp<Tensor (const Tensor &, IntArrayRef)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 700, "view")(*this, size);
}
inline Tensor & Tensor::put_(const Tensor & index, const Tensor & source, bool accumulate) {
    return globalATenDispatch().getOp<Tensor & (Tensor &, const Tensor &, const Tensor &, bool)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 701, "put_")(*this, index, source, accumulate);
}
inline Tensor & Tensor::index_add_(int64_t dim, const Tensor & index, const Tensor & source) {
    return globalATenDispatch().getOp<Tensor & (Tensor &, int64_t, const Tensor &, const Tensor &)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 702, "index_add_")(*this, dim, index, source);
}
inline Tensor Tensor::index_add(int64_t dim, const Tensor & index, const Tensor & source) const {
    return globalATenDispatch().getOp<Tensor (const Tensor &, int64_t, const Tensor &, const Tensor &)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 703, "index_add")(*this, dim, index, source);
}
inline Tensor & Tensor::index_fill_(int64_t dim, const Tensor & index, Scalar value) {
    return globalATenDispatch().getOp<Tensor & (Tensor &, int64_t, const Tensor &, Scalar)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 704, "index_fill_")(*this, dim, index, value);
}
inline Tensor Tensor::index_fill(int64_t dim, const Tensor & index, Scalar value) const {
    return globalATenDispatch().getOp<Tensor (const Tensor &, int64_t, const Tensor &, Scalar)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 705, "index_fill")(*this, dim, index, value);
}
inline Tensor & Tensor::index_fill_(int64_t dim, const Tensor & index, const Tensor & value) {
    return globalATenDispatch().getOp<Tensor & (Tensor &, int64_t, const Tensor &, const Tensor &)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 706, "index_fill_")(*this, dim, index, value);
}
inline Tensor Tensor::index_fill(int64_t dim, const Tensor & index, const Tensor & value) const {
    return globalATenDispatch().getOp<Tensor (const Tensor &, int64_t, const Tensor &, const Tensor &)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 707, "index_fill")(*this, dim, index, value);
}
inline Tensor & Tensor::scatter_(int64_t dim, const Tensor & index, const Tensor & src) {
    return globalATenDispatch().getOp<Tensor & (Tensor &, int64_t, const Tensor &, const Tensor &)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 708, "scatter_")(*this, dim, index, src);
}
inline Tensor Tensor::scatter(int64_t dim, const Tensor & index, const Tensor & src) const {
    return globalATenDispatch().getOp<Tensor (const Tensor &, int64_t, const Tensor &, const Tensor &)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 709, "scatter")(*this, dim, index, src);
}
inline Tensor & Tensor::scatter_(int64_t dim, const Tensor & index, Scalar value) {
    return globalATenDispatch().getOp<Tensor & (Tensor &, int64_t, const Tensor &, Scalar)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 710, "scatter_")(*this, dim, index, value);
}
inline Tensor Tensor::scatter(int64_t dim, const Tensor & index, Scalar value) const {
    return globalATenDispatch().getOp<Tensor (const Tensor &, int64_t, const Tensor &, Scalar)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 711, "scatter")(*this, dim, index, value);
}
inline Tensor & Tensor::scatter_add_(int64_t dim, const Tensor & index, const Tensor & src) {
    return globalATenDispatch().getOp<Tensor & (Tensor &, int64_t, const Tensor &, const Tensor &)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 712, "scatter_add_")(*this, dim, index, src);
}
inline Tensor Tensor::scatter_add(int64_t dim, const Tensor & index, const Tensor & src) const {
    return globalATenDispatch().getOp<Tensor (const Tensor &, int64_t, const Tensor &, const Tensor &)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 713, "scatter_add")(*this, dim, index, src);
}
inline Tensor & Tensor::lt_(Scalar other) {
    return globalATenDispatch().getOp<Tensor & (Tensor &, Scalar)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 714, "lt_")(*this, other);
}
inline Tensor & Tensor::lt_(const Tensor & other) {
    return globalATenDispatch().getOp<Tensor & (Tensor &, const Tensor &)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 715, "lt_")(*this, other);
}
inline Tensor & Tensor::gt_(Scalar other) {
    return globalATenDispatch().getOp<Tensor & (Tensor &, Scalar)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 716, "gt_")(*this, other);
}
inline Tensor & Tensor::gt_(const Tensor & other) {
    return globalATenDispatch().getOp<Tensor & (Tensor &, const Tensor &)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 717, "gt_")(*this, other);
}
inline Tensor & Tensor::le_(Scalar other) {
    return globalATenDispatch().getOp<Tensor & (Tensor &, Scalar)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 718, "le_")(*this, other);
}
inline Tensor & Tensor::le_(const Tensor & other) {
    return globalATenDispatch().getOp<Tensor & (Tensor &, const Tensor &)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 719, "le_")(*this, other);
}
inline Tensor & Tensor::ge_(Scalar other) {
    return globalATenDispatch().getOp<Tensor & (Tensor &, Scalar)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 720, "ge_")(*this, other);
}
inline Tensor & Tensor::ge_(const Tensor & other) {
    return globalATenDispatch().getOp<Tensor & (Tensor &, const Tensor &)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 721, "ge_")(*this, other);
}
inline Tensor & Tensor::eq_(Scalar other) {
    return globalATenDispatch().getOp<Tensor & (Tensor &, Scalar)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 722, "eq_")(*this, other);
}
inline Tensor & Tensor::eq_(const Tensor & other) {
    return globalATenDispatch().getOp<Tensor & (Tensor &, const Tensor &)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 723, "eq_")(*this, other);
}
inline Tensor & Tensor::ne_(Scalar other) {
    return globalATenDispatch().getOp<Tensor & (Tensor &, Scalar)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 724, "ne_")(*this, other);
}
inline Tensor & Tensor::ne_(const Tensor & other) {
    return globalATenDispatch().getOp<Tensor & (Tensor &, const Tensor &)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 725, "ne_")(*this, other);
}
inline Tensor Tensor::__and__(Scalar other) const {
    return globalATenDispatch().getOp<Tensor (const Tensor &, Scalar)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 726, "__and__")(*this, other);
}
inline Tensor Tensor::__and__(const Tensor & other) const {
    return globalATenDispatch().getOp<Tensor (const Tensor &, const Tensor &)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 727, "__and__")(*this, other);
}
inline Tensor & Tensor::__iand__(Scalar other) {
    return globalATenDispatch().getOp<Tensor & (Tensor &, Scalar)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 728, "__iand__")(*this, other);
}
inline Tensor & Tensor::__iand__(const Tensor & other) {
    return globalATenDispatch().getOp<Tensor & (Tensor &, const Tensor &)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 729, "__iand__")(*this, other);
}
inline Tensor Tensor::__or__(Scalar other) const {
    return globalATenDispatch().getOp<Tensor (const Tensor &, Scalar)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 730, "__or__")(*this, other);
}
inline Tensor Tensor::__or__(const Tensor & other) const {
    return globalATenDispatch().getOp<Tensor (const Tensor &, const Tensor &)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 731, "__or__")(*this, other);
}
inline Tensor & Tensor::__ior__(Scalar other) {
    return globalATenDispatch().getOp<Tensor & (Tensor &, Scalar)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 732, "__ior__")(*this, other);
}
inline Tensor & Tensor::__ior__(const Tensor & other) {
    return globalATenDispatch().getOp<Tensor & (Tensor &, const Tensor &)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 733, "__ior__")(*this, other);
}
inline Tensor Tensor::__xor__(Scalar other) const {
    return globalATenDispatch().getOp<Tensor (const Tensor &, Scalar)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 734, "__xor__")(*this, other);
}
inline Tensor Tensor::__xor__(const Tensor & other) const {
    return globalATenDispatch().getOp<Tensor (const Tensor &, const Tensor &)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 735, "__xor__")(*this, other);
}
inline Tensor & Tensor::__ixor__(Scalar other) {
    return globalATenDispatch().getOp<Tensor & (Tensor &, Scalar)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 736, "__ixor__")(*this, other);
}
inline Tensor & Tensor::__ixor__(const Tensor & other) {
    return globalATenDispatch().getOp<Tensor & (Tensor &, const Tensor &)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 737, "__ixor__")(*this, other);
}
inline Tensor Tensor::__lshift__(Scalar other) const {
    return globalATenDispatch().getOp<Tensor (const Tensor &, Scalar)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 738, "__lshift__")(*this, other);
}
inline Tensor Tensor::__lshift__(const Tensor & other) const {
    return globalATenDispatch().getOp<Tensor (const Tensor &, const Tensor &)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 739, "__lshift__")(*this, other);
}
inline Tensor & Tensor::__ilshift__(Scalar other) {
    return globalATenDispatch().getOp<Tensor & (Tensor &, Scalar)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 740, "__ilshift__")(*this, other);
}
inline Tensor & Tensor::__ilshift__(const Tensor & other) {
    return globalATenDispatch().getOp<Tensor & (Tensor &, const Tensor &)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 741, "__ilshift__")(*this, other);
}
inline Tensor Tensor::__rshift__(Scalar other) const {
    return globalATenDispatch().getOp<Tensor (const Tensor &, Scalar)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 742, "__rshift__")(*this, other);
}
inline Tensor Tensor::__rshift__(const Tensor & other) const {
    return globalATenDispatch().getOp<Tensor (const Tensor &, const Tensor &)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 743, "__rshift__")(*this, other);
}
inline Tensor & Tensor::__irshift__(Scalar other) {
    return globalATenDispatch().getOp<Tensor & (Tensor &, Scalar)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 744, "__irshift__")(*this, other);
}
inline Tensor & Tensor::__irshift__(const Tensor & other) {
    return globalATenDispatch().getOp<Tensor & (Tensor &, const Tensor &)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 745, "__irshift__")(*this, other);
}
inline Tensor & Tensor::lgamma_() {
    return globalATenDispatch().getOp<Tensor & (Tensor &)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 746, "lgamma_")(*this);
}
inline Tensor & Tensor::atan2_(const Tensor & other) {
    return globalATenDispatch().getOp<Tensor & (Tensor &, const Tensor &)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 747, "atan2_")(*this, other);
}
inline Tensor & Tensor::tril_(int64_t diagonal) {
    return globalATenDispatch().getOp<Tensor & (Tensor &, int64_t)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 748, "tril_")(*this, diagonal);
}
inline Tensor & Tensor::triu_(int64_t diagonal) {
    return globalATenDispatch().getOp<Tensor & (Tensor &, int64_t)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 749, "triu_")(*this, diagonal);
}
inline Tensor & Tensor::digamma_() {
    return globalATenDispatch().getOp<Tensor & (Tensor &)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 750, "digamma_")(*this);
}
inline Tensor & Tensor::polygamma_(int64_t n) {
    return globalATenDispatch().getOp<Tensor & (Tensor &, int64_t)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 751, "polygamma_")(*this, n);
}
inline Tensor & Tensor::erfinv_() {
    return globalATenDispatch().getOp<Tensor & (Tensor &)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 752, "erfinv_")(*this);
}
inline Tensor & Tensor::renorm_(Scalar p, int64_t dim, Scalar maxnorm) {
    return globalATenDispatch().getOp<Tensor & (Tensor &, Scalar, int64_t, Scalar)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 753, "renorm_")(*this, p, dim, maxnorm);
}
inline Tensor & Tensor::pow_(Scalar exponent) {
    return globalATenDispatch().getOp<Tensor & (Tensor &, Scalar)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 754, "pow_")(*this, exponent);
}
inline Tensor & Tensor::pow_(const Tensor & exponent) {
    return globalATenDispatch().getOp<Tensor & (Tensor &, const Tensor &)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 755, "pow_")(*this, exponent);
}
inline Tensor & Tensor::lerp_(const Tensor & end, Scalar weight) {
    return globalATenDispatch().getOp<Tensor & (Tensor &, const Tensor &, Scalar)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 756, "lerp_")(*this, end, weight);
}
inline Tensor & Tensor::lerp_(const Tensor & end, const Tensor & weight) {
    return globalATenDispatch().getOp<Tensor & (Tensor &, const Tensor &, const Tensor &)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 757, "lerp_")(*this, end, weight);
}
inline Tensor & Tensor::sign_() {
    return globalATenDispatch().getOp<Tensor & (Tensor &)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 758, "sign_")(*this);
}
inline Tensor & Tensor::fmod_(Scalar other) {
    return globalATenDispatch().getOp<Tensor & (Tensor &, Scalar)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 759, "fmod_")(*this, other);
}
inline Tensor & Tensor::fmod_(const Tensor & other) {
    return globalATenDispatch().getOp<Tensor & (Tensor &, const Tensor &)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 760, "fmod_")(*this, other);
}
inline Tensor & Tensor::remainder_(Scalar other) {
    return globalATenDispatch().getOp<Tensor & (Tensor &, Scalar)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 761, "remainder_")(*this, other);
}
inline Tensor & Tensor::remainder_(const Tensor & other) {
    return globalATenDispatch().getOp<Tensor & (Tensor &, const Tensor &)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 762, "remainder_")(*this, other);
}
inline Tensor & Tensor::addbmm_(const Tensor & batch1, const Tensor & batch2, Scalar beta, Scalar alpha) {
    return globalATenDispatch().getOp<Tensor & (Tensor &, const Tensor &, const Tensor &, Scalar, Scalar)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 763, "addbmm_")(*this, batch1, batch2, beta, alpha);
}
inline Tensor Tensor::addbmm(const Tensor & batch1, const Tensor & batch2, Scalar beta, Scalar alpha) const {
    return globalATenDispatch().getOp<Tensor (const Tensor &, const Tensor &, const Tensor &, Scalar, Scalar)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 765, "addbmm")(*this, batch1, batch2, beta, alpha);
}
inline Tensor & Tensor::addcmul_(const Tensor & tensor1, const Tensor & tensor2, Scalar value) {
    return globalATenDispatch().getOp<Tensor & (Tensor &, const Tensor &, const Tensor &, Scalar)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 766, "addcmul_")(*this, tensor1, tensor2, value);
}
inline Tensor & Tensor::addcdiv_(const Tensor & tensor1, const Tensor & tensor2, Scalar value) {
    return globalATenDispatch().getOp<Tensor & (Tensor &, const Tensor &, const Tensor &, Scalar)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 767, "addcdiv_")(*this, tensor1, tensor2, value);
}
inline Tensor & Tensor::random_(int64_t from, int64_t to, Generator * generator) {
    return globalATenDispatch().getOp<Tensor & (Tensor &, int64_t, int64_t, Generator *)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 768, "random_")(*this, from, to, generator);
}
inline Tensor & Tensor::random_(int64_t to, Generator * generator) {
    return globalATenDispatch().getOp<Tensor & (Tensor &, int64_t, Generator *)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 769, "random_")(*this, to, generator);
}
inline Tensor & Tensor::random_(Generator * generator) {
    return globalATenDispatch().getOp<Tensor & (Tensor &, Generator *)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 770, "random_")(*this, generator);
}
inline Tensor & Tensor::uniform_(double from, double to, Generator * generator) {
    return globalATenDispatch().getOp<Tensor & (Tensor &, double, double, Generator *)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 771, "uniform_")(*this, from, to, generator);
}
inline Tensor & Tensor::normal_(double mean, double std, Generator * generator) {
    return globalATenDispatch().getOp<Tensor & (Tensor &, double, double, Generator *)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 772, "normal_")(*this, mean, std, generator);
}
inline Tensor & Tensor::cauchy_(double median, double sigma, Generator * generator) {
    return globalATenDispatch().getOp<Tensor & (Tensor &, double, double, Generator *)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 773, "cauchy_")(*this, median, sigma, generator);
}
inline Tensor & Tensor::log_normal_(double mean, double std, Generator * generator) {
    return globalATenDispatch().getOp<Tensor & (Tensor &, double, double, Generator *)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 774, "log_normal_")(*this, mean, std, generator);
}
inline Tensor & Tensor::exponential_(double lambd, Generator * generator) {
    return globalATenDispatch().getOp<Tensor & (Tensor &, double, Generator *)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 775, "exponential_")(*this, lambd, generator);
}
inline Tensor & Tensor::geometric_(double p, Generator * generator) {
    return globalATenDispatch().getOp<Tensor & (Tensor &, double, Generator *)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 776, "geometric_")(*this, p, generator);
}
inline Tensor Tensor::diag(int64_t diagonal) const {
    return globalATenDispatch().getOp<Tensor (const Tensor &, int64_t)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 778, "diag")(*this, diagonal);
}
inline Tensor Tensor::cross(const Tensor & other, c10::optional<int64_t> dim) const {
    return globalATenDispatch().getOp<Tensor (const Tensor &, const Tensor &, c10::optional<int64_t>)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 780, "cross")(*this, other, dim);
}
inline Tensor Tensor::triu(int64_t diagonal) const {
    return globalATenDispatch().getOp<Tensor (const Tensor &, int64_t)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 782, "triu")(*this, diagonal);
}
inline Tensor Tensor::tril(int64_t diagonal) const {
    return globalATenDispatch().getOp<Tensor (const Tensor &, int64_t)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 784, "tril")(*this, diagonal);
}
inline Tensor Tensor::trace() const {
    return globalATenDispatch().getOp<Tensor (const Tensor &)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 787, "trace")(*this);
}
inline Tensor Tensor::ne(Scalar other) const {
    return globalATenDispatch().getOp<Tensor (const Tensor &, Scalar)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 789, "ne")(*this, other);
}
inline Tensor Tensor::ne(const Tensor & other) const {
    return globalATenDispatch().getOp<Tensor (const Tensor &, const Tensor &)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 791, "ne")(*this, other);
}
inline Tensor Tensor::eq(Scalar other) const {
    return globalATenDispatch().getOp<Tensor (const Tensor &, Scalar)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 793, "eq")(*this, other);
}
inline Tensor Tensor::eq(const Tensor & other) const {
    return globalATenDispatch().getOp<Tensor (const Tensor &, const Tensor &)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 795, "eq")(*this, other);
}
inline Tensor Tensor::ge(Scalar other) const {
    return globalATenDispatch().getOp<Tensor (const Tensor &, Scalar)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 797, "ge")(*this, other);
}
inline Tensor Tensor::ge(const Tensor & other) const {
    return globalATenDispatch().getOp<Tensor (const Tensor &, const Tensor &)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 799, "ge")(*this, other);
}
inline Tensor Tensor::le(Scalar other) const {
    return globalATenDispatch().getOp<Tensor (const Tensor &, Scalar)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 801, "le")(*this, other);
}
inline Tensor Tensor::le(const Tensor & other) const {
    return globalATenDispatch().getOp<Tensor (const Tensor &, const Tensor &)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 803, "le")(*this, other);
}
inline Tensor Tensor::gt(Scalar other) const {
    return globalATenDispatch().getOp<Tensor (const Tensor &, Scalar)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 805, "gt")(*this, other);
}
inline Tensor Tensor::gt(const Tensor & other) const {
    return globalATenDispatch().getOp<Tensor (const Tensor &, const Tensor &)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 807, "gt")(*this, other);
}
inline Tensor Tensor::lt(Scalar other) const {
    return globalATenDispatch().getOp<Tensor (const Tensor &, Scalar)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 809, "lt")(*this, other);
}
inline Tensor Tensor::lt(const Tensor & other) const {
    return globalATenDispatch().getOp<Tensor (const Tensor &, const Tensor &)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 811, "lt")(*this, other);
}
inline Tensor Tensor::take(const Tensor & index) const {
    return globalATenDispatch().getOp<Tensor (const Tensor &, const Tensor &)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 813, "take")(*this, index);
}
inline Tensor Tensor::index_select(int64_t dim, const Tensor & index) const {
    return globalATenDispatch().getOp<Tensor (const Tensor &, int64_t, const Tensor &)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 815, "index_select")(*this, dim, index);
}
inline Tensor Tensor::masked_select(const Tensor & mask) const {
    return globalATenDispatch().getOp<Tensor (const Tensor &, const Tensor &)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 817, "masked_select")(*this, mask);
}
inline Tensor Tensor::nonzero() const {
    return globalATenDispatch().getOp<Tensor (const Tensor &)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 819, "nonzero")(*this);
}
inline Tensor Tensor::gather(int64_t dim, const Tensor & index, bool sparse_grad) const {
    return globalATenDispatch().getOp<Tensor (const Tensor &, int64_t, const Tensor &, bool)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 821, "gather")(*this, dim, index, sparse_grad);
}
inline Tensor Tensor::addcmul(const Tensor & tensor1, const Tensor & tensor2, Scalar value) const {
    return globalATenDispatch().getOp<Tensor (const Tensor &, const Tensor &, const Tensor &, Scalar)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 824, "addcmul")(*this, tensor1, tensor2, value);
}
inline Tensor Tensor::addcdiv(const Tensor & tensor1, const Tensor & tensor2, Scalar value) const {
    return globalATenDispatch().getOp<Tensor (const Tensor &, const Tensor &, const Tensor &, Scalar)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 826, "addcdiv")(*this, tensor1, tensor2, value);
}
inline std::tuple<Tensor,Tensor> Tensor::gels(const Tensor & A) const {
    return globalATenDispatch().getOp<std::tuple<Tensor,Tensor> (const Tensor &, const Tensor &)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 828, "gels")(*this, A);
}
inline std::tuple<Tensor,Tensor> Tensor::triangular_solve(const Tensor & A, bool upper, bool transpose, bool unitriangular) const {
    return globalATenDispatch().getOp<std::tuple<Tensor,Tensor> (const Tensor &, const Tensor &, bool, bool, bool)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 830, "triangular_solve")(*this, A, upper, transpose, unitriangular);
}
inline std::tuple<Tensor,Tensor> Tensor::symeig(bool eigenvectors, bool upper) const {
    return globalATenDispatch().getOp<std::tuple<Tensor,Tensor> (const Tensor &, bool, bool)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 833, "symeig")(*this, eigenvectors, upper);
}
inline std::tuple<Tensor,Tensor> Tensor::eig(bool eigenvectors) const {
    return globalATenDispatch().getOp<std::tuple<Tensor,Tensor> (const Tensor &, bool)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 835, "eig")(*this, eigenvectors);
}
inline std::tuple<Tensor,Tensor,Tensor> Tensor::svd(bool some, bool compute_uv) const {
    return globalATenDispatch().getOp<std::tuple<Tensor,Tensor,Tensor> (const Tensor &, bool, bool)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 837, "svd")(*this, some, compute_uv);
}
inline Tensor Tensor::cholesky(bool upper) const {
    return globalATenDispatch().getOp<Tensor (const Tensor &, bool)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 839, "cholesky")(*this, upper);
}
inline Tensor Tensor::cholesky_solve(const Tensor & input2, bool upper) const {
    return globalATenDispatch().getOp<Tensor (const Tensor &, const Tensor &, bool)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 842, "cholesky_solve")(*this, input2, upper);
}
inline std::tuple<Tensor,Tensor> Tensor::solve(const Tensor & A) const {
    return globalATenDispatch().getOp<std::tuple<Tensor,Tensor> (const Tensor &, const Tensor &)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 844, "solve")(*this, A);
}
inline Tensor Tensor::cholesky_inverse(bool upper) const {
    return globalATenDispatch().getOp<Tensor (const Tensor &, bool)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 848, "cholesky_inverse")(*this, upper);
}
inline std::tuple<Tensor,Tensor> Tensor::pstrf(bool upper, Scalar tol) const {
    return globalATenDispatch().getOp<std::tuple<Tensor,Tensor> (const Tensor &, bool, Scalar)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 850, "pstrf")(*this, upper, tol);
}
inline std::tuple<Tensor,Tensor> Tensor::qr(bool some) const {
    return globalATenDispatch().getOp<std::tuple<Tensor,Tensor> (const Tensor &, bool)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 852, "qr")(*this, some);
}
inline std::tuple<Tensor,Tensor> Tensor::geqrf() const {
    return globalATenDispatch().getOp<std::tuple<Tensor,Tensor> (const Tensor &)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 855, "geqrf")(*this);
}
inline Tensor Tensor::orgqr(const Tensor & input2) const {
    return globalATenDispatch().getOp<Tensor (const Tensor &, const Tensor &)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 857, "orgqr")(*this, input2);
}
inline Tensor Tensor::ormqr(const Tensor & input2, const Tensor & input3, bool left, bool transpose) const {
    return globalATenDispatch().getOp<Tensor (const Tensor &, const Tensor &, const Tensor &, bool, bool)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 859, "ormqr")(*this, input2, input3, left, transpose);
}
inline Tensor Tensor::lu_solve(const Tensor & LU_data, const Tensor & LU_pivots) const {
    return globalATenDispatch().getOp<Tensor (const Tensor &, const Tensor &, const Tensor &)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 862, "lu_solve")(*this, LU_data, LU_pivots);
}
inline Tensor Tensor::multinomial(int64_t num_samples, bool replacement, Generator * generator) const {
    return globalATenDispatch().getOp<Tensor (const Tensor &, int64_t, bool, Generator *)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 864, "multinomial")(*this, num_samples, replacement, generator);
}
inline Tensor Tensor::lgamma() const {
    return globalATenDispatch().getOp<Tensor (const Tensor &)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 868, "lgamma")(*this);
}
inline Tensor Tensor::digamma() const {
    return globalATenDispatch().getOp<Tensor (const Tensor &)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 870, "digamma")(*this);
}
inline Tensor Tensor::polygamma(int64_t n) const {
    return globalATenDispatch().getOp<Tensor (int64_t, const Tensor &)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 872, "polygamma")(n, *this);
}
inline Tensor Tensor::erfinv() const {
    return globalATenDispatch().getOp<Tensor (const Tensor &)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 874, "erfinv")(*this);
}
inline Tensor Tensor::dist(const Tensor & other, Scalar p) const {
    return globalATenDispatch().getOp<Tensor (const Tensor &, const Tensor &, Scalar)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 875, "dist")(*this, other, p);
}
inline Tensor Tensor::atan2(const Tensor & other) const {
    return globalATenDispatch().getOp<Tensor (const Tensor &, const Tensor &)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 877, "atan2")(*this, other);
}
inline Tensor Tensor::lerp(const Tensor & end, Scalar weight) const {
    return globalATenDispatch().getOp<Tensor (const Tensor &, const Tensor &, Scalar)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 880, "lerp")(*this, end, weight);
}
inline Tensor Tensor::lerp(const Tensor & end, const Tensor & weight) const {
    return globalATenDispatch().getOp<Tensor (const Tensor &, const Tensor &, const Tensor &)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 881, "lerp")(*this, end, weight);
}
inline Tensor Tensor::histc(int64_t bins, Scalar min, Scalar max) const {
    return globalATenDispatch().getOp<Tensor (const Tensor &, int64_t, Scalar, Scalar)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 883, "histc")(*this, bins, min, max);
}
inline Tensor Tensor::sign() const {
    return globalATenDispatch().getOp<Tensor (const Tensor &)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 885, "sign")(*this);
}
inline Tensor Tensor::fmod(Scalar other) const {
    return globalATenDispatch().getOp<Tensor (const Tensor &, Scalar)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 887, "fmod")(*this, other);
}
inline Tensor Tensor::fmod(const Tensor & other) const {
    return globalATenDispatch().getOp<Tensor (const Tensor &, const Tensor &)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 889, "fmod")(*this, other);
}
inline Tensor Tensor::remainder(Scalar other) const {
    return globalATenDispatch().getOp<Tensor (const Tensor &, Scalar)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 891, "remainder")(*this, other);
}
inline Tensor Tensor::remainder(const Tensor & other) const {
    return globalATenDispatch().getOp<Tensor (const Tensor &, const Tensor &)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 893, "remainder")(*this, other);
}
inline Tensor Tensor::min(const Tensor & other) const {
    return globalATenDispatch().getOp<Tensor (const Tensor &, const Tensor &)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 895, "min")(*this, other);
}
inline Tensor Tensor::min() const {
    return globalATenDispatch().getOp<Tensor (const Tensor &)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 896, "min")(*this);
}
inline Tensor Tensor::max(const Tensor & other) const {
    return globalATenDispatch().getOp<Tensor (const Tensor &, const Tensor &)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 898, "max")(*this, other);
}
inline Tensor Tensor::max() const {
    return globalATenDispatch().getOp<Tensor (const Tensor &)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 899, "max")(*this);
}
inline Tensor Tensor::median() const {
    return globalATenDispatch().getOp<Tensor (const Tensor &)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 900, "median")(*this);
}
inline std::tuple<Tensor,Tensor> Tensor::sort(int64_t dim, bool descending) const {
    return globalATenDispatch().getOp<std::tuple<Tensor,Tensor> (const Tensor &, int64_t, bool)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 902, "sort")(*this, dim, descending);
}
inline Tensor Tensor::argsort(int64_t dim, bool descending) const {
    return globalATenDispatch().getOp<Tensor (const Tensor &, int64_t, bool)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 903, "argsort")(*this, dim, descending);
}
inline std::tuple<Tensor,Tensor> Tensor::topk(int64_t k, int64_t dim, bool largest, bool sorted) const {
    return globalATenDispatch().getOp<std::tuple<Tensor,Tensor> (const Tensor &, int64_t, int64_t, bool, bool)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 905, "topk")(*this, k, dim, largest, sorted);
}
inline Tensor Tensor::all() const {
    return globalATenDispatch().getOp<Tensor (const Tensor &)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 906, "all")(*this);
}
inline Tensor Tensor::any() const {
    return globalATenDispatch().getOp<Tensor (const Tensor &)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 907, "any")(*this);
}
inline Tensor Tensor::renorm(Scalar p, int64_t dim, Scalar maxnorm) const {
    return globalATenDispatch().getOp<Tensor (const Tensor &, Scalar, int64_t, Scalar)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 909, "renorm")(*this, p, dim, maxnorm);
}
inline Tensor Tensor::unfold(int64_t dimension, int64_t size, int64_t step) const {
    return globalATenDispatch().getOp<Tensor (const Tensor &, int64_t, int64_t, int64_t)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 910, "unfold")(*this, dimension, size, step);
}
inline bool Tensor::equal(const Tensor & other) const {
    return globalATenDispatch().getOp<bool (const Tensor &, const Tensor &)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 911, "equal")(*this, other);
}
inline Tensor Tensor::pow(const Tensor & exponent) const {
    return globalATenDispatch().getOp<Tensor (const Tensor &, const Tensor &)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 913, "pow")(*this, exponent);
}
inline Tensor Tensor::alias() const {
    return globalATenDispatch().getOp<Tensor (const Tensor &)>(
        tensorTypeIdToBackend(type_id()), is_variable(), 922, "alias")(*this);
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
