#pragma once

#include <ATen/core/Tensor.h>
#include <c10/core/Scalar.h>
#include <c10/macros/Macros.h>
#include <ATen/core/SparseTensorRef.h>
#include <ATen/core/Type.h>
#include <c10/core/TensorOptions.h>

namespace at {

inline Tensor Tensor::toType(const Type & t, bool non_blocking) const {
  if(type() == t)
    return *this;
  return t.copy(*this, non_blocking);
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

inline Tensor & Tensor::copy_(const Tensor & src, bool non_blocking) {
  return type().copy_(*this, src, non_blocking);
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
  type().backward(*this, std::move(gradient), keep_graph, create_graph);
}

inline void Tensor::set_data(Tensor new_data) {
  type().set_data(*this, new_data);
}

// all static inline to allow for inlining of the non-dynamic part of dispatch
inline Tensor Tensor::abs() const {
    return type().abs(*this);
}
inline Tensor & Tensor::abs_() {
    return type().abs_(*this);
}
inline Tensor Tensor::acos() const {
    return type().acos(*this);
}
inline Tensor & Tensor::acos_() {
    return type().acos_(*this);
}
inline Tensor Tensor::add(const Tensor & other, Scalar alpha) const {
    return type().add(*this, other, alpha);
}
inline Tensor & Tensor::add_(const Tensor & other, Scalar alpha) {
    return type().add_(*this, other, alpha);
}
inline Tensor Tensor::add(Scalar other, Scalar alpha) const {
    return type().add(*this, other, alpha);
}
inline Tensor & Tensor::add_(Scalar other, Scalar alpha) {
    return type().add_(*this, other, alpha);
}
inline Tensor Tensor::addmv(const Tensor & mat, const Tensor & vec, Scalar beta, Scalar alpha) const {
    return type().addmv(*this, mat, vec, beta, alpha);
}
inline Tensor & Tensor::addmv_(const Tensor & mat, const Tensor & vec, Scalar beta, Scalar alpha) {
    return type().addmv_(*this, mat, vec, beta, alpha);
}
inline Tensor Tensor::addr(const Tensor & vec1, const Tensor & vec2, Scalar beta, Scalar alpha) const {
    return type().addr(*this, vec1, vec2, beta, alpha);
}
inline Tensor & Tensor::addr_(const Tensor & vec1, const Tensor & vec2, Scalar beta, Scalar alpha) {
    return type().addr_(*this, vec1, vec2, beta, alpha);
}
inline Tensor Tensor::all(int64_t dim, bool keepdim) const {
    return type().all(*this, dim, keepdim);
}
inline bool Tensor::allclose(const Tensor & other, double rtol, double atol, bool equal_nan) const {
    return type().allclose(*this, other, rtol, atol, equal_nan);
}
inline Tensor Tensor::any(int64_t dim, bool keepdim) const {
    return type().any(*this, dim, keepdim);
}
inline Tensor Tensor::argmax(int64_t dim, bool keepdim) const {
    return type().argmax(*this, dim, keepdim);
}
inline Tensor Tensor::argmax() const {
    return type().argmax(*this);
}
inline Tensor Tensor::argmin(int64_t dim, bool keepdim) const {
    return type().argmin(*this, dim, keepdim);
}
inline Tensor Tensor::argmin() const {
    return type().argmin(*this);
}
inline Tensor Tensor::as_strided(IntArrayRef size, IntArrayRef stride, c10::optional<int64_t> storage_offset) const {
    return type().as_strided(*this, size, stride, storage_offset);
}
inline Tensor & Tensor::as_strided_(IntArrayRef size, IntArrayRef stride, c10::optional<int64_t> storage_offset) {
    return type().as_strided_(*this, size, stride, storage_offset);
}
inline Tensor Tensor::asin() const {
    return type().asin(*this);
}
inline Tensor & Tensor::asin_() {
    return type().asin_(*this);
}
inline Tensor Tensor::atan() const {
    return type().atan(*this);
}
inline Tensor & Tensor::atan_() {
    return type().atan_(*this);
}
inline Tensor Tensor::baddbmm(const Tensor & batch1, const Tensor & batch2, Scalar beta, Scalar alpha) const {
    return type().baddbmm(*this, batch1, batch2, beta, alpha);
}
inline Tensor & Tensor::baddbmm_(const Tensor & batch1, const Tensor & batch2, Scalar beta, Scalar alpha) {
    return type().baddbmm_(*this, batch1, batch2, beta, alpha);
}
inline Tensor Tensor::bernoulli(Generator * generator) const {
    return type().bernoulli(*this, generator);
}
inline Tensor & Tensor::bernoulli_(const Tensor & p, Generator * generator) {
    return type().bernoulli_(*this, p, generator);
}
inline Tensor & Tensor::bernoulli_(double p, Generator * generator) {
    return type().bernoulli_(*this, p, generator);
}
inline Tensor Tensor::bernoulli(double p, Generator * generator) const {
    return type().bernoulli(*this, p, generator);
}
inline Tensor Tensor::bincount(const Tensor & weights, int64_t minlength) const {
    return type().bincount(*this, weights, minlength);
}
inline Tensor Tensor::bmm(const Tensor & mat2) const {
    return type().bmm(*this, mat2);
}
inline Tensor Tensor::ceil() const {
    return type().ceil(*this);
}
inline Tensor & Tensor::ceil_() {
    return type().ceil_(*this);
}
inline std::vector<Tensor> Tensor::chunk(int64_t chunks, int64_t dim) const {
    return type().chunk(*this, chunks, dim);
}
inline Tensor Tensor::clamp(c10::optional<Scalar> min, c10::optional<Scalar> max) const {
    return type().clamp(*this, min, max);
}
inline Tensor & Tensor::clamp_(c10::optional<Scalar> min, c10::optional<Scalar> max) {
    return type().clamp_(*this, min, max);
}
inline Tensor Tensor::clamp_max(Scalar max) const {
    return type().clamp_max(*this, max);
}
inline Tensor & Tensor::clamp_max_(Scalar max) {
    return type().clamp_max_(*this, max);
}
inline Tensor Tensor::clamp_min(Scalar min) const {
    return type().clamp_min(*this, min);
}
inline Tensor & Tensor::clamp_min_(Scalar min) {
    return type().clamp_min_(*this, min);
}
inline Tensor Tensor::contiguous() const {
    return type().contiguous(*this);
}
inline Tensor Tensor::cos() const {
    return type().cos(*this);
}
inline Tensor & Tensor::cos_() {
    return type().cos_(*this);
}
inline Tensor Tensor::cosh() const {
    return type().cosh(*this);
}
inline Tensor & Tensor::cosh_() {
    return type().cosh_(*this);
}
inline Tensor Tensor::cumsum(int64_t dim, ScalarType dtype) const {
    return type().cumsum(*this, dim, dtype);
}
inline Tensor Tensor::cumsum(int64_t dim) const {
    return type().cumsum(*this, dim);
}
inline Tensor Tensor::cumprod(int64_t dim, ScalarType dtype) const {
    return type().cumprod(*this, dim, dtype);
}
inline Tensor Tensor::cumprod(int64_t dim) const {
    return type().cumprod(*this, dim);
}
inline Tensor Tensor::det() const {
    return type().det(*this);
}
inline Tensor Tensor::diag_embed(int64_t offset, int64_t dim1, int64_t dim2) const {
    return type().diag_embed(*this, offset, dim1, dim2);
}
inline Tensor Tensor::diagflat(int64_t offset) const {
    return type().diagflat(*this, offset);
}
inline Tensor Tensor::diagonal(int64_t offset, int64_t dim1, int64_t dim2) const {
    return type().diagonal(*this, offset, dim1, dim2);
}
inline Tensor Tensor::div(const Tensor & other) const {
    return type().div(*this, other);
}
inline Tensor & Tensor::div_(const Tensor & other) {
    return type().div_(*this, other);
}
inline Tensor Tensor::div(Scalar other) const {
    return type().div(*this, other);
}
inline Tensor & Tensor::div_(Scalar other) {
    return type().div_(*this, other);
}
inline Tensor Tensor::dot(const Tensor & tensor) const {
    return type().dot(*this, tensor);
}
inline Tensor & Tensor::resize_(IntArrayRef size) {
    return type().resize_(*this, size);
}
inline Tensor Tensor::erf() const {
    return type().erf(*this);
}
inline Tensor & Tensor::erf_() {
    return type().erf_(*this);
}
inline Tensor Tensor::erfc() const {
    return type().erfc(*this);
}
inline Tensor & Tensor::erfc_() {
    return type().erfc_(*this);
}
inline Tensor Tensor::exp() const {
    return type().exp(*this);
}
inline Tensor & Tensor::exp_() {
    return type().exp_(*this);
}
inline Tensor Tensor::expm1() const {
    return type().expm1(*this);
}
inline Tensor & Tensor::expm1_() {
    return type().expm1_(*this);
}
inline Tensor Tensor::expand(IntArrayRef size, bool implicit) const {
    return type().expand(*this, size, implicit);
}
inline Tensor Tensor::expand_as(const Tensor & other) const {
    return type().expand_as(*this, other);
}
inline Tensor Tensor::flatten(int64_t start_dim, int64_t end_dim) const {
    return type().flatten(*this, start_dim, end_dim);
}
inline Tensor & Tensor::fill_(Scalar value) {
    return type().fill_(*this, value);
}
inline Tensor & Tensor::fill_(const Tensor & value) {
    return type().fill_(*this, value);
}
inline Tensor Tensor::floor() const {
    return type().floor(*this);
}
inline Tensor & Tensor::floor_() {
    return type().floor_(*this);
}
inline Tensor Tensor::ger(const Tensor & vec2) const {
    return type().ger(*this, vec2);
}
inline std::tuple<Tensor,Tensor> Tensor::gesv(const Tensor & A) const {
    return type().gesv(*this, A);
}
inline Tensor Tensor::fft(int64_t signal_ndim, bool normalized) const {
    return type().fft(*this, signal_ndim, normalized);
}
inline Tensor Tensor::ifft(int64_t signal_ndim, bool normalized) const {
    return type().ifft(*this, signal_ndim, normalized);
}
inline Tensor Tensor::rfft(int64_t signal_ndim, bool normalized, bool onesided) const {
    return type().rfft(*this, signal_ndim, normalized, onesided);
}
inline Tensor Tensor::irfft(int64_t signal_ndim, bool normalized, bool onesided, IntArrayRef signal_sizes) const {
    return type().irfft(*this, signal_ndim, normalized, onesided, signal_sizes);
}
inline Tensor Tensor::index(TensorList indices) const {
    return type().index(*this, indices);
}
inline Tensor & Tensor::index_copy_(int64_t dim, const Tensor & index, const Tensor & source) {
    return type().index_copy_(*this, dim, index, source);
}
inline Tensor Tensor::index_put(TensorList indices, const Tensor & values, bool accumulate) const {
    return type().index_put(*this, indices, values, accumulate);
}
inline Tensor & Tensor::index_put_(TensorList indices, const Tensor & values, bool accumulate) {
    return type().index_put_(*this, indices, values, accumulate);
}
inline Tensor Tensor::inverse() const {
    return type().inverse(*this);
}
inline Tensor Tensor::isclose(const Tensor & other, double rtol, double atol, bool equal_nan) const {
    return type().isclose(*this, other, rtol, atol, equal_nan);
}
inline bool Tensor::is_distributed() const {
    return type().is_distributed(*this);
}
inline bool Tensor::is_floating_point() const {
    return type().is_floating_point(*this);
}
inline bool Tensor::is_complex() const {
    return type().is_complex(*this);
}
inline bool Tensor::is_nonzero() const {
    return type().is_nonzero(*this);
}
inline bool Tensor::is_same_size(const Tensor & other) const {
    return type().is_same_size(*this, other);
}
inline bool Tensor::is_signed() const {
    return type().is_signed(*this);
}
inline std::tuple<Tensor,Tensor> Tensor::kthvalue(int64_t k, int64_t dim, bool keepdim) const {
    return type().kthvalue(*this, k, dim, keepdim);
}
inline Tensor Tensor::log() const {
    return type().log(*this);
}
inline Tensor & Tensor::log_() {
    return type().log_(*this);
}
inline Tensor Tensor::log10() const {
    return type().log10(*this);
}
inline Tensor & Tensor::log10_() {
    return type().log10_(*this);
}
inline Tensor Tensor::log1p() const {
    return type().log1p(*this);
}
inline Tensor & Tensor::log1p_() {
    return type().log1p_(*this);
}
inline Tensor Tensor::log2() const {
    return type().log2(*this);
}
inline Tensor & Tensor::log2_() {
    return type().log2_(*this);
}
inline Tensor Tensor::logdet() const {
    return type().logdet(*this);
}
inline Tensor Tensor::log_softmax(int64_t dim, ScalarType dtype) const {
    return type().log_softmax(*this, dim, dtype);
}
inline Tensor Tensor::log_softmax(int64_t dim) const {
    return type().log_softmax(*this, dim);
}
inline Tensor Tensor::logsumexp(IntArrayRef dim, bool keepdim) const {
    return type().logsumexp(*this, dim, keepdim);
}
inline Tensor Tensor::matmul(const Tensor & other) const {
    return type().matmul(*this, other);
}
inline Tensor Tensor::matrix_power(int64_t n) const {
    return type().matrix_power(*this, n);
}
inline std::tuple<Tensor,Tensor> Tensor::max(int64_t dim, bool keepdim) const {
    return type().max(*this, dim, keepdim);
}
inline Tensor Tensor::max_values(IntArrayRef dim, bool keepdim) const {
    return type().max_values(*this, dim, keepdim);
}
inline Tensor Tensor::mean(ScalarType dtype) const {
    return type().mean(*this, dtype);
}
inline Tensor Tensor::mean() const {
    return type().mean(*this);
}
inline Tensor Tensor::mean(IntArrayRef dim, bool keepdim, ScalarType dtype) const {
    return type().mean(*this, dim, keepdim, dtype);
}
inline Tensor Tensor::mean(IntArrayRef dim, bool keepdim) const {
    return type().mean(*this, dim, keepdim);
}
inline Tensor Tensor::mean(IntArrayRef dim, ScalarType dtype) const {
    return type().mean(*this, dim, dtype);
}
inline std::tuple<Tensor,Tensor> Tensor::median(int64_t dim, bool keepdim) const {
    return type().median(*this, dim, keepdim);
}
inline std::tuple<Tensor,Tensor> Tensor::min(int64_t dim, bool keepdim) const {
    return type().min(*this, dim, keepdim);
}
inline Tensor Tensor::min_values(IntArrayRef dim, bool keepdim) const {
    return type().min_values(*this, dim, keepdim);
}
inline Tensor Tensor::mm(const Tensor & mat2) const {
    return type().mm(*this, mat2);
}
inline std::tuple<Tensor,Tensor> Tensor::mode(int64_t dim, bool keepdim) const {
    return type().mode(*this, dim, keepdim);
}
inline Tensor Tensor::mul(const Tensor & other) const {
    return type().mul(*this, other);
}
inline Tensor & Tensor::mul_(const Tensor & other) {
    return type().mul_(*this, other);
}
inline Tensor Tensor::mul(Scalar other) const {
    return type().mul(*this, other);
}
inline Tensor & Tensor::mul_(Scalar other) {
    return type().mul_(*this, other);
}
inline Tensor Tensor::mv(const Tensor & vec) const {
    return type().mv(*this, vec);
}
inline Tensor Tensor::mvlgamma(int64_t p) const {
    return type().mvlgamma(*this, p);
}
inline Tensor & Tensor::mvlgamma_(int64_t p) {
    return type().mvlgamma_(*this, p);
}
inline Tensor Tensor::narrow_copy(int64_t dim, int64_t start, int64_t length) const {
    return type().narrow_copy(*this, dim, start, length);
}
inline Tensor Tensor::narrow(int64_t dim, int64_t start, int64_t length) const {
    return type().narrow(*this, dim, start, length);
}
inline Tensor Tensor::permute(IntArrayRef dims) const {
    return type().permute(*this, dims);
}
inline Tensor Tensor::pin_memory() const {
    return type().pin_memory(*this);
}
inline Tensor Tensor::pinverse(double rcond) const {
    return type().pinverse(*this, rcond);
}
inline Tensor Tensor::repeat(IntArrayRef repeats) const {
    return type().repeat(*this, repeats);
}
inline Tensor Tensor::reshape(IntArrayRef shape) const {
    return type().reshape(*this, shape);
}
inline Tensor Tensor::reshape_as(const Tensor & other) const {
    return type().reshape_as(*this, other);
}
inline Tensor Tensor::round() const {
    return type().round(*this);
}
inline Tensor & Tensor::round_() {
    return type().round_(*this);
}
inline Tensor Tensor::relu() const {
    return type().relu(*this);
}
inline Tensor & Tensor::relu_() {
    return type().relu_(*this);
}
inline Tensor Tensor::prelu(const Tensor & weight) const {
    return type().prelu(*this, weight);
}
inline std::tuple<Tensor,Tensor> Tensor::prelu_backward(const Tensor & grad_output, const Tensor & weight) const {
    return type().prelu_backward(grad_output, *this, weight);
}
inline Tensor Tensor::hardshrink(Scalar lambd) const {
    return type().hardshrink(*this, lambd);
}
inline Tensor Tensor::hardshrink_backward(const Tensor & grad_out, Scalar lambd) const {
    return type().hardshrink_backward(grad_out, *this, lambd);
}
inline Tensor Tensor::rsqrt() const {
    return type().rsqrt(*this);
}
inline Tensor & Tensor::rsqrt_() {
    return type().rsqrt_(*this);
}
inline Tensor Tensor::select(int64_t dim, int64_t index) const {
    return type().select(*this, dim, index);
}
inline Tensor Tensor::sigmoid() const {
    return type().sigmoid(*this);
}
inline Tensor & Tensor::sigmoid_() {
    return type().sigmoid_(*this);
}
inline Tensor Tensor::sin() const {
    return type().sin(*this);
}
inline Tensor & Tensor::sin_() {
    return type().sin_(*this);
}
inline Tensor Tensor::sinh() const {
    return type().sinh(*this);
}
inline Tensor & Tensor::sinh_() {
    return type().sinh_(*this);
}
inline Tensor Tensor::detach() const {
    return type().detach(*this);
}
inline Tensor & Tensor::detach_() {
    return type().detach_(*this);
}
inline int64_t Tensor::size(int64_t dim) const {
    return type().size(*this, dim);
}
inline Tensor Tensor::slice(int64_t dim, int64_t start, int64_t end, int64_t step) const {
    return type().slice(*this, dim, start, end, step);
}
inline std::tuple<Tensor,Tensor> Tensor::slogdet() const {
    return type().slogdet(*this);
}
inline Tensor Tensor::smm(const Tensor & mat2) const {
    return type().smm(*this, mat2);
}
inline Tensor Tensor::softmax(int64_t dim, ScalarType dtype) const {
    return type().softmax(*this, dim, dtype);
}
inline Tensor Tensor::softmax(int64_t dim) const {
    return type().softmax(*this, dim);
}
inline std::vector<Tensor> Tensor::split(int64_t split_size, int64_t dim) const {
    return type().split(*this, split_size, dim);
}
inline std::vector<Tensor> Tensor::split_with_sizes(IntArrayRef split_sizes, int64_t dim) const {
    return type().split_with_sizes(*this, split_sizes, dim);
}
inline Tensor Tensor::squeeze() const {
    return type().squeeze(*this);
}
inline Tensor Tensor::squeeze(int64_t dim) const {
    return type().squeeze(*this, dim);
}
inline Tensor & Tensor::squeeze_() {
    return type().squeeze_(*this);
}
inline Tensor & Tensor::squeeze_(int64_t dim) {
    return type().squeeze_(*this, dim);
}
inline Tensor Tensor::sspaddmm(const Tensor & mat1, const Tensor & mat2, Scalar beta, Scalar alpha) const {
    return type().sspaddmm(*this, mat1, mat2, beta, alpha);
}
inline Tensor Tensor::stft(int64_t n_fft, c10::optional<int64_t> hop_length, c10::optional<int64_t> win_length, const Tensor & window, bool normalized, bool onesided) const {
    return type().stft(*this, n_fft, hop_length, win_length, window, normalized, onesided);
}
inline int64_t Tensor::stride(int64_t dim) const {
    return type().stride(*this, dim);
}
inline Tensor Tensor::sum(ScalarType dtype) const {
    return type().sum(*this, dtype);
}
inline Tensor Tensor::sum() const {
    return type().sum(*this);
}
inline Tensor Tensor::sum(IntArrayRef dim, bool keepdim, ScalarType dtype) const {
    return type().sum(*this, dim, keepdim, dtype);
}
inline Tensor Tensor::sum(IntArrayRef dim, bool keepdim) const {
    return type().sum(*this, dim, keepdim);
}
inline Tensor Tensor::sum(IntArrayRef dim, ScalarType dtype) const {
    return type().sum(*this, dim, dtype);
}
inline Tensor Tensor::sum_to_size(IntArrayRef size) const {
    return type().sum_to_size(*this, size);
}
inline Tensor Tensor::sqrt() const {
    return type().sqrt(*this);
}
inline Tensor & Tensor::sqrt_() {
    return type().sqrt_(*this);
}
inline Tensor Tensor::std(bool unbiased) const {
    return type().std(*this, unbiased);
}
inline Tensor Tensor::std(IntArrayRef dim, bool unbiased, bool keepdim) const {
    return type().std(*this, dim, unbiased, keepdim);
}
inline Tensor Tensor::prod(ScalarType dtype) const {
    return type().prod(*this, dtype);
}
inline Tensor Tensor::prod() const {
    return type().prod(*this);
}
inline Tensor Tensor::prod(int64_t dim, bool keepdim, ScalarType dtype) const {
    return type().prod(*this, dim, keepdim, dtype);
}
inline Tensor Tensor::prod(int64_t dim, bool keepdim) const {
    return type().prod(*this, dim, keepdim);
}
inline Tensor Tensor::prod(int64_t dim, ScalarType dtype) const {
    return type().prod(*this, dim, dtype);
}
inline Tensor Tensor::t() const {
    return type().t(*this);
}
inline Tensor & Tensor::t_() {
    return type().t_(*this);
}
inline Tensor Tensor::tan() const {
    return type().tan(*this);
}
inline Tensor & Tensor::tan_() {
    return type().tan_(*this);
}
inline Tensor Tensor::tanh() const {
    return type().tanh(*this);
}
inline Tensor & Tensor::tanh_() {
    return type().tanh_(*this);
}
inline Tensor Tensor::transpose(int64_t dim0, int64_t dim1) const {
    return type().transpose(*this, dim0, dim1);
}
inline Tensor & Tensor::transpose_(int64_t dim0, int64_t dim1) {
    return type().transpose_(*this, dim0, dim1);
}
inline Tensor Tensor::flip(IntArrayRef dims) const {
    return type().flip(*this, dims);
}
inline Tensor Tensor::roll(IntArrayRef shifts, IntArrayRef dims) const {
    return type().roll(*this, shifts, dims);
}
inline Tensor Tensor::rot90(int64_t k, IntArrayRef dims) const {
    return type().rot90(*this, k, dims);
}
inline Tensor Tensor::trunc() const {
    return type().trunc(*this);
}
inline Tensor & Tensor::trunc_() {
    return type().trunc_(*this);
}
inline Tensor Tensor::type_as(const Tensor & other) const {
    return type().type_as(*this, other);
}
inline Tensor Tensor::unsqueeze(int64_t dim) const {
    return type().unsqueeze(*this, dim);
}
inline Tensor & Tensor::unsqueeze_(int64_t dim) {
    return type().unsqueeze_(*this, dim);
}
inline Tensor Tensor::var(bool unbiased) const {
    return type().var(*this, unbiased);
}
inline Tensor Tensor::var(IntArrayRef dim, bool unbiased, bool keepdim) const {
    return type().var(*this, dim, unbiased, keepdim);
}
inline Tensor Tensor::view_as(const Tensor & other) const {
    return type().view_as(*this, other);
}
inline Tensor Tensor::where(const Tensor & condition, const Tensor & other) const {
    return type().where(condition, *this, other);
}
inline Tensor Tensor::norm(c10::optional<Scalar> p, ScalarType dtype) const {
    return type().norm(*this, p, dtype);
}
inline Tensor Tensor::norm(Scalar p) const {
    return type().norm(*this, p);
}
inline Tensor Tensor::norm(c10::optional<Scalar> p, IntArrayRef dim, bool keepdim, ScalarType dtype) const {
    return type().norm(*this, p, dim, keepdim, dtype);
}
inline Tensor Tensor::norm(c10::optional<Scalar> p, IntArrayRef dim, bool keepdim) const {
    return type().norm(*this, p, dim, keepdim);
}
inline Tensor Tensor::clone() const {
    return type().clone(*this);
}
inline Tensor & Tensor::resize_as_(const Tensor & the_template) {
    return type().resize_as_(*this, the_template);
}
inline Tensor Tensor::pow(Scalar exponent) const {
    return type().pow(*this, exponent);
}
inline Tensor & Tensor::zero_() {
    return type().zero_(*this);
}
inline Tensor Tensor::sub(const Tensor & other, Scalar alpha) const {
    return type().sub(*this, other, alpha);
}
inline Tensor & Tensor::sub_(const Tensor & other, Scalar alpha) {
    return type().sub_(*this, other, alpha);
}
inline Tensor Tensor::sub(Scalar other, Scalar alpha) const {
    return type().sub(*this, other, alpha);
}
inline Tensor & Tensor::sub_(Scalar other, Scalar alpha) {
    return type().sub_(*this, other, alpha);
}
inline Tensor Tensor::addmm(const Tensor & mat1, const Tensor & mat2, Scalar beta, Scalar alpha) const {
    return type().addmm(*this, mat1, mat2, beta, alpha);
}
inline Tensor & Tensor::addmm_(const Tensor & mat1, const Tensor & mat2, Scalar beta, Scalar alpha) {
    return type().addmm_(*this, mat1, mat2, beta, alpha);
}
inline Tensor & Tensor::sparse_resize_(IntArrayRef size, int64_t sparse_dim, int64_t dense_dim) {
    return type().sparse_resize_(*this, size, sparse_dim, dense_dim);
}
inline Tensor & Tensor::sparse_resize_and_clear_(IntArrayRef size, int64_t sparse_dim, int64_t dense_dim) {
    return type().sparse_resize_and_clear_(*this, size, sparse_dim, dense_dim);
}
inline Tensor Tensor::sparse_mask(SparseTensorRef mask) const {
    return type().sparse_mask(*this, mask);
}
inline Tensor Tensor::to_dense() const {
    return type().to_dense(*this);
}
inline int64_t Tensor::sparse_dim() const {
    return type().sparse_dim(*this);
}
inline int64_t Tensor::_dimI() const {
    return type()._dimI(*this);
}
inline int64_t Tensor::dense_dim() const {
    return type().dense_dim(*this);
}
inline int64_t Tensor::_dimV() const {
    return type()._dimV(*this);
}
inline int64_t Tensor::_nnz() const {
    return type()._nnz(*this);
}
inline Tensor Tensor::coalesce() const {
    return type().coalesce(*this);
}
inline bool Tensor::is_coalesced() const {
    return type().is_coalesced(*this);
}
inline Tensor Tensor::_indices() const {
    return type()._indices(*this);
}
inline Tensor Tensor::_values() const {
    return type()._values(*this);
}
inline Tensor & Tensor::_coalesced_(bool coalesced) {
    return type()._coalesced_(*this, coalesced);
}
inline Tensor Tensor::indices() const {
    return type().indices(*this);
}
inline Tensor Tensor::values() const {
    return type().values(*this);
}
inline int64_t Tensor::numel() const {
    return type().numel(*this);
}
inline std::vector<Tensor> Tensor::unbind(int64_t dim) const {
    return type().unbind(*this, dim);
}
inline Tensor Tensor::to_sparse(int64_t sparse_dim) const {
    return type().to_sparse(*this, sparse_dim);
}
inline Tensor Tensor::to_sparse() const {
    return type().to_sparse(*this);
}
inline Tensor Tensor::to(const TensorOptions & options, bool non_blocking, bool copy) const {
    return type().to(*this, options, non_blocking, copy);
}
inline Tensor Tensor::to(Device device, ScalarType dtype, bool non_blocking, bool copy) const {
    return type().to(*this, device, dtype, non_blocking, copy);
}
inline Tensor Tensor::to(ScalarType dtype, bool non_blocking, bool copy) const {
    return type().to(*this, dtype, non_blocking, copy);
}
inline Tensor Tensor::to(const Tensor & other, bool non_blocking, bool copy) const {
    return type().to(*this, other, non_blocking, copy);
}
inline Scalar Tensor::item() const {
    return type().item(*this);
}
inline void* Tensor::data_ptr() const {
    return type().data_ptr(*this);
}
inline Tensor & Tensor::set_(Storage source) {
    return type().set_(*this, source);
}
inline Tensor & Tensor::set_(Storage source, int64_t storage_offset, IntArrayRef size, IntArrayRef stride) {
    return type().set_(*this, source, storage_offset, size, stride);
}
inline Tensor & Tensor::set_(const Tensor & source) {
    return type().set_(*this, source);
}
inline Tensor & Tensor::set_() {
    return type().set_(*this);
}
inline bool Tensor::is_set_to(const Tensor & tensor) const {
    return type().is_set_to(*this, tensor);
}
inline Tensor & Tensor::masked_fill_(const Tensor & mask, Scalar value) {
    return type().masked_fill_(*this, mask, value);
}
inline Tensor & Tensor::masked_fill_(const Tensor & mask, const Tensor & value) {
    return type().masked_fill_(*this, mask, value);
}
inline Tensor & Tensor::masked_scatter_(const Tensor & mask, const Tensor & source) {
    return type().masked_scatter_(*this, mask, source);
}
inline Tensor Tensor::view(IntArrayRef size) const {
    return type().view(*this, size);
}
inline Tensor & Tensor::put_(const Tensor & index, const Tensor & source, bool accumulate) {
    return type().put_(*this, index, source, accumulate);
}
inline Tensor & Tensor::index_add_(int64_t dim, const Tensor & index, const Tensor & source) {
    return type().index_add_(*this, dim, index, source);
}
inline Tensor & Tensor::index_fill_(int64_t dim, const Tensor & index, Scalar value) {
    return type().index_fill_(*this, dim, index, value);
}
inline Tensor & Tensor::index_fill_(int64_t dim, const Tensor & index, const Tensor & value) {
    return type().index_fill_(*this, dim, index, value);
}
inline Tensor & Tensor::scatter_(int64_t dim, const Tensor & index, const Tensor & src) {
    return type().scatter_(*this, dim, index, src);
}
inline Tensor & Tensor::scatter_(int64_t dim, const Tensor & index, Scalar value) {
    return type().scatter_(*this, dim, index, value);
}
inline Tensor & Tensor::scatter_add_(int64_t dim, const Tensor & index, const Tensor & src) {
    return type().scatter_add_(*this, dim, index, src);
}
inline Tensor & Tensor::lt_(Scalar other) {
    return type().lt_(*this, other);
}
inline Tensor & Tensor::lt_(const Tensor & other) {
    return type().lt_(*this, other);
}
inline Tensor & Tensor::gt_(Scalar other) {
    return type().gt_(*this, other);
}
inline Tensor & Tensor::gt_(const Tensor & other) {
    return type().gt_(*this, other);
}
inline Tensor & Tensor::le_(Scalar other) {
    return type().le_(*this, other);
}
inline Tensor & Tensor::le_(const Tensor & other) {
    return type().le_(*this, other);
}
inline Tensor & Tensor::ge_(Scalar other) {
    return type().ge_(*this, other);
}
inline Tensor & Tensor::ge_(const Tensor & other) {
    return type().ge_(*this, other);
}
inline Tensor & Tensor::eq_(Scalar other) {
    return type().eq_(*this, other);
}
inline Tensor & Tensor::eq_(const Tensor & other) {
    return type().eq_(*this, other);
}
inline Tensor & Tensor::ne_(Scalar other) {
    return type().ne_(*this, other);
}
inline Tensor & Tensor::ne_(const Tensor & other) {
    return type().ne_(*this, other);
}
inline Tensor Tensor::__and__(Scalar other) const {
    return type().__and__(*this, other);
}
inline Tensor Tensor::__and__(const Tensor & other) const {
    return type().__and__(*this, other);
}
inline Tensor & Tensor::__iand__(Scalar other) {
    return type().__iand__(*this, other);
}
inline Tensor & Tensor::__iand__(const Tensor & other) {
    return type().__iand__(*this, other);
}
inline Tensor Tensor::__or__(Scalar other) const {
    return type().__or__(*this, other);
}
inline Tensor Tensor::__or__(const Tensor & other) const {
    return type().__or__(*this, other);
}
inline Tensor & Tensor::__ior__(Scalar other) {
    return type().__ior__(*this, other);
}
inline Tensor & Tensor::__ior__(const Tensor & other) {
    return type().__ior__(*this, other);
}
inline Tensor Tensor::__xor__(Scalar other) const {
    return type().__xor__(*this, other);
}
inline Tensor Tensor::__xor__(const Tensor & other) const {
    return type().__xor__(*this, other);
}
inline Tensor & Tensor::__ixor__(Scalar other) {
    return type().__ixor__(*this, other);
}
inline Tensor & Tensor::__ixor__(const Tensor & other) {
    return type().__ixor__(*this, other);
}
inline Tensor Tensor::__lshift__(Scalar other) const {
    return type().__lshift__(*this, other);
}
inline Tensor Tensor::__lshift__(const Tensor & other) const {
    return type().__lshift__(*this, other);
}
inline Tensor & Tensor::__ilshift__(Scalar other) {
    return type().__ilshift__(*this, other);
}
inline Tensor & Tensor::__ilshift__(const Tensor & other) {
    return type().__ilshift__(*this, other);
}
inline Tensor Tensor::__rshift__(Scalar other) const {
    return type().__rshift__(*this, other);
}
inline Tensor Tensor::__rshift__(const Tensor & other) const {
    return type().__rshift__(*this, other);
}
inline Tensor & Tensor::__irshift__(Scalar other) {
    return type().__irshift__(*this, other);
}
inline Tensor & Tensor::__irshift__(const Tensor & other) {
    return type().__irshift__(*this, other);
}
inline Tensor & Tensor::lgamma_() {
    return type().lgamma_(*this);
}
inline Tensor & Tensor::atan2_(const Tensor & other) {
    return type().atan2_(*this, other);
}
inline Tensor & Tensor::tril_(int64_t diagonal) {
    return type().tril_(*this, diagonal);
}
inline Tensor & Tensor::triu_(int64_t diagonal) {
    return type().triu_(*this, diagonal);
}
inline Tensor & Tensor::digamma_() {
    return type().digamma_(*this);
}
inline Tensor & Tensor::polygamma_(int64_t n) {
    return type().polygamma_(*this, n);
}
inline Tensor & Tensor::erfinv_() {
    return type().erfinv_(*this);
}
inline Tensor & Tensor::frac_() {
    return type().frac_(*this);
}
inline Tensor & Tensor::renorm_(Scalar p, int64_t dim, Scalar maxnorm) {
    return type().renorm_(*this, p, dim, maxnorm);
}
inline Tensor & Tensor::reciprocal_() {
    return type().reciprocal_(*this);
}
inline Tensor & Tensor::neg_() {
    return type().neg_(*this);
}
inline Tensor & Tensor::pow_(Scalar exponent) {
    return type().pow_(*this, exponent);
}
inline Tensor & Tensor::pow_(const Tensor & exponent) {
    return type().pow_(*this, exponent);
}
inline Tensor & Tensor::lerp_(const Tensor & end, Scalar weight) {
    return type().lerp_(*this, end, weight);
}
inline Tensor & Tensor::sign_() {
    return type().sign_(*this);
}
inline Tensor & Tensor::fmod_(Scalar other) {
    return type().fmod_(*this, other);
}
inline Tensor & Tensor::fmod_(const Tensor & other) {
    return type().fmod_(*this, other);
}
inline Tensor & Tensor::remainder_(Scalar other) {
    return type().remainder_(*this, other);
}
inline Tensor & Tensor::remainder_(const Tensor & other) {
    return type().remainder_(*this, other);
}
inline Tensor & Tensor::addbmm_(const Tensor & batch1, const Tensor & batch2, Scalar beta, Scalar alpha) {
    return type().addbmm_(*this, batch1, batch2, beta, alpha);
}
inline Tensor Tensor::addbmm(const Tensor & batch1, const Tensor & batch2, Scalar beta, Scalar alpha) const {
    return type().addbmm(*this, batch1, batch2, beta, alpha);
}
inline Tensor & Tensor::addcmul_(const Tensor & tensor1, const Tensor & tensor2, Scalar value) {
    return type().addcmul_(*this, tensor1, tensor2, value);
}
inline Tensor & Tensor::addcdiv_(const Tensor & tensor1, const Tensor & tensor2, Scalar value) {
    return type().addcdiv_(*this, tensor1, tensor2, value);
}
inline Tensor & Tensor::random_(int64_t from, int64_t to, Generator * generator) {
    return type().random_(*this, from, to, generator);
}
inline Tensor & Tensor::random_(int64_t to, Generator * generator) {
    return type().random_(*this, to, generator);
}
inline Tensor & Tensor::random_(Generator * generator) {
    return type().random_(*this, generator);
}
inline Tensor & Tensor::uniform_(double from, double to, Generator * generator) {
    return type().uniform_(*this, from, to, generator);
}
inline Tensor & Tensor::normal_(double mean, double std, Generator * generator) {
    return type().normal_(*this, mean, std, generator);
}
inline Tensor & Tensor::cauchy_(double median, double sigma, Generator * generator) {
    return type().cauchy_(*this, median, sigma, generator);
}
inline Tensor & Tensor::log_normal_(double mean, double std, Generator * generator) {
    return type().log_normal_(*this, mean, std, generator);
}
inline Tensor & Tensor::exponential_(double lambd, Generator * generator) {
    return type().exponential_(*this, lambd, generator);
}
inline Tensor & Tensor::geometric_(double p, Generator * generator) {
    return type().geometric_(*this, p, generator);
}
inline Tensor Tensor::diag(int64_t diagonal) const {
    return type().diag(*this, diagonal);
}
inline Tensor Tensor::cross(const Tensor & other, int64_t dim) const {
    return type().cross(*this, other, dim);
}
inline Tensor Tensor::triu(int64_t diagonal) const {
    return type().triu(*this, diagonal);
}
inline Tensor Tensor::tril(int64_t diagonal) const {
    return type().tril(*this, diagonal);
}
inline Tensor Tensor::trace() const {
    return type().trace(*this);
}
inline Tensor Tensor::ne(Scalar other) const {
    return type().ne(*this, other);
}
inline Tensor Tensor::ne(const Tensor & other) const {
    return type().ne(*this, other);
}
inline Tensor Tensor::eq(Scalar other) const {
    return type().eq(*this, other);
}
inline Tensor Tensor::eq(const Tensor & other) const {
    return type().eq(*this, other);
}
inline Tensor Tensor::ge(Scalar other) const {
    return type().ge(*this, other);
}
inline Tensor Tensor::ge(const Tensor & other) const {
    return type().ge(*this, other);
}
inline Tensor Tensor::le(Scalar other) const {
    return type().le(*this, other);
}
inline Tensor Tensor::le(const Tensor & other) const {
    return type().le(*this, other);
}
inline Tensor Tensor::gt(Scalar other) const {
    return type().gt(*this, other);
}
inline Tensor Tensor::gt(const Tensor & other) const {
    return type().gt(*this, other);
}
inline Tensor Tensor::lt(Scalar other) const {
    return type().lt(*this, other);
}
inline Tensor Tensor::lt(const Tensor & other) const {
    return type().lt(*this, other);
}
inline Tensor Tensor::take(const Tensor & index) const {
    return type().take(*this, index);
}
inline Tensor Tensor::index_select(int64_t dim, const Tensor & index) const {
    return type().index_select(*this, dim, index);
}
inline Tensor Tensor::masked_select(const Tensor & mask) const {
    return type().masked_select(*this, mask);
}
inline Tensor Tensor::nonzero() const {
    return type().nonzero(*this);
}
inline Tensor Tensor::gather(int64_t dim, const Tensor & index) const {
    return type().gather(*this, dim, index);
}
inline Tensor Tensor::addcmul(const Tensor & tensor1, const Tensor & tensor2, Scalar value) const {
    return type().addcmul(*this, tensor1, tensor2, value);
}
inline Tensor Tensor::addcdiv(const Tensor & tensor1, const Tensor & tensor2, Scalar value) const {
    return type().addcdiv(*this, tensor1, tensor2, value);
}
inline std::tuple<Tensor,Tensor> Tensor::gels(const Tensor & A) const {
    return type().gels(*this, A);
}
inline std::tuple<Tensor,Tensor> Tensor::trtrs(const Tensor & A, bool upper, bool transpose, bool unitriangular) const {
    return type().trtrs(*this, A, upper, transpose, unitriangular);
}
inline std::tuple<Tensor,Tensor> Tensor::symeig(bool eigenvectors, bool upper) const {
    return type().symeig(*this, eigenvectors, upper);
}
inline std::tuple<Tensor,Tensor> Tensor::eig(bool eigenvectors) const {
    return type().eig(*this, eigenvectors);
}
inline std::tuple<Tensor,Tensor,Tensor> Tensor::svd(bool some, bool compute_uv) const {
    return type().svd(*this, some, compute_uv);
}
inline Tensor Tensor::cholesky(bool upper) const {
    return type().cholesky(*this, upper);
}
inline Tensor Tensor::cholesky_solve(const Tensor & input2, bool upper) const {
    return type().cholesky_solve(*this, input2, upper);
}
inline Tensor Tensor::potri(bool upper) const {
    return type().potri(*this, upper);
}
inline std::tuple<Tensor,Tensor> Tensor::pstrf(bool upper, Scalar tol) const {
    return type().pstrf(*this, upper, tol);
}
inline std::tuple<Tensor,Tensor> Tensor::qr() const {
    return type().qr(*this);
}
inline std::tuple<Tensor,Tensor> Tensor::geqrf() const {
    return type().geqrf(*this);
}
inline Tensor Tensor::orgqr(const Tensor & input2) const {
    return type().orgqr(*this, input2);
}
inline Tensor Tensor::ormqr(const Tensor & input2, const Tensor & input3, bool left, bool transpose) const {
    return type().ormqr(*this, input2, input3, left, transpose);
}
inline std::tuple<Tensor,Tensor> Tensor::btrifact(bool pivot) const {
    return type().btrifact(*this, pivot);
}
inline std::tuple<Tensor,Tensor,Tensor> Tensor::btrifact_with_info(bool pivot) const {
    return type().btrifact_with_info(*this, pivot);
}
inline Tensor Tensor::btrisolve(const Tensor & LU_data, const Tensor & LU_pivots) const {
    return type().btrisolve(*this, LU_data, LU_pivots);
}
inline Tensor Tensor::multinomial(int64_t num_samples, bool replacement, Generator * generator) const {
    return type().multinomial(*this, num_samples, replacement, generator);
}
inline Tensor Tensor::lgamma() const {
    return type().lgamma(*this);
}
inline Tensor Tensor::digamma() const {
    return type().digamma(*this);
}
inline Tensor Tensor::polygamma(int64_t n) const {
    return type().polygamma(n, *this);
}
inline Tensor Tensor::erfinv() const {
    return type().erfinv(*this);
}
inline Tensor Tensor::frac() const {
    return type().frac(*this);
}
inline Tensor Tensor::dist(const Tensor & other, Scalar p) const {
    return type().dist(*this, other, p);
}
inline Tensor Tensor::reciprocal() const {
    return type().reciprocal(*this);
}
inline Tensor Tensor::neg() const {
    return type().neg(*this);
}
inline Tensor Tensor::atan2(const Tensor & other) const {
    return type().atan2(*this, other);
}
inline Tensor Tensor::lerp(const Tensor & end, Scalar weight) const {
    return type().lerp(*this, end, weight);
}
inline Tensor Tensor::histc(int64_t bins, Scalar min, Scalar max) const {
    return type().histc(*this, bins, min, max);
}
inline Tensor Tensor::sign() const {
    return type().sign(*this);
}
inline Tensor Tensor::fmod(Scalar other) const {
    return type().fmod(*this, other);
}
inline Tensor Tensor::fmod(const Tensor & other) const {
    return type().fmod(*this, other);
}
inline Tensor Tensor::remainder(Scalar other) const {
    return type().remainder(*this, other);
}
inline Tensor Tensor::remainder(const Tensor & other) const {
    return type().remainder(*this, other);
}
inline Tensor Tensor::min(const Tensor & other) const {
    return type().min(*this, other);
}
inline Tensor Tensor::min() const {
    return type().min(*this);
}
inline Tensor Tensor::max(const Tensor & other) const {
    return type().max(*this, other);
}
inline Tensor Tensor::max() const {
    return type().max(*this);
}
inline Tensor Tensor::median() const {
    return type().median(*this);
}
inline std::tuple<Tensor,Tensor> Tensor::sort(int64_t dim, bool descending) const {
    return type().sort(*this, dim, descending);
}
inline std::tuple<Tensor,Tensor> Tensor::topk(int64_t k, int64_t dim, bool largest, bool sorted) const {
    return type().topk(*this, k, dim, largest, sorted);
}
inline Tensor Tensor::all() const {
    return type().all(*this);
}
inline Tensor Tensor::any() const {
    return type().any(*this);
}
inline Tensor Tensor::renorm(Scalar p, int64_t dim, Scalar maxnorm) const {
    return type().renorm(*this, p, dim, maxnorm);
}
inline Tensor Tensor::unfold(int64_t dimension, int64_t size, int64_t step) const {
    return type().unfold(*this, dimension, size, step);
}
inline bool Tensor::equal(const Tensor & other) const {
    return type().equal(*this, other);
}
inline Tensor Tensor::pow(const Tensor & exponent) const {
    return type().pow(*this, exponent);
}
inline Tensor Tensor::alias() const {
    return type().alias(*this);
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

#define DEFINE_CAST(T, name, _)                  \
  template <>                                    \
  inline T* Tensor::data() const {               \
    AT_CHECK(                                    \
        type().scalarType() == ScalarType::name, \
        "expected scalar type ",                 \
        #name,                                   \
        " but found ",                           \
        c10::toString(type().scalarType()));     \
    return static_cast<T*>(this->data_ptr());    \
  }

AT_FORALL_SCALAR_TYPES_WITH_COMPLEX_EXCEPT_COMPLEX_HALF(DEFINE_CAST)
#undef DEFINE_CAST

#define DEFINE_ITEM(T, name, _)   \
  template <>                     \
  inline T Tensor::item() const { \
    return item().to##name();     \
  }

AT_FORALL_SCALAR_TYPES_WITH_COMPLEX_EXCEPT_COMPLEX_HALF(DEFINE_ITEM)
#undef DEFINE_ITEM

} //namespace at
