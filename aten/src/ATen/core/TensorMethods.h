#pragma once

#include <ATen/core/Tensor.h>
#include <c10/core/Scalar.h>
#include <c10/macros/Macros.h>
#include <ATen/core/SparseTensorRef.h>
#include <ATen/core/Type.h>
#include <c10/core/TensorOptions.h>
#include <ATen/core/DeprecatedTypeProperties.h>

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
    return dispatch_type().abs(*this);
}
inline Tensor & Tensor::abs_() {
    return dispatch_type().abs_(*this);
}
inline Tensor Tensor::acos() const {
    return dispatch_type().acos(*this);
}
inline Tensor & Tensor::acos_() {
    return dispatch_type().acos_(*this);
}
inline Tensor Tensor::add(const Tensor & other, Scalar alpha) const {
    return dispatch_type().add(*this, other, alpha);
}
inline Tensor & Tensor::add_(const Tensor & other, Scalar alpha) {
    return dispatch_type().add_(*this, other, alpha);
}
inline Tensor Tensor::add(Scalar other, Scalar alpha) const {
    return dispatch_type().add(*this, other, alpha);
}
inline Tensor & Tensor::add_(Scalar other, Scalar alpha) {
    return dispatch_type().add_(*this, other, alpha);
}
inline Tensor Tensor::addmv(const Tensor & mat, const Tensor & vec, Scalar beta, Scalar alpha) const {
    return dispatch_type().addmv(*this, mat, vec, beta, alpha);
}
inline Tensor & Tensor::addmv_(const Tensor & mat, const Tensor & vec, Scalar beta, Scalar alpha) {
    return dispatch_type().addmv_(*this, mat, vec, beta, alpha);
}
inline Tensor Tensor::addr(const Tensor & vec1, const Tensor & vec2, Scalar beta, Scalar alpha) const {
    return dispatch_type().addr(*this, vec1, vec2, beta, alpha);
}
inline Tensor & Tensor::addr_(const Tensor & vec1, const Tensor & vec2, Scalar beta, Scalar alpha) {
    return dispatch_type().addr_(*this, vec1, vec2, beta, alpha);
}
inline Tensor Tensor::all(int64_t dim, bool keepdim) const {
    return dispatch_type().all(*this, dim, keepdim);
}
inline bool Tensor::allclose(const Tensor & other, double rtol, double atol, bool equal_nan) const {
    return dispatch_type().allclose(*this, other, rtol, atol, equal_nan);
}
inline Tensor Tensor::any(int64_t dim, bool keepdim) const {
    return dispatch_type().any(*this, dim, keepdim);
}
inline Tensor Tensor::argmax(c10::optional<int64_t> dim, bool keepdim) const {
    return dispatch_type().argmax(*this, dim, keepdim);
}
inline Tensor Tensor::argmin(c10::optional<int64_t> dim, bool keepdim) const {
    return dispatch_type().argmin(*this, dim, keepdim);
}
inline Tensor Tensor::as_strided(IntArrayRef size, IntArrayRef stride, c10::optional<int64_t> storage_offset) const {
    return dispatch_type().as_strided(*this, size, stride, storage_offset);
}
inline Tensor & Tensor::as_strided_(IntArrayRef size, IntArrayRef stride, c10::optional<int64_t> storage_offset) {
    return dispatch_type().as_strided_(*this, size, stride, storage_offset);
}
inline Tensor Tensor::asin() const {
    return dispatch_type().asin(*this);
}
inline Tensor & Tensor::asin_() {
    return dispatch_type().asin_(*this);
}
inline Tensor Tensor::atan() const {
    return dispatch_type().atan(*this);
}
inline Tensor & Tensor::atan_() {
    return dispatch_type().atan_(*this);
}
inline Tensor Tensor::baddbmm(const Tensor & batch1, const Tensor & batch2, Scalar beta, Scalar alpha) const {
    return dispatch_type().baddbmm(*this, batch1, batch2, beta, alpha);
}
inline Tensor & Tensor::baddbmm_(const Tensor & batch1, const Tensor & batch2, Scalar beta, Scalar alpha) {
    return dispatch_type().baddbmm_(*this, batch1, batch2, beta, alpha);
}
inline Tensor Tensor::bernoulli(Generator * generator) const {
    return dispatch_type().bernoulli(*this, generator);
}
inline Tensor & Tensor::bernoulli_(const Tensor & p, Generator * generator) {
    return dispatch_type().bernoulli_(*this, p, generator);
}
inline Tensor & Tensor::bernoulli_(double p, Generator * generator) {
    return dispatch_type().bernoulli_(*this, p, generator);
}
inline Tensor Tensor::bernoulli(double p, Generator * generator) const {
    return dispatch_type().bernoulli(*this, p, generator);
}
inline Tensor Tensor::bincount(const Tensor & weights, int64_t minlength) const {
    return dispatch_type().bincount(*this, weights, minlength);
}
inline Tensor Tensor::bmm(const Tensor & mat2) const {
    return dispatch_type().bmm(*this, mat2);
}
inline Tensor Tensor::ceil() const {
    return dispatch_type().ceil(*this);
}
inline Tensor & Tensor::ceil_() {
    return dispatch_type().ceil_(*this);
}
inline std::vector<Tensor> Tensor::chunk(int64_t chunks, int64_t dim) const {
    return dispatch_type().chunk(*this, chunks, dim);
}
inline Tensor Tensor::clamp(c10::optional<Scalar> min, c10::optional<Scalar> max) const {
    return dispatch_type().clamp(*this, min, max);
}
inline Tensor & Tensor::clamp_(c10::optional<Scalar> min, c10::optional<Scalar> max) {
    return dispatch_type().clamp_(*this, min, max);
}
inline Tensor Tensor::clamp_max(Scalar max) const {
    return dispatch_type().clamp_max(*this, max);
}
inline Tensor & Tensor::clamp_max_(Scalar max) {
    return dispatch_type().clamp_max_(*this, max);
}
inline Tensor Tensor::clamp_min(Scalar min) const {
    return dispatch_type().clamp_min(*this, min);
}
inline Tensor & Tensor::clamp_min_(Scalar min) {
    return dispatch_type().clamp_min_(*this, min);
}
inline Tensor Tensor::contiguous() const {
    return dispatch_type().contiguous(*this);
}
inline Tensor & Tensor::copy_(const Tensor & src, bool non_blocking) {
    return dispatch_type().copy_(*this, src, non_blocking);
}
inline Tensor Tensor::cos() const {
    return dispatch_type().cos(*this);
}
inline Tensor & Tensor::cos_() {
    return dispatch_type().cos_(*this);
}
inline Tensor Tensor::cosh() const {
    return dispatch_type().cosh(*this);
}
inline Tensor & Tensor::cosh_() {
    return dispatch_type().cosh_(*this);
}
inline Tensor Tensor::cumsum(int64_t dim, ScalarType dtype) const {
    return dispatch_type().cumsum(*this, dim, dtype);
}
inline Tensor Tensor::cumsum(int64_t dim) const {
    return dispatch_type().cumsum(*this, dim);
}
inline Tensor Tensor::cumprod(int64_t dim, ScalarType dtype) const {
    return dispatch_type().cumprod(*this, dim, dtype);
}
inline Tensor Tensor::cumprod(int64_t dim) const {
    return dispatch_type().cumprod(*this, dim);
}
inline Tensor Tensor::det() const {
    return dispatch_type().det(*this);
}
inline Tensor Tensor::diag_embed(int64_t offset, int64_t dim1, int64_t dim2) const {
    return dispatch_type().diag_embed(*this, offset, dim1, dim2);
}
inline Tensor Tensor::diagflat(int64_t offset) const {
    return dispatch_type().diagflat(*this, offset);
}
inline Tensor Tensor::diagonal(int64_t offset, int64_t dim1, int64_t dim2) const {
    return dispatch_type().diagonal(*this, offset, dim1, dim2);
}
inline Tensor Tensor::div(const Tensor & other) const {
    return dispatch_type().div(*this, other);
}
inline Tensor & Tensor::div_(const Tensor & other) {
    return dispatch_type().div_(*this, other);
}
inline Tensor Tensor::div(Scalar other) const {
    return dispatch_type().div(*this, other);
}
inline Tensor & Tensor::div_(Scalar other) {
    return dispatch_type().div_(*this, other);
}
inline Tensor Tensor::dot(const Tensor & tensor) const {
    return dispatch_type().dot(*this, tensor);
}
inline Tensor & Tensor::resize_(IntArrayRef size) {
    return dispatch_type().resize_(*this, size);
}
inline Tensor Tensor::erf() const {
    return dispatch_type().erf(*this);
}
inline Tensor & Tensor::erf_() {
    return dispatch_type().erf_(*this);
}
inline Tensor Tensor::erfc() const {
    return dispatch_type().erfc(*this);
}
inline Tensor & Tensor::erfc_() {
    return dispatch_type().erfc_(*this);
}
inline Tensor Tensor::exp() const {
    return dispatch_type().exp(*this);
}
inline Tensor & Tensor::exp_() {
    return dispatch_type().exp_(*this);
}
inline Tensor Tensor::expm1() const {
    return dispatch_type().expm1(*this);
}
inline Tensor & Tensor::expm1_() {
    return dispatch_type().expm1_(*this);
}
inline Tensor Tensor::expand(IntArrayRef size, bool implicit) const {
    return dispatch_type().expand(*this, size, implicit);
}
inline Tensor Tensor::expand_as(const Tensor & other) const {
    return dispatch_type().expand_as(*this, other);
}
inline Tensor Tensor::flatten(int64_t start_dim, int64_t end_dim) const {
    return dispatch_type().flatten(*this, start_dim, end_dim);
}
inline Tensor & Tensor::fill_(Scalar value) {
    return dispatch_type().fill_(*this, value);
}
inline Tensor & Tensor::fill_(const Tensor & value) {
    return dispatch_type().fill_(*this, value);
}
inline Tensor Tensor::floor() const {
    return dispatch_type().floor(*this);
}
inline Tensor & Tensor::floor_() {
    return dispatch_type().floor_(*this);
}
inline Tensor Tensor::frac() const {
    return dispatch_type().frac(*this);
}
inline Tensor & Tensor::frac_() {
    return dispatch_type().frac_(*this);
}
inline Tensor Tensor::ger(const Tensor & vec2) const {
    return dispatch_type().ger(*this, vec2);
}
inline Tensor Tensor::fft(int64_t signal_ndim, bool normalized) const {
    return dispatch_type().fft(*this, signal_ndim, normalized);
}
inline Tensor Tensor::ifft(int64_t signal_ndim, bool normalized) const {
    return dispatch_type().ifft(*this, signal_ndim, normalized);
}
inline Tensor Tensor::rfft(int64_t signal_ndim, bool normalized, bool onesided) const {
    return dispatch_type().rfft(*this, signal_ndim, normalized, onesided);
}
inline Tensor Tensor::irfft(int64_t signal_ndim, bool normalized, bool onesided, IntArrayRef signal_sizes) const {
    return dispatch_type().irfft(*this, signal_ndim, normalized, onesided, signal_sizes);
}
inline Tensor Tensor::index(TensorList indices) const {
    return dispatch_type().index(*this, indices);
}
inline Tensor & Tensor::index_copy_(int64_t dim, const Tensor & index, const Tensor & source) {
    return dispatch_type().index_copy_(*this, dim, index, source);
}
inline Tensor Tensor::index_copy(int64_t dim, const Tensor & index, const Tensor & source) const {
    return dispatch_type().index_copy(*this, dim, index, source);
}
inline Tensor & Tensor::index_put_(TensorList indices, const Tensor & values, bool accumulate) {
    return dispatch_type().index_put_(*this, indices, values, accumulate);
}
inline Tensor Tensor::index_put(TensorList indices, const Tensor & values, bool accumulate) const {
    return dispatch_type().index_put(*this, indices, values, accumulate);
}
inline Tensor Tensor::inverse() const {
    return dispatch_type().inverse(*this);
}
inline Tensor Tensor::isclose(const Tensor & other, double rtol, double atol, bool equal_nan) const {
    return dispatch_type().isclose(*this, other, rtol, atol, equal_nan);
}
inline bool Tensor::is_distributed() const {
    return dispatch_type().is_distributed(*this);
}
inline bool Tensor::is_floating_point() const {
    return dispatch_type().is_floating_point(*this);
}
inline bool Tensor::is_complex() const {
    return dispatch_type().is_complex(*this);
}
inline bool Tensor::is_nonzero() const {
    return dispatch_type().is_nonzero(*this);
}
inline bool Tensor::is_same_size(const Tensor & other) const {
    return dispatch_type().is_same_size(*this, other);
}
inline bool Tensor::is_signed() const {
    return dispatch_type().is_signed(*this);
}
inline std::tuple<Tensor,Tensor> Tensor::kthvalue(int64_t k, int64_t dim, bool keepdim) const {
    return dispatch_type().kthvalue(*this, k, dim, keepdim);
}
inline Tensor Tensor::log() const {
    return dispatch_type().log(*this);
}
inline Tensor & Tensor::log_() {
    return dispatch_type().log_(*this);
}
inline Tensor Tensor::log10() const {
    return dispatch_type().log10(*this);
}
inline Tensor & Tensor::log10_() {
    return dispatch_type().log10_(*this);
}
inline Tensor Tensor::log1p() const {
    return dispatch_type().log1p(*this);
}
inline Tensor & Tensor::log1p_() {
    return dispatch_type().log1p_(*this);
}
inline Tensor Tensor::log2() const {
    return dispatch_type().log2(*this);
}
inline Tensor & Tensor::log2_() {
    return dispatch_type().log2_(*this);
}
inline Tensor Tensor::logdet() const {
    return dispatch_type().logdet(*this);
}
inline Tensor Tensor::log_softmax(int64_t dim, ScalarType dtype) const {
    return dispatch_type().log_softmax(*this, dim, dtype);
}
inline Tensor Tensor::log_softmax(int64_t dim) const {
    return dispatch_type().log_softmax(*this, dim);
}
inline Tensor Tensor::logsumexp(IntArrayRef dim, bool keepdim) const {
    return dispatch_type().logsumexp(*this, dim, keepdim);
}
inline Tensor Tensor::matmul(const Tensor & other) const {
    return dispatch_type().matmul(*this, other);
}
inline Tensor Tensor::matrix_power(int64_t n) const {
    return dispatch_type().matrix_power(*this, n);
}
inline std::tuple<Tensor,Tensor> Tensor::max(int64_t dim, bool keepdim) const {
    return dispatch_type().max(*this, dim, keepdim);
}
inline Tensor Tensor::max_values(IntArrayRef dim, bool keepdim) const {
    return dispatch_type().max_values(*this, dim, keepdim);
}
inline Tensor Tensor::mean(ScalarType dtype) const {
    return dispatch_type().mean(*this, dtype);
}
inline Tensor Tensor::mean() const {
    return dispatch_type().mean(*this);
}
inline Tensor Tensor::mean(IntArrayRef dim, bool keepdim, ScalarType dtype) const {
    return dispatch_type().mean(*this, dim, keepdim, dtype);
}
inline Tensor Tensor::mean(IntArrayRef dim, bool keepdim) const {
    return dispatch_type().mean(*this, dim, keepdim);
}
inline Tensor Tensor::mean(IntArrayRef dim, ScalarType dtype) const {
    return dispatch_type().mean(*this, dim, dtype);
}
inline std::tuple<Tensor,Tensor> Tensor::median(int64_t dim, bool keepdim) const {
    return dispatch_type().median(*this, dim, keepdim);
}
inline std::tuple<Tensor,Tensor> Tensor::min(int64_t dim, bool keepdim) const {
    return dispatch_type().min(*this, dim, keepdim);
}
inline Tensor Tensor::min_values(IntArrayRef dim, bool keepdim) const {
    return dispatch_type().min_values(*this, dim, keepdim);
}
inline Tensor Tensor::mm(const Tensor & mat2) const {
    return dispatch_type().mm(*this, mat2);
}
inline std::tuple<Tensor,Tensor> Tensor::mode(int64_t dim, bool keepdim) const {
    return dispatch_type().mode(*this, dim, keepdim);
}
inline Tensor Tensor::mul(const Tensor & other) const {
    return dispatch_type().mul(*this, other);
}
inline Tensor & Tensor::mul_(const Tensor & other) {
    return dispatch_type().mul_(*this, other);
}
inline Tensor Tensor::mul(Scalar other) const {
    return dispatch_type().mul(*this, other);
}
inline Tensor & Tensor::mul_(Scalar other) {
    return dispatch_type().mul_(*this, other);
}
inline Tensor Tensor::mv(const Tensor & vec) const {
    return dispatch_type().mv(*this, vec);
}
inline Tensor Tensor::mvlgamma(int64_t p) const {
    return dispatch_type().mvlgamma(*this, p);
}
inline Tensor & Tensor::mvlgamma_(int64_t p) {
    return dispatch_type().mvlgamma_(*this, p);
}
inline Tensor Tensor::narrow_copy(int64_t dim, int64_t start, int64_t length) const {
    return dispatch_type().narrow_copy(*this, dim, start, length);
}
inline Tensor Tensor::narrow(int64_t dim, int64_t start, int64_t length) const {
    return dispatch_type().narrow(*this, dim, start, length);
}
inline Tensor Tensor::permute(IntArrayRef dims) const {
    return dispatch_type().permute(*this, dims);
}
inline Tensor Tensor::pin_memory() const {
    return dispatch_type().pin_memory(*this);
}
inline Tensor Tensor::pinverse(double rcond) const {
    return dispatch_type().pinverse(*this, rcond);
}
inline Tensor Tensor::reciprocal() const {
    return dispatch_type().reciprocal(*this);
}
inline Tensor & Tensor::reciprocal_() {
    return dispatch_type().reciprocal_(*this);
}
inline Tensor Tensor::neg() const {
    return dispatch_type().neg(*this);
}
inline Tensor & Tensor::neg_() {
    return dispatch_type().neg_(*this);
}
inline Tensor Tensor::repeat(IntArrayRef repeats) const {
    return dispatch_type().repeat(*this, repeats);
}
inline Tensor Tensor::repeat_interleave(const Tensor & repeats, c10::optional<int64_t> dim) const {
    return dispatch_type().repeat_interleave(*this, repeats, dim);
}
inline Tensor Tensor::repeat_interleave(int64_t repeats, c10::optional<int64_t> dim) const {
    return dispatch_type().repeat_interleave(*this, repeats, dim);
}
inline Tensor Tensor::reshape(IntArrayRef shape) const {
    return dispatch_type().reshape(*this, shape);
}
inline Tensor Tensor::reshape_as(const Tensor & other) const {
    return dispatch_type().reshape_as(*this, other);
}
inline Tensor Tensor::round() const {
    return dispatch_type().round(*this);
}
inline Tensor & Tensor::round_() {
    return dispatch_type().round_(*this);
}
inline Tensor Tensor::relu() const {
    return dispatch_type().relu(*this);
}
inline Tensor & Tensor::relu_() {
    return dispatch_type().relu_(*this);
}
inline Tensor Tensor::prelu(const Tensor & weight) const {
    return dispatch_type().prelu(*this, weight);
}
inline std::tuple<Tensor,Tensor> Tensor::prelu_backward(const Tensor & grad_output, const Tensor & weight) const {
    return dispatch_type().prelu_backward(grad_output, *this, weight);
}
inline Tensor Tensor::hardshrink(Scalar lambd) const {
    return dispatch_type().hardshrink(*this, lambd);
}
inline Tensor Tensor::hardshrink_backward(const Tensor & grad_out, Scalar lambd) const {
    return dispatch_type().hardshrink_backward(grad_out, *this, lambd);
}
inline Tensor Tensor::rsqrt() const {
    return dispatch_type().rsqrt(*this);
}
inline Tensor & Tensor::rsqrt_() {
    return dispatch_type().rsqrt_(*this);
}
inline Tensor Tensor::select(int64_t dim, int64_t index) const {
    return dispatch_type().select(*this, dim, index);
}
inline Tensor Tensor::sigmoid() const {
    return dispatch_type().sigmoid(*this);
}
inline Tensor & Tensor::sigmoid_() {
    return dispatch_type().sigmoid_(*this);
}
inline Tensor Tensor::sin() const {
    return dispatch_type().sin(*this);
}
inline Tensor & Tensor::sin_() {
    return dispatch_type().sin_(*this);
}
inline Tensor Tensor::sinh() const {
    return dispatch_type().sinh(*this);
}
inline Tensor & Tensor::sinh_() {
    return dispatch_type().sinh_(*this);
}
inline Tensor Tensor::detach() const {
    return dispatch_type().detach(*this);
}
inline Tensor & Tensor::detach_() {
    return dispatch_type().detach_(*this);
}
inline int64_t Tensor::size(int64_t dim) const {
    return dispatch_type().size(*this, dim);
}
inline Tensor Tensor::slice(int64_t dim, int64_t start, int64_t end, int64_t step) const {
    return dispatch_type().slice(*this, dim, start, end, step);
}
inline std::tuple<Tensor,Tensor> Tensor::slogdet() const {
    return dispatch_type().slogdet(*this);
}
inline Tensor Tensor::smm(const Tensor & mat2) const {
    return dispatch_type().smm(*this, mat2);
}
inline Tensor Tensor::softmax(int64_t dim, ScalarType dtype) const {
    return dispatch_type().softmax(*this, dim, dtype);
}
inline Tensor Tensor::softmax(int64_t dim) const {
    return dispatch_type().softmax(*this, dim);
}
inline std::vector<Tensor> Tensor::split(int64_t split_size, int64_t dim) const {
    return dispatch_type().split(*this, split_size, dim);
}
inline std::vector<Tensor> Tensor::split_with_sizes(IntArrayRef split_sizes, int64_t dim) const {
    return dispatch_type().split_with_sizes(*this, split_sizes, dim);
}
inline Tensor Tensor::squeeze() const {
    return dispatch_type().squeeze(*this);
}
inline Tensor Tensor::squeeze(int64_t dim) const {
    return dispatch_type().squeeze(*this, dim);
}
inline Tensor & Tensor::squeeze_() {
    return dispatch_type().squeeze_(*this);
}
inline Tensor & Tensor::squeeze_(int64_t dim) {
    return dispatch_type().squeeze_(*this, dim);
}
inline Tensor Tensor::sspaddmm(const Tensor & mat1, const Tensor & mat2, Scalar beta, Scalar alpha) const {
    return dispatch_type().sspaddmm(*this, mat1, mat2, beta, alpha);
}
inline Tensor Tensor::stft(int64_t n_fft, c10::optional<int64_t> hop_length, c10::optional<int64_t> win_length, const Tensor & window, bool normalized, bool onesided) const {
    return dispatch_type().stft(*this, n_fft, hop_length, win_length, window, normalized, onesided);
}
inline int64_t Tensor::stride(int64_t dim) const {
    return dispatch_type().stride(*this, dim);
}
inline Tensor Tensor::sum(ScalarType dtype) const {
    return dispatch_type().sum(*this, dtype);
}
inline Tensor Tensor::sum() const {
    return dispatch_type().sum(*this);
}
inline Tensor Tensor::sum(IntArrayRef dim, bool keepdim, ScalarType dtype) const {
    return dispatch_type().sum(*this, dim, keepdim, dtype);
}
inline Tensor Tensor::sum(IntArrayRef dim, bool keepdim) const {
    return dispatch_type().sum(*this, dim, keepdim);
}
inline Tensor Tensor::sum(IntArrayRef dim, ScalarType dtype) const {
    return dispatch_type().sum(*this, dim, dtype);
}
inline Tensor Tensor::sum_to_size(IntArrayRef size) const {
    return dispatch_type().sum_to_size(*this, size);
}
inline Tensor Tensor::sqrt() const {
    return dispatch_type().sqrt(*this);
}
inline Tensor & Tensor::sqrt_() {
    return dispatch_type().sqrt_(*this);
}
inline Tensor Tensor::std(bool unbiased) const {
    return dispatch_type().std(*this, unbiased);
}
inline Tensor Tensor::std(IntArrayRef dim, bool unbiased, bool keepdim) const {
    return dispatch_type().std(*this, dim, unbiased, keepdim);
}
inline Tensor Tensor::prod(ScalarType dtype) const {
    return dispatch_type().prod(*this, dtype);
}
inline Tensor Tensor::prod() const {
    return dispatch_type().prod(*this);
}
inline Tensor Tensor::prod(int64_t dim, bool keepdim, ScalarType dtype) const {
    return dispatch_type().prod(*this, dim, keepdim, dtype);
}
inline Tensor Tensor::prod(int64_t dim, bool keepdim) const {
    return dispatch_type().prod(*this, dim, keepdim);
}
inline Tensor Tensor::prod(int64_t dim, ScalarType dtype) const {
    return dispatch_type().prod(*this, dim, dtype);
}
inline Tensor Tensor::t() const {
    return dispatch_type().t(*this);
}
inline Tensor & Tensor::t_() {
    return dispatch_type().t_(*this);
}
inline Tensor Tensor::tan() const {
    return dispatch_type().tan(*this);
}
inline Tensor & Tensor::tan_() {
    return dispatch_type().tan_(*this);
}
inline Tensor Tensor::tanh() const {
    return dispatch_type().tanh(*this);
}
inline Tensor & Tensor::tanh_() {
    return dispatch_type().tanh_(*this);
}
inline Tensor Tensor::transpose(int64_t dim0, int64_t dim1) const {
    return dispatch_type().transpose(*this, dim0, dim1);
}
inline Tensor & Tensor::transpose_(int64_t dim0, int64_t dim1) {
    return dispatch_type().transpose_(*this, dim0, dim1);
}
inline Tensor Tensor::flip(IntArrayRef dims) const {
    return dispatch_type().flip(*this, dims);
}
inline Tensor Tensor::roll(IntArrayRef shifts, IntArrayRef dims) const {
    return dispatch_type().roll(*this, shifts, dims);
}
inline Tensor Tensor::rot90(int64_t k, IntArrayRef dims) const {
    return dispatch_type().rot90(*this, k, dims);
}
inline Tensor Tensor::trunc() const {
    return dispatch_type().trunc(*this);
}
inline Tensor & Tensor::trunc_() {
    return dispatch_type().trunc_(*this);
}
inline Tensor Tensor::type_as(const Tensor & other) const {
    return dispatch_type().type_as(*this, other);
}
inline Tensor Tensor::unsqueeze(int64_t dim) const {
    return dispatch_type().unsqueeze(*this, dim);
}
inline Tensor & Tensor::unsqueeze_(int64_t dim) {
    return dispatch_type().unsqueeze_(*this, dim);
}
inline Tensor Tensor::var(bool unbiased) const {
    return dispatch_type().var(*this, unbiased);
}
inline Tensor Tensor::var(IntArrayRef dim, bool unbiased, bool keepdim) const {
    return dispatch_type().var(*this, dim, unbiased, keepdim);
}
inline Tensor Tensor::view_as(const Tensor & other) const {
    return dispatch_type().view_as(*this, other);
}
inline Tensor Tensor::where(const Tensor & condition, const Tensor & other) const {
    return dispatch_type().where(condition, *this, other);
}
inline Tensor Tensor::norm(c10::optional<Scalar> p, ScalarType dtype) const {
    return dispatch_type().norm(*this, p, dtype);
}
inline Tensor Tensor::norm(Scalar p) const {
    return dispatch_type().norm(*this, p);
}
inline Tensor Tensor::norm(c10::optional<Scalar> p, IntArrayRef dim, bool keepdim, ScalarType dtype) const {
    return dispatch_type().norm(*this, p, dim, keepdim, dtype);
}
inline Tensor Tensor::norm(c10::optional<Scalar> p, IntArrayRef dim, bool keepdim) const {
    return dispatch_type().norm(*this, p, dim, keepdim);
}
inline Tensor Tensor::clone() const {
    return dispatch_type().clone(*this);
}
inline Tensor & Tensor::resize_as_(const Tensor & the_template) {
    return dispatch_type().resize_as_(*this, the_template);
}
inline Tensor Tensor::pow(Scalar exponent) const {
    return dispatch_type().pow(*this, exponent);
}
inline Tensor & Tensor::zero_() {
    return dispatch_type().zero_(*this);
}
inline Tensor Tensor::sub(const Tensor & other, Scalar alpha) const {
    return dispatch_type().sub(*this, other, alpha);
}
inline Tensor & Tensor::sub_(const Tensor & other, Scalar alpha) {
    return dispatch_type().sub_(*this, other, alpha);
}
inline Tensor Tensor::sub(Scalar other, Scalar alpha) const {
    return dispatch_type().sub(*this, other, alpha);
}
inline Tensor & Tensor::sub_(Scalar other, Scalar alpha) {
    return dispatch_type().sub_(*this, other, alpha);
}
inline Tensor Tensor::addmm(const Tensor & mat1, const Tensor & mat2, Scalar beta, Scalar alpha) const {
    return dispatch_type().addmm(*this, mat1, mat2, beta, alpha);
}
inline Tensor & Tensor::addmm_(const Tensor & mat1, const Tensor & mat2, Scalar beta, Scalar alpha) {
    return dispatch_type().addmm_(*this, mat1, mat2, beta, alpha);
}
inline Tensor & Tensor::sparse_resize_(IntArrayRef size, int64_t sparse_dim, int64_t dense_dim) {
    return dispatch_type().sparse_resize_(*this, size, sparse_dim, dense_dim);
}
inline Tensor & Tensor::sparse_resize_and_clear_(IntArrayRef size, int64_t sparse_dim, int64_t dense_dim) {
    return dispatch_type().sparse_resize_and_clear_(*this, size, sparse_dim, dense_dim);
}
inline Tensor Tensor::sparse_mask(SparseTensorRef mask) const {
    return dispatch_type().sparse_mask(*this, mask);
}
inline Tensor Tensor::to_dense() const {
    return dispatch_type().to_dense(*this);
}
inline int64_t Tensor::sparse_dim() const {
    return dispatch_type().sparse_dim(*this);
}
inline int64_t Tensor::_dimI() const {
    return dispatch_type()._dimI(*this);
}
inline int64_t Tensor::dense_dim() const {
    return dispatch_type().dense_dim(*this);
}
inline int64_t Tensor::_dimV() const {
    return dispatch_type()._dimV(*this);
}
inline int64_t Tensor::_nnz() const {
    return dispatch_type()._nnz(*this);
}
inline Tensor Tensor::coalesce() const {
    return dispatch_type().coalesce(*this);
}
inline bool Tensor::is_coalesced() const {
    return dispatch_type().is_coalesced(*this);
}
inline Tensor Tensor::_indices() const {
    return dispatch_type()._indices(*this);
}
inline Tensor Tensor::_values() const {
    return dispatch_type()._values(*this);
}
inline Tensor & Tensor::_coalesced_(bool coalesced) {
    return dispatch_type()._coalesced_(*this, coalesced);
}
inline Tensor Tensor::indices() const {
    return dispatch_type().indices(*this);
}
inline Tensor Tensor::values() const {
    return dispatch_type().values(*this);
}
inline int64_t Tensor::numel() const {
    return dispatch_type().numel(*this);
}
inline std::vector<Tensor> Tensor::unbind(int64_t dim) const {
    return dispatch_type().unbind(*this, dim);
}
inline Tensor Tensor::to_sparse(int64_t sparse_dim) const {
    return dispatch_type().to_sparse(*this, sparse_dim);
}
inline Tensor Tensor::to_sparse() const {
    return dispatch_type().to_sparse(*this);
}
inline Tensor Tensor::to_mkldnn() const {
    return dispatch_type().to_mkldnn(*this);
}
inline Tensor Tensor::quantize_linear(double scale, int64_t zero_point) const {
    return dispatch_type().quantize_linear(*this, scale, zero_point);
}
inline Tensor Tensor::dequantize() const {
    return dispatch_type().dequantize(*this);
}
inline Scalar Tensor::q_scale() const {
    return dispatch_type().q_scale(*this);
}
inline Scalar Tensor::q_zero_point() const {
    return dispatch_type().q_zero_point(*this);
}
inline Tensor Tensor::int_repr() const {
    return dispatch_type().int_repr(*this);
}
inline Tensor Tensor::to(const TensorOptions & options, bool non_blocking, bool copy) const {
    return dispatch_type().to(*this, options, non_blocking, copy);
}
inline Tensor Tensor::to(Device device, ScalarType dtype, bool non_blocking, bool copy) const {
    return dispatch_type().to(*this, device, dtype, non_blocking, copy);
}
inline Tensor Tensor::to(ScalarType dtype, bool non_blocking, bool copy) const {
    return dispatch_type().to(*this, dtype, non_blocking, copy);
}
inline Tensor Tensor::to(const Tensor & other, bool non_blocking, bool copy) const {
    return dispatch_type().to(*this, other, non_blocking, copy);
}
inline Scalar Tensor::item() const {
    return dispatch_type().item(*this);
}
inline void* Tensor::data_ptr() const {
    return dispatch_type().data_ptr(*this);
}
inline Tensor & Tensor::set_(Storage source) {
    return dispatch_type().set_(*this, source);
}
inline Tensor & Tensor::set_(Storage source, int64_t storage_offset, IntArrayRef size, IntArrayRef stride) {
    return dispatch_type().set_(*this, source, storage_offset, size, stride);
}
inline Tensor & Tensor::set_(const Tensor & source) {
    return dispatch_type().set_(*this, source);
}
inline Tensor & Tensor::set_() {
    return dispatch_type().set_(*this);
}
inline bool Tensor::is_set_to(const Tensor & tensor) const {
    return dispatch_type().is_set_to(*this, tensor);
}
inline Tensor & Tensor::masked_fill_(const Tensor & mask, Scalar value) {
    return dispatch_type().masked_fill_(*this, mask, value);
}
inline Tensor Tensor::masked_fill(const Tensor & mask, Scalar value) const {
    return dispatch_type().masked_fill(*this, mask, value);
}
inline Tensor & Tensor::masked_fill_(const Tensor & mask, const Tensor & value) {
    return dispatch_type().masked_fill_(*this, mask, value);
}
inline Tensor Tensor::masked_fill(const Tensor & mask, const Tensor & value) const {
    return dispatch_type().masked_fill(*this, mask, value);
}
inline Tensor & Tensor::masked_scatter_(const Tensor & mask, const Tensor & source) {
    return dispatch_type().masked_scatter_(*this, mask, source);
}
inline Tensor Tensor::masked_scatter(const Tensor & mask, const Tensor & source) const {
    return dispatch_type().masked_scatter(*this, mask, source);
}
inline Tensor Tensor::view(IntArrayRef size) const {
    return dispatch_type().view(*this, size);
}
inline Tensor & Tensor::put_(const Tensor & index, const Tensor & source, bool accumulate) {
    return dispatch_type().put_(*this, index, source, accumulate);
}
inline Tensor & Tensor::index_add_(int64_t dim, const Tensor & index, const Tensor & source) {
    return dispatch_type().index_add_(*this, dim, index, source);
}
inline Tensor Tensor::index_add(int64_t dim, const Tensor & index, const Tensor & source) const {
    return dispatch_type().index_add(*this, dim, index, source);
}
inline Tensor & Tensor::index_fill_(int64_t dim, const Tensor & index, Scalar value) {
    return dispatch_type().index_fill_(*this, dim, index, value);
}
inline Tensor Tensor::index_fill(int64_t dim, const Tensor & index, Scalar value) const {
    return dispatch_type().index_fill(*this, dim, index, value);
}
inline Tensor & Tensor::index_fill_(int64_t dim, const Tensor & index, const Tensor & value) {
    return dispatch_type().index_fill_(*this, dim, index, value);
}
inline Tensor Tensor::index_fill(int64_t dim, const Tensor & index, const Tensor & value) const {
    return dispatch_type().index_fill(*this, dim, index, value);
}
inline Tensor & Tensor::scatter_(int64_t dim, const Tensor & index, const Tensor & src) {
    return dispatch_type().scatter_(*this, dim, index, src);
}
inline Tensor Tensor::scatter(int64_t dim, const Tensor & index, const Tensor & src) const {
    return dispatch_type().scatter(*this, dim, index, src);
}
inline Tensor & Tensor::scatter_(int64_t dim, const Tensor & index, Scalar value) {
    return dispatch_type().scatter_(*this, dim, index, value);
}
inline Tensor Tensor::scatter(int64_t dim, const Tensor & index, Scalar value) const {
    return dispatch_type().scatter(*this, dim, index, value);
}
inline Tensor & Tensor::scatter_add_(int64_t dim, const Tensor & index, const Tensor & src) {
    return dispatch_type().scatter_add_(*this, dim, index, src);
}
inline Tensor Tensor::scatter_add(int64_t dim, const Tensor & index, const Tensor & src) const {
    return dispatch_type().scatter_add(*this, dim, index, src);
}
inline Tensor & Tensor::lt_(Scalar other) {
    return dispatch_type().lt_(*this, other);
}
inline Tensor & Tensor::lt_(const Tensor & other) {
    return dispatch_type().lt_(*this, other);
}
inline Tensor & Tensor::gt_(Scalar other) {
    return dispatch_type().gt_(*this, other);
}
inline Tensor & Tensor::gt_(const Tensor & other) {
    return dispatch_type().gt_(*this, other);
}
inline Tensor & Tensor::le_(Scalar other) {
    return dispatch_type().le_(*this, other);
}
inline Tensor & Tensor::le_(const Tensor & other) {
    return dispatch_type().le_(*this, other);
}
inline Tensor & Tensor::ge_(Scalar other) {
    return dispatch_type().ge_(*this, other);
}
inline Tensor & Tensor::ge_(const Tensor & other) {
    return dispatch_type().ge_(*this, other);
}
inline Tensor & Tensor::eq_(Scalar other) {
    return dispatch_type().eq_(*this, other);
}
inline Tensor & Tensor::eq_(const Tensor & other) {
    return dispatch_type().eq_(*this, other);
}
inline Tensor & Tensor::ne_(Scalar other) {
    return dispatch_type().ne_(*this, other);
}
inline Tensor & Tensor::ne_(const Tensor & other) {
    return dispatch_type().ne_(*this, other);
}
inline Tensor Tensor::__and__(Scalar other) const {
    return dispatch_type().__and__(*this, other);
}
inline Tensor Tensor::__and__(const Tensor & other) const {
    return dispatch_type().__and__(*this, other);
}
inline Tensor & Tensor::__iand__(Scalar other) {
    return dispatch_type().__iand__(*this, other);
}
inline Tensor & Tensor::__iand__(const Tensor & other) {
    return dispatch_type().__iand__(*this, other);
}
inline Tensor Tensor::__or__(Scalar other) const {
    return dispatch_type().__or__(*this, other);
}
inline Tensor Tensor::__or__(const Tensor & other) const {
    return dispatch_type().__or__(*this, other);
}
inline Tensor & Tensor::__ior__(Scalar other) {
    return dispatch_type().__ior__(*this, other);
}
inline Tensor & Tensor::__ior__(const Tensor & other) {
    return dispatch_type().__ior__(*this, other);
}
inline Tensor Tensor::__xor__(Scalar other) const {
    return dispatch_type().__xor__(*this, other);
}
inline Tensor Tensor::__xor__(const Tensor & other) const {
    return dispatch_type().__xor__(*this, other);
}
inline Tensor & Tensor::__ixor__(Scalar other) {
    return dispatch_type().__ixor__(*this, other);
}
inline Tensor & Tensor::__ixor__(const Tensor & other) {
    return dispatch_type().__ixor__(*this, other);
}
inline Tensor Tensor::__lshift__(Scalar other) const {
    return dispatch_type().__lshift__(*this, other);
}
inline Tensor Tensor::__lshift__(const Tensor & other) const {
    return dispatch_type().__lshift__(*this, other);
}
inline Tensor & Tensor::__ilshift__(Scalar other) {
    return dispatch_type().__ilshift__(*this, other);
}
inline Tensor & Tensor::__ilshift__(const Tensor & other) {
    return dispatch_type().__ilshift__(*this, other);
}
inline Tensor Tensor::__rshift__(Scalar other) const {
    return dispatch_type().__rshift__(*this, other);
}
inline Tensor Tensor::__rshift__(const Tensor & other) const {
    return dispatch_type().__rshift__(*this, other);
}
inline Tensor & Tensor::__irshift__(Scalar other) {
    return dispatch_type().__irshift__(*this, other);
}
inline Tensor & Tensor::__irshift__(const Tensor & other) {
    return dispatch_type().__irshift__(*this, other);
}
inline Tensor & Tensor::lgamma_() {
    return dispatch_type().lgamma_(*this);
}
inline Tensor & Tensor::atan2_(const Tensor & other) {
    return dispatch_type().atan2_(*this, other);
}
inline Tensor & Tensor::tril_(int64_t diagonal) {
    return dispatch_type().tril_(*this, diagonal);
}
inline Tensor & Tensor::triu_(int64_t diagonal) {
    return dispatch_type().triu_(*this, diagonal);
}
inline Tensor & Tensor::digamma_() {
    return dispatch_type().digamma_(*this);
}
inline Tensor & Tensor::polygamma_(int64_t n) {
    return dispatch_type().polygamma_(*this, n);
}
inline Tensor & Tensor::erfinv_() {
    return dispatch_type().erfinv_(*this);
}
inline Tensor & Tensor::renorm_(Scalar p, int64_t dim, Scalar maxnorm) {
    return dispatch_type().renorm_(*this, p, dim, maxnorm);
}
inline Tensor & Tensor::pow_(Scalar exponent) {
    return dispatch_type().pow_(*this, exponent);
}
inline Tensor & Tensor::pow_(const Tensor & exponent) {
    return dispatch_type().pow_(*this, exponent);
}
inline Tensor & Tensor::lerp_(const Tensor & end, Scalar weight) {
    return dispatch_type().lerp_(*this, end, weight);
}
inline Tensor & Tensor::lerp_(const Tensor & end, const Tensor & weight) {
    return dispatch_type().lerp_(*this, end, weight);
}
inline Tensor & Tensor::sign_() {
    return dispatch_type().sign_(*this);
}
inline Tensor & Tensor::fmod_(Scalar other) {
    return dispatch_type().fmod_(*this, other);
}
inline Tensor & Tensor::fmod_(const Tensor & other) {
    return dispatch_type().fmod_(*this, other);
}
inline Tensor & Tensor::remainder_(Scalar other) {
    return dispatch_type().remainder_(*this, other);
}
inline Tensor & Tensor::remainder_(const Tensor & other) {
    return dispatch_type().remainder_(*this, other);
}
inline Tensor & Tensor::addbmm_(const Tensor & batch1, const Tensor & batch2, Scalar beta, Scalar alpha) {
    return dispatch_type().addbmm_(*this, batch1, batch2, beta, alpha);
}
inline Tensor Tensor::addbmm(const Tensor & batch1, const Tensor & batch2, Scalar beta, Scalar alpha) const {
    return dispatch_type().addbmm(*this, batch1, batch2, beta, alpha);
}
inline Tensor & Tensor::addcmul_(const Tensor & tensor1, const Tensor & tensor2, Scalar value) {
    return dispatch_type().addcmul_(*this, tensor1, tensor2, value);
}
inline Tensor & Tensor::addcdiv_(const Tensor & tensor1, const Tensor & tensor2, Scalar value) {
    return dispatch_type().addcdiv_(*this, tensor1, tensor2, value);
}
inline Tensor & Tensor::random_(int64_t from, int64_t to, Generator * generator) {
    return dispatch_type().random_(*this, from, to, generator);
}
inline Tensor & Tensor::random_(int64_t to, Generator * generator) {
    return dispatch_type().random_(*this, to, generator);
}
inline Tensor & Tensor::random_(Generator * generator) {
    return dispatch_type().random_(*this, generator);
}
inline Tensor & Tensor::uniform_(double from, double to, Generator * generator) {
    return dispatch_type().uniform_(*this, from, to, generator);
}
inline Tensor & Tensor::normal_(double mean, double std, Generator * generator) {
    return dispatch_type().normal_(*this, mean, std, generator);
}
inline Tensor & Tensor::cauchy_(double median, double sigma, Generator * generator) {
    return dispatch_type().cauchy_(*this, median, sigma, generator);
}
inline Tensor & Tensor::log_normal_(double mean, double std, Generator * generator) {
    return dispatch_type().log_normal_(*this, mean, std, generator);
}
inline Tensor & Tensor::exponential_(double lambd, Generator * generator) {
    return dispatch_type().exponential_(*this, lambd, generator);
}
inline Tensor & Tensor::geometric_(double p, Generator * generator) {
    return dispatch_type().geometric_(*this, p, generator);
}
inline Tensor Tensor::diag(int64_t diagonal) const {
    return dispatch_type().diag(*this, diagonal);
}
inline Tensor Tensor::cross(const Tensor & other, c10::optional<int64_t> dim) const {
    return dispatch_type().cross(*this, other, dim);
}
inline Tensor Tensor::triu(int64_t diagonal) const {
    return dispatch_type().triu(*this, diagonal);
}
inline Tensor Tensor::tril(int64_t diagonal) const {
    return dispatch_type().tril(*this, diagonal);
}
inline Tensor Tensor::trace() const {
    return dispatch_type().trace(*this);
}
inline Tensor Tensor::ne(Scalar other) const {
    return dispatch_type().ne(*this, other);
}
inline Tensor Tensor::ne(const Tensor & other) const {
    return dispatch_type().ne(*this, other);
}
inline Tensor Tensor::eq(Scalar other) const {
    return dispatch_type().eq(*this, other);
}
inline Tensor Tensor::eq(const Tensor & other) const {
    return dispatch_type().eq(*this, other);
}
inline Tensor Tensor::ge(Scalar other) const {
    return dispatch_type().ge(*this, other);
}
inline Tensor Tensor::ge(const Tensor & other) const {
    return dispatch_type().ge(*this, other);
}
inline Tensor Tensor::le(Scalar other) const {
    return dispatch_type().le(*this, other);
}
inline Tensor Tensor::le(const Tensor & other) const {
    return dispatch_type().le(*this, other);
}
inline Tensor Tensor::gt(Scalar other) const {
    return dispatch_type().gt(*this, other);
}
inline Tensor Tensor::gt(const Tensor & other) const {
    return dispatch_type().gt(*this, other);
}
inline Tensor Tensor::lt(Scalar other) const {
    return dispatch_type().lt(*this, other);
}
inline Tensor Tensor::lt(const Tensor & other) const {
    return dispatch_type().lt(*this, other);
}
inline Tensor Tensor::take(const Tensor & index) const {
    return dispatch_type().take(*this, index);
}
inline Tensor Tensor::index_select(int64_t dim, const Tensor & index) const {
    return dispatch_type().index_select(*this, dim, index);
}
inline Tensor Tensor::masked_select(const Tensor & mask) const {
    return dispatch_type().masked_select(*this, mask);
}
inline Tensor Tensor::nonzero() const {
    return dispatch_type().nonzero(*this);
}
inline Tensor Tensor::gather(int64_t dim, const Tensor & index, bool sparse_grad) const {
    return dispatch_type().gather(*this, dim, index, sparse_grad);
}
inline Tensor Tensor::addcmul(const Tensor & tensor1, const Tensor & tensor2, Scalar value) const {
    return dispatch_type().addcmul(*this, tensor1, tensor2, value);
}
inline Tensor Tensor::addcdiv(const Tensor & tensor1, const Tensor & tensor2, Scalar value) const {
    return dispatch_type().addcdiv(*this, tensor1, tensor2, value);
}
inline std::tuple<Tensor,Tensor> Tensor::gels(const Tensor & A) const {
    return dispatch_type().gels(*this, A);
}
inline std::tuple<Tensor,Tensor> Tensor::triangular_solve(const Tensor & A, bool upper, bool transpose, bool unitriangular) const {
    return dispatch_type().triangular_solve(*this, A, upper, transpose, unitriangular);
}
inline std::tuple<Tensor,Tensor> Tensor::symeig(bool eigenvectors, bool upper) const {
    return dispatch_type().symeig(*this, eigenvectors, upper);
}
inline std::tuple<Tensor,Tensor> Tensor::eig(bool eigenvectors) const {
    return dispatch_type().eig(*this, eigenvectors);
}
inline std::tuple<Tensor,Tensor,Tensor> Tensor::svd(bool some, bool compute_uv) const {
    return dispatch_type().svd(*this, some, compute_uv);
}
inline Tensor Tensor::cholesky(bool upper) const {
    return dispatch_type().cholesky(*this, upper);
}
inline Tensor Tensor::cholesky_solve(const Tensor & input2, bool upper) const {
    return dispatch_type().cholesky_solve(*this, input2, upper);
}
inline std::tuple<Tensor,Tensor> Tensor::solve(const Tensor & A) const {
    return dispatch_type().solve(*this, A);
}
inline Tensor Tensor::cholesky_inverse(bool upper) const {
    return dispatch_type().cholesky_inverse(*this, upper);
}
inline std::tuple<Tensor,Tensor> Tensor::pstrf(bool upper, Scalar tol) const {
    return dispatch_type().pstrf(*this, upper, tol);
}
inline std::tuple<Tensor,Tensor> Tensor::qr() const {
    return dispatch_type().qr(*this);
}
inline std::tuple<Tensor,Tensor> Tensor::geqrf() const {
    return dispatch_type().geqrf(*this);
}
inline Tensor Tensor::orgqr(const Tensor & input2) const {
    return dispatch_type().orgqr(*this, input2);
}
inline Tensor Tensor::ormqr(const Tensor & input2, const Tensor & input3, bool left, bool transpose) const {
    return dispatch_type().ormqr(*this, input2, input3, left, transpose);
}
inline Tensor Tensor::lu_solve(const Tensor & LU_data, const Tensor & LU_pivots) const {
    return dispatch_type().lu_solve(*this, LU_data, LU_pivots);
}
inline Tensor Tensor::multinomial(int64_t num_samples, bool replacement, Generator * generator) const {
    return dispatch_type().multinomial(*this, num_samples, replacement, generator);
}
inline Tensor Tensor::lgamma() const {
    return dispatch_type().lgamma(*this);
}
inline Tensor Tensor::digamma() const {
    return dispatch_type().digamma(*this);
}
inline Tensor Tensor::polygamma(int64_t n) const {
    return dispatch_type().polygamma(n, *this);
}
inline Tensor Tensor::erfinv() const {
    return dispatch_type().erfinv(*this);
}
inline Tensor Tensor::dist(const Tensor & other, Scalar p) const {
    return dispatch_type().dist(*this, other, p);
}
inline Tensor Tensor::atan2(const Tensor & other) const {
    return dispatch_type().atan2(*this, other);
}
inline Tensor Tensor::lerp(const Tensor & end, Scalar weight) const {
    return dispatch_type().lerp(*this, end, weight);
}
inline Tensor Tensor::lerp(const Tensor & end, const Tensor & weight) const {
    return dispatch_type().lerp(*this, end, weight);
}
inline Tensor Tensor::histc(int64_t bins, Scalar min, Scalar max) const {
    return dispatch_type().histc(*this, bins, min, max);
}
inline Tensor Tensor::sign() const {
    return dispatch_type().sign(*this);
}
inline Tensor Tensor::fmod(Scalar other) const {
    return dispatch_type().fmod(*this, other);
}
inline Tensor Tensor::fmod(const Tensor & other) const {
    return dispatch_type().fmod(*this, other);
}
inline Tensor Tensor::remainder(Scalar other) const {
    return dispatch_type().remainder(*this, other);
}
inline Tensor Tensor::remainder(const Tensor & other) const {
    return dispatch_type().remainder(*this, other);
}
inline Tensor Tensor::min(const Tensor & other) const {
    return dispatch_type().min(*this, other);
}
inline Tensor Tensor::min() const {
    return dispatch_type().min(*this);
}
inline Tensor Tensor::max(const Tensor & other) const {
    return dispatch_type().max(*this, other);
}
inline Tensor Tensor::max() const {
    return dispatch_type().max(*this);
}
inline Tensor Tensor::median() const {
    return dispatch_type().median(*this);
}
inline std::tuple<Tensor,Tensor> Tensor::sort(int64_t dim, bool descending) const {
    return dispatch_type().sort(*this, dim, descending);
}
inline Tensor Tensor::argsort(int64_t dim, bool descending) const {
    return dispatch_type().argsort(*this, dim, descending);
}
inline std::tuple<Tensor,Tensor> Tensor::topk(int64_t k, int64_t dim, bool largest, bool sorted) const {
    return dispatch_type().topk(*this, k, dim, largest, sorted);
}
inline Tensor Tensor::all() const {
    return dispatch_type().all(*this);
}
inline Tensor Tensor::any() const {
    return dispatch_type().any(*this);
}
inline Tensor Tensor::renorm(Scalar p, int64_t dim, Scalar maxnorm) const {
    return dispatch_type().renorm(*this, p, dim, maxnorm);
}
inline Tensor Tensor::unfold(int64_t dimension, int64_t size, int64_t step) const {
    return dispatch_type().unfold(*this, dimension, size, step);
}
inline bool Tensor::equal(const Tensor & other) const {
    return dispatch_type().equal(*this, other);
}
inline Tensor Tensor::pow(const Tensor & exponent) const {
    return dispatch_type().pow(*this, exponent);
}
inline Tensor Tensor::alias() const {
    return dispatch_type().alias(*this);
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
    AT_CHECK(                                    \
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
