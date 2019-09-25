#pragma once

// @generated from tools/autograd/templates/python_variable_methods_dispatch.h

#include <torch/csrc/utils/auto_gil.h>
#include <torch/csrc/utils/cuda_lazy_init.h>

#include <ATen/ATen.h>

// Contains inline wrappers around ATen functions that release the GIL and
// switch to the correct CUDA device.

namespace torch { namespace autograd {

using at::Tensor;
using at::Scalar;
using at::TensorList;
using at::IntArrayRef;
using at::Generator;
using at::Storage;

inline Tensor dispatch___and__(Tensor & self, const Tensor & other) {

  AutoNoGIL no_gil;
  return self.__and__(other);
}
inline Tensor dispatch___and__(Tensor & self, Scalar other) {

  AutoNoGIL no_gil;
  return self.__and__(other);
}
inline Tensor dispatch___iand__(Tensor & self, const Tensor & other) {

  AutoNoGIL no_gil;
  return self.__iand__(other);
}
inline Tensor dispatch___iand__(Tensor & self, Scalar other) {

  AutoNoGIL no_gil;
  return self.__iand__(other);
}
inline Tensor dispatch___ilshift__(Tensor & self, const Tensor & other) {

  AutoNoGIL no_gil;
  return self.__ilshift__(other);
}
inline Tensor dispatch___ilshift__(Tensor & self, Scalar other) {

  AutoNoGIL no_gil;
  return self.__ilshift__(other);
}
inline Tensor dispatch___ior__(Tensor & self, const Tensor & other) {

  AutoNoGIL no_gil;
  return self.__ior__(other);
}
inline Tensor dispatch___ior__(Tensor & self, Scalar other) {

  AutoNoGIL no_gil;
  return self.__ior__(other);
}
inline Tensor dispatch___irshift__(Tensor & self, const Tensor & other) {

  AutoNoGIL no_gil;
  return self.__irshift__(other);
}
inline Tensor dispatch___irshift__(Tensor & self, Scalar other) {

  AutoNoGIL no_gil;
  return self.__irshift__(other);
}
inline Tensor dispatch___ixor__(Tensor & self, const Tensor & other) {

  AutoNoGIL no_gil;
  return self.__ixor__(other);
}
inline Tensor dispatch___ixor__(Tensor & self, Scalar other) {

  AutoNoGIL no_gil;
  return self.__ixor__(other);
}
inline Tensor dispatch___lshift__(Tensor & self, const Tensor & other) {

  AutoNoGIL no_gil;
  return self.__lshift__(other);
}
inline Tensor dispatch___lshift__(Tensor & self, Scalar other) {

  AutoNoGIL no_gil;
  return self.__lshift__(other);
}
inline Tensor dispatch___or__(Tensor & self, const Tensor & other) {

  AutoNoGIL no_gil;
  return self.__or__(other);
}
inline Tensor dispatch___or__(Tensor & self, Scalar other) {

  AutoNoGIL no_gil;
  return self.__or__(other);
}
inline Tensor dispatch___rshift__(Tensor & self, const Tensor & other) {

  AutoNoGIL no_gil;
  return self.__rshift__(other);
}
inline Tensor dispatch___rshift__(Tensor & self, Scalar other) {

  AutoNoGIL no_gil;
  return self.__rshift__(other);
}
inline Tensor dispatch___xor__(Tensor & self, const Tensor & other) {

  AutoNoGIL no_gil;
  return self.__xor__(other);
}
inline Tensor dispatch___xor__(Tensor & self, Scalar other) {

  AutoNoGIL no_gil;
  return self.__xor__(other);
}
inline Tensor dispatch__coalesced_(Tensor & self, bool coalesced) {

  AutoNoGIL no_gil;
  return self._coalesced_(coalesced);
}
inline int64_t dispatch__dimI(Tensor & self) {

  AutoNoGIL no_gil;
  return self._dimI();
}
inline int64_t dispatch__dimV(Tensor & self) {

  AutoNoGIL no_gil;
  return self._dimV();
}
inline Tensor dispatch__indices(Tensor & self) {

  AutoNoGIL no_gil;
  return self._indices();
}
inline int64_t dispatch__nnz(Tensor & self) {

  AutoNoGIL no_gil;
  return self._nnz();
}
inline Tensor dispatch__values(Tensor & self) {

  AutoNoGIL no_gil;
  return self._values();
}
inline Tensor dispatch_abs(Tensor & self) {

  AutoNoGIL no_gil;
  return self.abs();
}
inline Tensor dispatch_abs_(Tensor & self) {

  AutoNoGIL no_gil;
  return self.abs_();
}
inline Tensor dispatch_acos(Tensor & self) {

  AutoNoGIL no_gil;
  return self.acos();
}
inline Tensor dispatch_acos_(Tensor & self) {

  AutoNoGIL no_gil;
  return self.acos_();
}
inline Tensor dispatch_add(Tensor & self, Scalar alpha, const Tensor & other) {

  AutoNoGIL no_gil;
  return self.add(other, alpha);
}
inline Tensor dispatch_add(Tensor & self, const Tensor & other, Scalar alpha) {

  AutoNoGIL no_gil;
  return self.add(other, alpha);
}
inline Tensor dispatch_add_(Tensor & self, Scalar alpha, const Tensor & other) {

  AutoNoGIL no_gil;
  return self.add_(other, alpha);
}
inline Tensor dispatch_add_(Tensor & self, const Tensor & other, Scalar alpha) {

  AutoNoGIL no_gil;
  return self.add_(other, alpha);
}
inline Tensor dispatch_addbmm(Scalar beta, Tensor & self, Scalar alpha, const Tensor & batch1, const Tensor & batch2) {

  AutoNoGIL no_gil;
  return self.addbmm(batch1, batch2, beta, alpha);
}
inline Tensor dispatch_addbmm(Scalar beta, Tensor & self, const Tensor & batch1, const Tensor & batch2) {

  AutoNoGIL no_gil;
  return self.addbmm(batch1, batch2, beta, 1);
}
inline Tensor dispatch_addbmm(Tensor & self, const Tensor & batch1, const Tensor & batch2, Scalar beta, Scalar alpha) {

  AutoNoGIL no_gil;
  return self.addbmm(batch1, batch2, beta, alpha);
}
inline Tensor dispatch_addbmm_(Scalar beta, Tensor & self, Scalar alpha, const Tensor & batch1, const Tensor & batch2) {

  AutoNoGIL no_gil;
  return self.addbmm_(batch1, batch2, beta, alpha);
}
inline Tensor dispatch_addbmm_(Scalar beta, Tensor & self, const Tensor & batch1, const Tensor & batch2) {

  AutoNoGIL no_gil;
  return self.addbmm_(batch1, batch2, beta, 1);
}
inline Tensor dispatch_addbmm_(Tensor & self, const Tensor & batch1, const Tensor & batch2, Scalar beta, Scalar alpha) {

  AutoNoGIL no_gil;
  return self.addbmm_(batch1, batch2, beta, alpha);
}
inline Tensor dispatch_addcdiv(Tensor & self, Scalar value, const Tensor & tensor1, const Tensor & tensor2) {

  AutoNoGIL no_gil;
  return self.addcdiv(tensor1, tensor2, value);
}
inline Tensor dispatch_addcdiv(Tensor & self, const Tensor & tensor1, const Tensor & tensor2, Scalar value) {

  AutoNoGIL no_gil;
  return self.addcdiv(tensor1, tensor2, value);
}
inline Tensor dispatch_addcdiv_(Tensor & self, Scalar value, const Tensor & tensor1, const Tensor & tensor2) {

  AutoNoGIL no_gil;
  return self.addcdiv_(tensor1, tensor2, value);
}
inline Tensor dispatch_addcdiv_(Tensor & self, const Tensor & tensor1, const Tensor & tensor2, Scalar value) {

  AutoNoGIL no_gil;
  return self.addcdiv_(tensor1, tensor2, value);
}
inline Tensor dispatch_addcmul(Tensor & self, Scalar value, const Tensor & tensor1, const Tensor & tensor2) {

  AutoNoGIL no_gil;
  return self.addcmul(tensor1, tensor2, value);
}
inline Tensor dispatch_addcmul(Tensor & self, const Tensor & tensor1, const Tensor & tensor2, Scalar value) {

  AutoNoGIL no_gil;
  return self.addcmul(tensor1, tensor2, value);
}
inline Tensor dispatch_addcmul_(Tensor & self, Scalar value, const Tensor & tensor1, const Tensor & tensor2) {

  AutoNoGIL no_gil;
  return self.addcmul_(tensor1, tensor2, value);
}
inline Tensor dispatch_addcmul_(Tensor & self, const Tensor & tensor1, const Tensor & tensor2, Scalar value) {

  AutoNoGIL no_gil;
  return self.addcmul_(tensor1, tensor2, value);
}
inline Tensor dispatch_addmm(Scalar beta, Tensor & self, Scalar alpha, const Tensor & mat1, const Tensor & mat2) {

  AutoNoGIL no_gil;
  return self.addmm(mat1, mat2, beta, alpha);
}
inline Tensor dispatch_addmm(Scalar beta, Tensor & self, const Tensor & mat1, const Tensor & mat2) {

  AutoNoGIL no_gil;
  return self.addmm(mat1, mat2, beta, 1);
}
inline Tensor dispatch_addmm(Tensor & self, const Tensor & mat1, const Tensor & mat2, Scalar beta, Scalar alpha) {

  AutoNoGIL no_gil;
  return self.addmm(mat1, mat2, beta, alpha);
}
inline Tensor dispatch_addmm_(Scalar beta, Tensor & self, Scalar alpha, const Tensor & mat1, const Tensor & mat2) {

  AutoNoGIL no_gil;
  return self.addmm_(mat1, mat2, beta, alpha);
}
inline Tensor dispatch_addmm_(Scalar beta, Tensor & self, const Tensor & mat1, const Tensor & mat2) {

  AutoNoGIL no_gil;
  return self.addmm_(mat1, mat2, beta, 1);
}
inline Tensor dispatch_addmm_(Tensor & self, const Tensor & mat1, const Tensor & mat2, Scalar beta, Scalar alpha) {

  AutoNoGIL no_gil;
  return self.addmm_(mat1, mat2, beta, alpha);
}
inline Tensor dispatch_addmv(Scalar beta, Tensor & self, Scalar alpha, const Tensor & mat, const Tensor & vec) {

  AutoNoGIL no_gil;
  return self.addmv(mat, vec, beta, alpha);
}
inline Tensor dispatch_addmv(Scalar beta, Tensor & self, const Tensor & mat, const Tensor & vec) {

  AutoNoGIL no_gil;
  return self.addmv(mat, vec, beta, 1);
}
inline Tensor dispatch_addmv(Tensor & self, const Tensor & mat, const Tensor & vec, Scalar beta, Scalar alpha) {

  AutoNoGIL no_gil;
  return self.addmv(mat, vec, beta, alpha);
}
inline Tensor dispatch_addmv_(Scalar beta, Tensor & self, Scalar alpha, const Tensor & mat, const Tensor & vec) {

  AutoNoGIL no_gil;
  return self.addmv_(mat, vec, beta, alpha);
}
inline Tensor dispatch_addmv_(Scalar beta, Tensor & self, const Tensor & mat, const Tensor & vec) {

  AutoNoGIL no_gil;
  return self.addmv_(mat, vec, beta, 1);
}
inline Tensor dispatch_addmv_(Tensor & self, const Tensor & mat, const Tensor & vec, Scalar beta, Scalar alpha) {

  AutoNoGIL no_gil;
  return self.addmv_(mat, vec, beta, alpha);
}
inline Tensor dispatch_addr(Scalar beta, Tensor & self, Scalar alpha, const Tensor & vec1, const Tensor & vec2) {

  AutoNoGIL no_gil;
  return self.addr(vec1, vec2, beta, alpha);
}
inline Tensor dispatch_addr(Scalar beta, Tensor & self, const Tensor & vec1, const Tensor & vec2) {

  AutoNoGIL no_gil;
  return self.addr(vec1, vec2, beta, 1);
}
inline Tensor dispatch_addr(Tensor & self, const Tensor & vec1, const Tensor & vec2, Scalar beta, Scalar alpha) {

  AutoNoGIL no_gil;
  return self.addr(vec1, vec2, beta, alpha);
}
inline Tensor dispatch_addr_(Scalar beta, Tensor & self, Scalar alpha, const Tensor & vec1, const Tensor & vec2) {

  AutoNoGIL no_gil;
  return self.addr_(vec1, vec2, beta, alpha);
}
inline Tensor dispatch_addr_(Scalar beta, Tensor & self, const Tensor & vec1, const Tensor & vec2) {

  AutoNoGIL no_gil;
  return self.addr_(vec1, vec2, beta, 1);
}
inline Tensor dispatch_addr_(Tensor & self, const Tensor & vec1, const Tensor & vec2, Scalar beta, Scalar alpha) {

  AutoNoGIL no_gil;
  return self.addr_(vec1, vec2, beta, alpha);
}
inline Tensor dispatch_align_as(Tensor & self, const Tensor & other) {

  AutoNoGIL no_gil;
  return self.align_as(other);
}
inline Tensor dispatch_align_to(Tensor & self, DimnameList names) {

  AutoNoGIL no_gil;
  return self.align_to(names);
}
inline Tensor dispatch_all(Tensor & self) {

  AutoNoGIL no_gil;
  return self.all();
}
inline Tensor dispatch_all(Tensor & self, Dimname dim, bool keepdim) {

  AutoNoGIL no_gil;
  return self.all(dim, keepdim);
}
inline Tensor dispatch_all(Tensor & self, int64_t dim, bool keepdim) {

  AutoNoGIL no_gil;
  return self.all(dim, keepdim);
}
inline bool dispatch_allclose(Tensor & self, const Tensor & other, double rtol, double atol, bool equal_nan) {

  AutoNoGIL no_gil;
  return self.allclose(other, rtol, atol, equal_nan);
}
inline Tensor dispatch_any(Tensor & self) {

  AutoNoGIL no_gil;
  return self.any();
}
inline Tensor dispatch_any(Tensor & self, Dimname dim, bool keepdim) {

  AutoNoGIL no_gil;
  return self.any(dim, keepdim);
}
inline Tensor dispatch_any(Tensor & self, int64_t dim, bool keepdim) {

  AutoNoGIL no_gil;
  return self.any(dim, keepdim);
}
inline Tensor dispatch_argmax(Tensor & self, c10::optional<int64_t> dim, bool keepdim) {

  AutoNoGIL no_gil;
  return self.argmax(dim, keepdim);
}
inline Tensor dispatch_argmin(Tensor & self, c10::optional<int64_t> dim, bool keepdim) {

  AutoNoGIL no_gil;
  return self.argmin(dim, keepdim);
}
inline Tensor dispatch_argsort(Tensor & self, Dimname dim, bool descending) {

  AutoNoGIL no_gil;
  return self.argsort(dim, descending);
}
inline Tensor dispatch_argsort(Tensor & self, int64_t dim, bool descending) {

  AutoNoGIL no_gil;
  return self.argsort(dim, descending);
}
inline Tensor dispatch_as_strided(Tensor & self, IntArrayRef size, IntArrayRef stride, c10::optional<int64_t> storage_offset) {

  AutoNoGIL no_gil;
  return self.as_strided(size, stride, storage_offset);
}
inline Tensor dispatch_as_strided_(Tensor & self, IntArrayRef size, IntArrayRef stride, c10::optional<int64_t> storage_offset) {

  AutoNoGIL no_gil;
  return self.as_strided_(size, stride, storage_offset);
}
inline Tensor dispatch_asin(Tensor & self) {

  AutoNoGIL no_gil;
  return self.asin();
}
inline Tensor dispatch_asin_(Tensor & self) {

  AutoNoGIL no_gil;
  return self.asin_();
}
inline Tensor dispatch_atan(Tensor & self) {

  AutoNoGIL no_gil;
  return self.atan();
}
inline Tensor dispatch_atan2(Tensor & self, const Tensor & other) {

  AutoNoGIL no_gil;
  return self.atan2(other);
}
inline Tensor dispatch_atan2_(Tensor & self, const Tensor & other) {

  AutoNoGIL no_gil;
  return self.atan2_(other);
}
inline Tensor dispatch_atan_(Tensor & self) {

  AutoNoGIL no_gil;
  return self.atan_();
}
inline void dispatch_backward(Tensor & self, const Tensor & gradient, bool keep_graph, bool create_graph) {

  AutoNoGIL no_gil;
  return self.backward(gradient, keep_graph, create_graph);
}
inline Tensor dispatch_baddbmm(Scalar beta, Tensor & self, Scalar alpha, const Tensor & batch1, const Tensor & batch2) {

  AutoNoGIL no_gil;
  return self.baddbmm(batch1, batch2, beta, alpha);
}
inline Tensor dispatch_baddbmm(Scalar beta, Tensor & self, const Tensor & batch1, const Tensor & batch2) {

  AutoNoGIL no_gil;
  return self.baddbmm(batch1, batch2, beta, 1);
}
inline Tensor dispatch_baddbmm(Tensor & self, const Tensor & batch1, const Tensor & batch2, Scalar beta, Scalar alpha) {

  AutoNoGIL no_gil;
  return self.baddbmm(batch1, batch2, beta, alpha);
}
inline Tensor dispatch_baddbmm_(Scalar beta, Tensor & self, Scalar alpha, const Tensor & batch1, const Tensor & batch2) {

  AutoNoGIL no_gil;
  return self.baddbmm_(batch1, batch2, beta, alpha);
}
inline Tensor dispatch_baddbmm_(Scalar beta, Tensor & self, const Tensor & batch1, const Tensor & batch2) {

  AutoNoGIL no_gil;
  return self.baddbmm_(batch1, batch2, beta, 1);
}
inline Tensor dispatch_baddbmm_(Tensor & self, const Tensor & batch1, const Tensor & batch2, Scalar beta, Scalar alpha) {

  AutoNoGIL no_gil;
  return self.baddbmm_(batch1, batch2, beta, alpha);
}
inline Tensor dispatch_bernoulli(Tensor & self, Generator * generator) {

  AutoNoGIL no_gil;
  return self.bernoulli(generator);
}
inline Tensor dispatch_bernoulli(Tensor & self, double p, Generator * generator) {

  AutoNoGIL no_gil;
  return self.bernoulli(p, generator);
}
inline Tensor dispatch_bernoulli_(Tensor & self, const Tensor & p, Generator * generator) {

  AutoNoGIL no_gil;
  return self.bernoulli_(p, generator);
}
inline Tensor dispatch_bernoulli_(Tensor & self, double p, Generator * generator) {

  AutoNoGIL no_gil;
  return self.bernoulli_(p, generator);
}
inline Tensor dispatch_bincount(Tensor & self, const Tensor & weights, int64_t minlength) {

  AutoNoGIL no_gil;
  return self.bincount(weights, minlength);
}
inline Tensor dispatch_bitwise_not(Tensor & self) {

  AutoNoGIL no_gil;
  return self.bitwise_not();
}
inline Tensor dispatch_bitwise_not_(Tensor & self) {

  AutoNoGIL no_gil;
  return self.bitwise_not_();
}
inline Tensor dispatch_bmm(Tensor & self, const Tensor & mat2) {

  AutoNoGIL no_gil;
  return self.bmm(mat2);
}
inline Tensor dispatch_cauchy_(Tensor & self, double median, double sigma, Generator * generator) {

  AutoNoGIL no_gil;
  return self.cauchy_(median, sigma, generator);
}
inline Tensor dispatch_ceil(Tensor & self) {

  AutoNoGIL no_gil;
  return self.ceil();
}
inline Tensor dispatch_ceil_(Tensor & self) {

  AutoNoGIL no_gil;
  return self.ceil_();
}
inline Tensor dispatch_cholesky(Tensor & self, bool upper) {

  AutoNoGIL no_gil;
  return self.cholesky(upper);
}
inline Tensor dispatch_cholesky_inverse(Tensor & self, bool upper) {

  AutoNoGIL no_gil;
  return self.cholesky_inverse(upper);
}
inline Tensor dispatch_cholesky_solve(Tensor & self, const Tensor & input2, bool upper) {

  AutoNoGIL no_gil;
  return self.cholesky_solve(input2, upper);
}
inline std::vector<Tensor> dispatch_chunk(Tensor & self, int64_t chunks, int64_t dim) {

  AutoNoGIL no_gil;
  return self.chunk(chunks, dim);
}
inline Tensor dispatch_clamp(Tensor & self, c10::optional<Scalar> min, c10::optional<Scalar> max) {

  AutoNoGIL no_gil;
  return self.clamp(min, max);
}
inline Tensor dispatch_clamp_(Tensor & self, c10::optional<Scalar> min, c10::optional<Scalar> max) {

  AutoNoGIL no_gil;
  return self.clamp_(min, max);
}
inline Tensor dispatch_clamp_max(Tensor & self, Scalar max) {

  AutoNoGIL no_gil;
  return self.clamp_max(max);
}
inline Tensor dispatch_clamp_max_(Tensor & self, Scalar max) {

  AutoNoGIL no_gil;
  return self.clamp_max_(max);
}
inline Tensor dispatch_clamp_min(Tensor & self, Scalar min) {

  AutoNoGIL no_gil;
  return self.clamp_min(min);
}
inline Tensor dispatch_clamp_min_(Tensor & self, Scalar min) {

  AutoNoGIL no_gil;
  return self.clamp_min_(min);
}
inline Tensor dispatch_clone(Tensor & self) {

  AutoNoGIL no_gil;
  return self.clone();
}
inline Tensor dispatch_coalesce(Tensor & self) {

  AutoNoGIL no_gil;
  return self.coalesce();
}
inline Tensor dispatch_cos(Tensor & self) {

  AutoNoGIL no_gil;
  return self.cos();
}
inline Tensor dispatch_cos_(Tensor & self) {

  AutoNoGIL no_gil;
  return self.cos_();
}
inline Tensor dispatch_cosh(Tensor & self) {

  AutoNoGIL no_gil;
  return self.cosh();
}
inline Tensor dispatch_cosh_(Tensor & self) {

  AutoNoGIL no_gil;
  return self.cosh_();
}
inline Tensor dispatch_cross(Tensor & self, const Tensor & other, c10::optional<int64_t> dim) {

  AutoNoGIL no_gil;
  return self.cross(other, dim);
}
inline Tensor dispatch_cumprod(Tensor & self, Dimname dim, c10::optional<ScalarType> dtype) {

  AutoNoGIL no_gil;
  return self.cumprod(dim, dtype);
}
inline Tensor dispatch_cumprod(Tensor & self, int64_t dim, c10::optional<ScalarType> dtype) {

  AutoNoGIL no_gil;
  return self.cumprod(dim, dtype);
}
inline Tensor dispatch_cumsum(Tensor & self, Dimname dim, c10::optional<ScalarType> dtype) {

  AutoNoGIL no_gil;
  return self.cumsum(dim, dtype);
}
inline Tensor dispatch_cumsum(Tensor & self, int64_t dim, c10::optional<ScalarType> dtype) {

  AutoNoGIL no_gil;
  return self.cumsum(dim, dtype);
}
inline int64_t dispatch_dense_dim(Tensor & self) {

  AutoNoGIL no_gil;
  return self.dense_dim();
}
inline Tensor dispatch_dequantize(Tensor & self) {

  AutoNoGIL no_gil;
  return self.dequantize();
}
inline Tensor dispatch_det(Tensor & self) {

  AutoNoGIL no_gil;
  return self.det();
}
inline Tensor dispatch_detach(Tensor & self) {

  AutoNoGIL no_gil;
  return self.detach();
}
inline Tensor dispatch_detach_(Tensor & self) {

  AutoNoGIL no_gil;
  return self.detach_();
}
inline Tensor dispatch_diag(Tensor & self, int64_t diagonal) {

  AutoNoGIL no_gil;
  return self.diag(diagonal);
}
inline Tensor dispatch_diag_embed(Tensor & self, int64_t offset, int64_t dim1, int64_t dim2) {

  AutoNoGIL no_gil;
  return self.diag_embed(offset, dim1, dim2);
}
inline Tensor dispatch_diagflat(Tensor & self, int64_t offset) {

  AutoNoGIL no_gil;
  return self.diagflat(offset);
}
inline Tensor dispatch_diagonal(Tensor & self, int64_t offset, int64_t dim1, int64_t dim2) {

  AutoNoGIL no_gil;
  return self.diagonal(offset, dim1, dim2);
}
inline Tensor dispatch_digamma(Tensor & self) {

  AutoNoGIL no_gil;
  return self.digamma();
}
inline Tensor dispatch_digamma_(Tensor & self) {

  AutoNoGIL no_gil;
  return self.digamma_();
}
inline Tensor dispatch_dist(Tensor & self, const Tensor & other, Scalar p) {

  AutoNoGIL no_gil;
  return self.dist(other, p);
}
inline Tensor dispatch_div(Tensor & self, const Tensor & other) {

  AutoNoGIL no_gil;
  return self.div(other);
}
inline Tensor dispatch_div_(Tensor & self, const Tensor & other) {

  AutoNoGIL no_gil;
  return self.div_(other);
}
inline Tensor dispatch_dot(Tensor & self, const Tensor & tensor) {

  AutoNoGIL no_gil;
  return self.dot(tensor);
}
inline std::tuple<Tensor,Tensor> dispatch_eig(Tensor & self, bool eigenvectors) {

  AutoNoGIL no_gil;
  return self.eig(eigenvectors);
}
inline Tensor dispatch_eq(Tensor & self, const Tensor & other) {

  AutoNoGIL no_gil;
  return self.eq(other);
}
inline Tensor dispatch_eq(Tensor & self, Scalar other) {

  AutoNoGIL no_gil;
  return self.eq(other);
}
inline Tensor dispatch_eq_(Tensor & self, const Tensor & other) {

  AutoNoGIL no_gil;
  return self.eq_(other);
}
inline Tensor dispatch_eq_(Tensor & self, Scalar other) {

  AutoNoGIL no_gil;
  return self.eq_(other);
}
inline bool dispatch_equal(Tensor & self, const Tensor & other) {

  AutoNoGIL no_gil;
  return self.equal(other);
}
inline Tensor dispatch_erf(Tensor & self) {

  AutoNoGIL no_gil;
  return self.erf();
}
inline Tensor dispatch_erf_(Tensor & self) {

  AutoNoGIL no_gil;
  return self.erf_();
}
inline Tensor dispatch_erfc(Tensor & self) {

  AutoNoGIL no_gil;
  return self.erfc();
}
inline Tensor dispatch_erfc_(Tensor & self) {

  AutoNoGIL no_gil;
  return self.erfc_();
}
inline Tensor dispatch_erfinv(Tensor & self) {

  AutoNoGIL no_gil;
  return self.erfinv();
}
inline Tensor dispatch_erfinv_(Tensor & self) {

  AutoNoGIL no_gil;
  return self.erfinv_();
}
inline Tensor dispatch_exp(Tensor & self) {

  AutoNoGIL no_gil;
  return self.exp();
}
inline Tensor dispatch_exp_(Tensor & self) {

  AutoNoGIL no_gil;
  return self.exp_();
}
inline Tensor dispatch_expand(Tensor & self, IntArrayRef size, bool implicit) {

  AutoNoGIL no_gil;
  return self.expand(size, implicit);
}
inline Tensor dispatch_expand_as(Tensor & self, const Tensor & other) {

  AutoNoGIL no_gil;
  return self.expand_as(other);
}
inline Tensor dispatch_expm1(Tensor & self) {

  AutoNoGIL no_gil;
  return self.expm1();
}
inline Tensor dispatch_expm1_(Tensor & self) {

  AutoNoGIL no_gil;
  return self.expm1_();
}
inline Tensor dispatch_exponential_(Tensor & self, double lambd, Generator * generator) {

  AutoNoGIL no_gil;
  return self.exponential_(lambd, generator);
}
inline Tensor dispatch_fft(Tensor & self, int64_t signal_ndim, bool normalized) {

  AutoNoGIL no_gil;
  return self.fft(signal_ndim, normalized);
}
inline Tensor dispatch_fill_(Tensor & self, const Tensor & value) {

  AutoNoGIL no_gil;
  return self.fill_(value);
}
inline Tensor dispatch_fill_(Tensor & self, Scalar value) {

  AutoNoGIL no_gil;
  return self.fill_(value);
}
inline Tensor dispatch_fill_diagonal_(Tensor & self, Scalar fill_value, bool wrap) {

  AutoNoGIL no_gil;
  return self.fill_diagonal_(fill_value, wrap);
}
inline Tensor dispatch_flatten(Tensor & self, Dimname start_dim, Dimname end_dim, Dimname out_dim) {

  AutoNoGIL no_gil;
  return self.flatten(start_dim, end_dim, out_dim);
}
inline Tensor dispatch_flatten(Tensor & self, DimnameList dims, Dimname out_dim) {

  AutoNoGIL no_gil;
  return self.flatten(dims, out_dim);
}
inline Tensor dispatch_flatten(Tensor & self, int64_t start_dim, int64_t end_dim, Dimname out_dim) {

  AutoNoGIL no_gil;
  return self.flatten(start_dim, end_dim, out_dim);
}
inline Tensor dispatch_flatten(Tensor & self, int64_t start_dim, int64_t end_dim) {

  AutoNoGIL no_gil;
  return self.flatten(start_dim, end_dim);
}
inline Tensor dispatch_flip(Tensor & self, IntArrayRef dims) {

  AutoNoGIL no_gil;
  return self.flip(dims);
}
inline Tensor dispatch_floor(Tensor & self) {

  AutoNoGIL no_gil;
  return self.floor();
}
inline Tensor dispatch_floor_(Tensor & self) {

  AutoNoGIL no_gil;
  return self.floor_();
}
inline Tensor dispatch_fmod(Tensor & self, const Tensor & other) {

  AutoNoGIL no_gil;
  return self.fmod(other);
}
inline Tensor dispatch_fmod(Tensor & self, Scalar other) {

  AutoNoGIL no_gil;
  return self.fmod(other);
}
inline Tensor dispatch_fmod_(Tensor & self, const Tensor & other) {

  AutoNoGIL no_gil;
  return self.fmod_(other);
}
inline Tensor dispatch_fmod_(Tensor & self, Scalar other) {

  AutoNoGIL no_gil;
  return self.fmod_(other);
}
inline Tensor dispatch_frac(Tensor & self) {

  AutoNoGIL no_gil;
  return self.frac();
}
inline Tensor dispatch_frac_(Tensor & self) {

  AutoNoGIL no_gil;
  return self.frac_();
}
inline Tensor dispatch_gather(Tensor & self, Dimname dim, const Tensor & index, bool sparse_grad) {

  AutoNoGIL no_gil;
  return self.gather(dim, index, sparse_grad);
}
inline Tensor dispatch_gather(Tensor & self, int64_t dim, const Tensor & index, bool sparse_grad) {

  AutoNoGIL no_gil;
  return self.gather(dim, index, sparse_grad);
}
inline Tensor dispatch_ge(Tensor & self, const Tensor & other) {

  AutoNoGIL no_gil;
  return self.ge(other);
}
inline Tensor dispatch_ge(Tensor & self, Scalar other) {

  AutoNoGIL no_gil;
  return self.ge(other);
}
inline Tensor dispatch_ge_(Tensor & self, const Tensor & other) {

  AutoNoGIL no_gil;
  return self.ge_(other);
}
inline Tensor dispatch_ge_(Tensor & self, Scalar other) {

  AutoNoGIL no_gil;
  return self.ge_(other);
}
inline Tensor dispatch_geometric_(Tensor & self, double p, Generator * generator) {

  AutoNoGIL no_gil;
  return self.geometric_(p, generator);
}
inline std::tuple<Tensor,Tensor> dispatch_geqrf(Tensor & self) {

  AutoNoGIL no_gil;
  return self.geqrf();
}
inline Tensor dispatch_ger(Tensor & self, const Tensor & vec2) {

  AutoNoGIL no_gil;
  return self.ger(vec2);
}
inline Tensor dispatch_gt(Tensor & self, const Tensor & other) {

  AutoNoGIL no_gil;
  return self.gt(other);
}
inline Tensor dispatch_gt(Tensor & self, Scalar other) {

  AutoNoGIL no_gil;
  return self.gt(other);
}
inline Tensor dispatch_gt_(Tensor & self, const Tensor & other) {

  AutoNoGIL no_gil;
  return self.gt_(other);
}
inline Tensor dispatch_gt_(Tensor & self, Scalar other) {

  AutoNoGIL no_gil;
  return self.gt_(other);
}
inline Tensor dispatch_hardshrink(Tensor & self, Scalar lambd) {

  AutoNoGIL no_gil;
  return self.hardshrink(lambd);
}
inline Tensor dispatch_histc(Tensor & self, int64_t bins, Scalar min, Scalar max) {

  AutoNoGIL no_gil;
  return self.histc(bins, min, max);
}
inline Tensor dispatch_ifft(Tensor & self, int64_t signal_ndim, bool normalized) {

  AutoNoGIL no_gil;
  return self.ifft(signal_ndim, normalized);
}
inline Tensor dispatch_index_add(Tensor & self, Dimname dim, const Tensor & index, const Tensor & source) {

  AutoNoGIL no_gil;
  return self.index_add(dim, index, source);
}
inline Tensor dispatch_index_add(Tensor & self, int64_t dim, const Tensor & index, const Tensor & source) {

  AutoNoGIL no_gil;
  return self.index_add(dim, index, source);
}
inline Tensor dispatch_index_add_(Tensor & self, int64_t dim, const Tensor & index, const Tensor & source) {

  AutoNoGIL no_gil;
  return self.index_add_(dim, index, source);
}
inline Tensor dispatch_index_copy(Tensor & self, Dimname dim, const Tensor & index, const Tensor & source) {

  AutoNoGIL no_gil;
  return self.index_copy(dim, index, source);
}
inline Tensor dispatch_index_copy(Tensor & self, int64_t dim, const Tensor & index, const Tensor & source) {

  AutoNoGIL no_gil;
  return self.index_copy(dim, index, source);
}
inline Tensor dispatch_index_copy_(Tensor & self, Dimname dim, const Tensor & index, const Tensor & source) {

  AutoNoGIL no_gil;
  return self.index_copy_(dim, index, source);
}
inline Tensor dispatch_index_copy_(Tensor & self, int64_t dim, const Tensor & index, const Tensor & source) {

  AutoNoGIL no_gil;
  return self.index_copy_(dim, index, source);
}
inline Tensor dispatch_index_fill(Tensor & self, Dimname dim, const Tensor & index, const Tensor & value) {

  AutoNoGIL no_gil;
  return self.index_fill(dim, index, value);
}
inline Tensor dispatch_index_fill(Tensor & self, int64_t dim, const Tensor & index, const Tensor & value) {

  AutoNoGIL no_gil;
  return self.index_fill(dim, index, value);
}
inline Tensor dispatch_index_fill(Tensor & self, Dimname dim, const Tensor & index, Scalar value) {

  AutoNoGIL no_gil;
  return self.index_fill(dim, index, value);
}
inline Tensor dispatch_index_fill(Tensor & self, int64_t dim, const Tensor & index, Scalar value) {

  AutoNoGIL no_gil;
  return self.index_fill(dim, index, value);
}
inline Tensor dispatch_index_fill_(Tensor & self, int64_t dim, const Tensor & index, const Tensor & value) {

  AutoNoGIL no_gil;
  return self.index_fill_(dim, index, value);
}
inline Tensor dispatch_index_fill_(Tensor & self, int64_t dim, const Tensor & index, Scalar value) {

  AutoNoGIL no_gil;
  return self.index_fill_(dim, index, value);
}
inline Tensor dispatch_index_put(Tensor & self, TensorList indices, const Tensor & values, bool accumulate) {

  AutoNoGIL no_gil;
  return self.index_put(indices, values, accumulate);
}
inline Tensor dispatch_index_put_(Tensor & self, TensorList indices, const Tensor & values, bool accumulate) {

  AutoNoGIL no_gil;
  return self.index_put_(indices, values, accumulate);
}
inline Tensor dispatch_index_select(Tensor & self, Dimname dim, const Tensor & index) {

  AutoNoGIL no_gil;
  return self.index_select(dim, index);
}
inline Tensor dispatch_index_select(Tensor & self, int64_t dim, const Tensor & index) {

  AutoNoGIL no_gil;
  return self.index_select(dim, index);
}
inline Tensor dispatch_indices(Tensor & self) {

  AutoNoGIL no_gil;
  return self.indices();
}
inline Tensor dispatch_int_repr(Tensor & self) {

  AutoNoGIL no_gil;
  return self.int_repr();
}
inline Tensor dispatch_inverse(Tensor & self) {

  AutoNoGIL no_gil;
  return self.inverse();
}
inline Tensor dispatch_irfft(Tensor & self, int64_t signal_ndim, bool normalized, bool onesided, IntArrayRef signal_sizes) {

  AutoNoGIL no_gil;
  return self.irfft(signal_ndim, normalized, onesided, signal_sizes);
}
inline bool dispatch_is_coalesced(Tensor & self) {

  AutoNoGIL no_gil;
  return self.is_coalesced();
}
inline bool dispatch_is_complex(Tensor & self) {

  AutoNoGIL no_gil;
  return self.is_complex();
}
inline bool dispatch_is_distributed(Tensor & self) {

  AutoNoGIL no_gil;
  return self.is_distributed();
}
inline bool dispatch_is_floating_point(Tensor & self) {

  AutoNoGIL no_gil;
  return self.is_floating_point();
}
inline bool dispatch_is_nonzero(Tensor & self) {

  AutoNoGIL no_gil;
  return self.is_nonzero();
}
inline bool dispatch_is_pinned(Tensor & self) {

  AutoNoGIL no_gil;
  return self.is_pinned();
}
inline bool dispatch_is_same_size(Tensor & self, const Tensor & other) {

  AutoNoGIL no_gil;
  return self.is_same_size(other);
}
inline bool dispatch_is_set_to(Tensor & self, const Tensor & tensor) {

  AutoNoGIL no_gil;
  return self.is_set_to(tensor);
}
inline bool dispatch_is_signed(Tensor & self) {

  AutoNoGIL no_gil;
  return self.is_signed();
}
inline Tensor dispatch_isclose(Tensor & self, const Tensor & other, double rtol, double atol, bool equal_nan) {

  AutoNoGIL no_gil;
  return self.isclose(other, rtol, atol, equal_nan);
}
inline std::tuple<Tensor,Tensor> dispatch_kthvalue(Tensor & self, int64_t k, Dimname dim, bool keepdim) {

  AutoNoGIL no_gil;
  return self.kthvalue(k, dim, keepdim);
}
inline std::tuple<Tensor,Tensor> dispatch_kthvalue(Tensor & self, int64_t k, int64_t dim, bool keepdim) {

  AutoNoGIL no_gil;
  return self.kthvalue(k, dim, keepdim);
}
inline Tensor dispatch_le(Tensor & self, const Tensor & other) {

  AutoNoGIL no_gil;
  return self.le(other);
}
inline Tensor dispatch_le(Tensor & self, Scalar other) {

  AutoNoGIL no_gil;
  return self.le(other);
}
inline Tensor dispatch_le_(Tensor & self, const Tensor & other) {

  AutoNoGIL no_gil;
  return self.le_(other);
}
inline Tensor dispatch_le_(Tensor & self, Scalar other) {

  AutoNoGIL no_gil;
  return self.le_(other);
}
inline Tensor dispatch_lerp(Tensor & self, const Tensor & end, const Tensor & weight) {

  AutoNoGIL no_gil;
  return self.lerp(end, weight);
}
inline Tensor dispatch_lerp(Tensor & self, const Tensor & end, Scalar weight) {

  AutoNoGIL no_gil;
  return self.lerp(end, weight);
}
inline Tensor dispatch_lerp_(Tensor & self, const Tensor & end, const Tensor & weight) {

  AutoNoGIL no_gil;
  return self.lerp_(end, weight);
}
inline Tensor dispatch_lerp_(Tensor & self, const Tensor & end, Scalar weight) {

  AutoNoGIL no_gil;
  return self.lerp_(end, weight);
}
inline Tensor dispatch_lgamma(Tensor & self) {

  AutoNoGIL no_gil;
  return self.lgamma();
}
inline Tensor dispatch_lgamma_(Tensor & self) {

  AutoNoGIL no_gil;
  return self.lgamma_();
}
inline Tensor dispatch_log(Tensor & self) {

  AutoNoGIL no_gil;
  return self.log();
}
inline Tensor dispatch_log10(Tensor & self) {

  AutoNoGIL no_gil;
  return self.log10();
}
inline Tensor dispatch_log10_(Tensor & self) {

  AutoNoGIL no_gil;
  return self.log10_();
}
inline Tensor dispatch_log1p(Tensor & self) {

  AutoNoGIL no_gil;
  return self.log1p();
}
inline Tensor dispatch_log1p_(Tensor & self) {

  AutoNoGIL no_gil;
  return self.log1p_();
}
inline Tensor dispatch_log2(Tensor & self) {

  AutoNoGIL no_gil;
  return self.log2();
}
inline Tensor dispatch_log2_(Tensor & self) {

  AutoNoGIL no_gil;
  return self.log2_();
}
inline Tensor dispatch_log_(Tensor & self) {

  AutoNoGIL no_gil;
  return self.log_();
}
inline Tensor dispatch_log_normal_(Tensor & self, double mean, double std, Generator * generator) {

  AutoNoGIL no_gil;
  return self.log_normal_(mean, std, generator);
}
inline Tensor dispatch_log_softmax(Tensor & self, Dimname dim, c10::optional<ScalarType> dtype) {

  AutoNoGIL no_gil;
  return self.log_softmax(dim, dtype);
}
inline Tensor dispatch_log_softmax(Tensor & self, int64_t dim, c10::optional<ScalarType> dtype) {

  AutoNoGIL no_gil;
  return self.log_softmax(dim, dtype);
}
inline Tensor dispatch_logdet(Tensor & self) {

  AutoNoGIL no_gil;
  return self.logdet();
}
inline Tensor dispatch_logical_not(Tensor & self) {

  AutoNoGIL no_gil;
  return self.logical_not();
}
inline Tensor dispatch_logical_not_(Tensor & self) {

  AutoNoGIL no_gil;
  return self.logical_not_();
}
inline Tensor dispatch_logical_xor(Tensor & self, const Tensor & other) {

  AutoNoGIL no_gil;
  return self.logical_xor(other);
}
inline Tensor dispatch_logical_xor_(Tensor & self, const Tensor & other) {

  AutoNoGIL no_gil;
  return self.logical_xor_(other);
}
inline Tensor dispatch_logsumexp(Tensor & self, DimnameList dim, bool keepdim) {

  AutoNoGIL no_gil;
  return self.logsumexp(dim, keepdim);
}
inline Tensor dispatch_logsumexp(Tensor & self, IntArrayRef dim, bool keepdim) {

  AutoNoGIL no_gil;
  return self.logsumexp(dim, keepdim);
}
inline std::tuple<Tensor,Tensor> dispatch_lstsq(Tensor & self, const Tensor & A) {

  AutoNoGIL no_gil;
  return self.lstsq(A);
}
inline Tensor dispatch_lt(Tensor & self, const Tensor & other) {

  AutoNoGIL no_gil;
  return self.lt(other);
}
inline Tensor dispatch_lt(Tensor & self, Scalar other) {

  AutoNoGIL no_gil;
  return self.lt(other);
}
inline Tensor dispatch_lt_(Tensor & self, const Tensor & other) {

  AutoNoGIL no_gil;
  return self.lt_(other);
}
inline Tensor dispatch_lt_(Tensor & self, Scalar other) {

  AutoNoGIL no_gil;
  return self.lt_(other);
}
inline Tensor dispatch_lu_solve(Tensor & self, const Tensor & LU_data, const Tensor & LU_pivots) {

  AutoNoGIL no_gil;
  return self.lu_solve(LU_data, LU_pivots);
}
inline Tensor dispatch_masked_fill(Tensor & self, const Tensor & mask, const Tensor & value) {

  AutoNoGIL no_gil;
  return self.masked_fill(mask, value);
}
inline Tensor dispatch_masked_fill(Tensor & self, const Tensor & mask, Scalar value) {

  AutoNoGIL no_gil;
  return self.masked_fill(mask, value);
}
inline Tensor dispatch_masked_fill_(Tensor & self, const Tensor & mask, const Tensor & value) {

  AutoNoGIL no_gil;
  return self.masked_fill_(mask, value);
}
inline Tensor dispatch_masked_fill_(Tensor & self, const Tensor & mask, Scalar value) {

  AutoNoGIL no_gil;
  return self.masked_fill_(mask, value);
}
inline Tensor dispatch_masked_scatter(Tensor & self, const Tensor & mask, const Tensor & source) {

  AutoNoGIL no_gil;
  return self.masked_scatter(mask, source);
}
inline Tensor dispatch_masked_scatter_(Tensor & self, const Tensor & mask, const Tensor & source) {

  AutoNoGIL no_gil;
  return self.masked_scatter_(mask, source);
}
inline Tensor dispatch_masked_select(Tensor & self, const Tensor & mask) {

  AutoNoGIL no_gil;
  return self.masked_select(mask);
}
inline Tensor dispatch_matmul(Tensor & self, const Tensor & other) {

  AutoNoGIL no_gil;
  return self.matmul(other);
}
inline Tensor dispatch_matrix_power(Tensor & self, int64_t n) {

  AutoNoGIL no_gil;
  return self.matrix_power(n);
}
inline Tensor dispatch_max(Tensor & self) {

  AutoNoGIL no_gil;
  return self.max();
}
inline std::tuple<Tensor,Tensor> dispatch_max(Tensor & self, Dimname dim, bool keepdim) {

  AutoNoGIL no_gil;
  return self.max(dim, keepdim);
}
inline Tensor dispatch_max(Tensor & self, const Tensor & other) {

  AutoNoGIL no_gil;
  return self.max(other);
}
inline std::tuple<Tensor,Tensor> dispatch_max(Tensor & self, int64_t dim, bool keepdim) {

  AutoNoGIL no_gil;
  return self.max(dim, keepdim);
}
inline Tensor dispatch_mean(Tensor & self, c10::optional<ScalarType> dtype) {

  AutoNoGIL no_gil;
  return self.mean(dtype);
}
inline Tensor dispatch_mean(Tensor & self, DimnameList dim, bool keepdim, c10::optional<ScalarType> dtype) {

  AutoNoGIL no_gil;
  return self.mean(dim, keepdim, dtype);
}
inline Tensor dispatch_mean(Tensor & self, IntArrayRef dim, bool keepdim, c10::optional<ScalarType> dtype) {

  AutoNoGIL no_gil;
  return self.mean(dim, keepdim, dtype);
}
inline Tensor dispatch_median(Tensor & self) {

  AutoNoGIL no_gil;
  return self.median();
}
inline std::tuple<Tensor,Tensor> dispatch_median(Tensor & self, Dimname dim, bool keepdim) {

  AutoNoGIL no_gil;
  return self.median(dim, keepdim);
}
inline std::tuple<Tensor,Tensor> dispatch_median(Tensor & self, int64_t dim, bool keepdim) {

  AutoNoGIL no_gil;
  return self.median(dim, keepdim);
}
inline Tensor dispatch_min(Tensor & self) {

  AutoNoGIL no_gil;
  return self.min();
}
inline std::tuple<Tensor,Tensor> dispatch_min(Tensor & self, Dimname dim, bool keepdim) {

  AutoNoGIL no_gil;
  return self.min(dim, keepdim);
}
inline Tensor dispatch_min(Tensor & self, const Tensor & other) {

  AutoNoGIL no_gil;
  return self.min(other);
}
inline std::tuple<Tensor,Tensor> dispatch_min(Tensor & self, int64_t dim, bool keepdim) {

  AutoNoGIL no_gil;
  return self.min(dim, keepdim);
}
inline Tensor dispatch_mm(Tensor & self, const Tensor & mat2) {

  AutoNoGIL no_gil;
  return self.mm(mat2);
}
inline std::tuple<Tensor,Tensor> dispatch_mode(Tensor & self, Dimname dim, bool keepdim) {

  AutoNoGIL no_gil;
  return self.mode(dim, keepdim);
}
inline std::tuple<Tensor,Tensor> dispatch_mode(Tensor & self, int64_t dim, bool keepdim) {

  AutoNoGIL no_gil;
  return self.mode(dim, keepdim);
}
inline Tensor dispatch_mul(Tensor & self, const Tensor & other) {

  AutoNoGIL no_gil;
  return self.mul(other);
}
inline Tensor dispatch_mul_(Tensor & self, const Tensor & other) {

  AutoNoGIL no_gil;
  return self.mul_(other);
}
inline Tensor dispatch_multinomial(Tensor & self, int64_t num_samples, bool replacement, Generator * generator) {

  AutoNoGIL no_gil;
  return self.multinomial(num_samples, replacement, generator);
}
inline Tensor dispatch_mv(Tensor & self, const Tensor & vec) {

  AutoNoGIL no_gil;
  return self.mv(vec);
}
inline Tensor dispatch_mvlgamma(Tensor & self, int64_t p) {

  AutoNoGIL no_gil;
  return self.mvlgamma(p);
}
inline Tensor dispatch_mvlgamma_(Tensor & self, int64_t p) {

  AutoNoGIL no_gil;
  return self.mvlgamma_(p);
}
inline Tensor dispatch_narrow(Tensor & self, int64_t dim, int64_t start, int64_t length) {

  AutoNoGIL no_gil;
  return self.narrow(dim, start, length);
}
inline Tensor dispatch_narrow_copy(Tensor & self, int64_t dim, int64_t start, int64_t length) {

  AutoNoGIL no_gil;
  return self.narrow_copy(dim, start, length);
}
inline Tensor dispatch_ne(Tensor & self, const Tensor & other) {

  AutoNoGIL no_gil;
  return self.ne(other);
}
inline Tensor dispatch_ne(Tensor & self, Scalar other) {

  AutoNoGIL no_gil;
  return self.ne(other);
}
inline Tensor dispatch_ne_(Tensor & self, const Tensor & other) {

  AutoNoGIL no_gil;
  return self.ne_(other);
}
inline Tensor dispatch_ne_(Tensor & self, Scalar other) {

  AutoNoGIL no_gil;
  return self.ne_(other);
}
inline Tensor dispatch_neg(Tensor & self) {

  AutoNoGIL no_gil;
  return self.neg();
}
inline Tensor dispatch_neg_(Tensor & self) {

  AutoNoGIL no_gil;
  return self.neg_();
}
inline Tensor dispatch_new_empty(Tensor & self, IntArrayRef size, const TensorOptions & options) {
  torch::utils::maybe_initialize_cuda(options);
  AutoNoGIL no_gil;
  return self.new_empty(size, options);
}
inline Tensor dispatch_new_full(Tensor & self, IntArrayRef size, Scalar fill_value, const TensorOptions & options) {
  torch::utils::maybe_initialize_cuda(options);
  AutoNoGIL no_gil;
  return self.new_full(size, fill_value, options);
}
inline Tensor dispatch_norm(Tensor & self, Scalar p) {

  AutoNoGIL no_gil;
  return self.norm(p);
}
inline Tensor dispatch_norm(Tensor & self, c10::optional<Scalar> p, ScalarType dtype) {

  AutoNoGIL no_gil;
  return self.norm(p, dtype);
}
inline Tensor dispatch_norm(Tensor & self, c10::optional<Scalar> p, DimnameList dim, bool keepdim, ScalarType dtype) {

  AutoNoGIL no_gil;
  return self.norm(p, dim, keepdim, dtype);
}
inline Tensor dispatch_norm(Tensor & self, c10::optional<Scalar> p, DimnameList dim, bool keepdim) {

  AutoNoGIL no_gil;
  return self.norm(p, dim, keepdim);
}
inline Tensor dispatch_norm(Tensor & self, c10::optional<Scalar> p, IntArrayRef dim, bool keepdim, ScalarType dtype) {

  AutoNoGIL no_gil;
  return self.norm(p, dim, keepdim, dtype);
}
inline Tensor dispatch_norm(Tensor & self, c10::optional<Scalar> p, IntArrayRef dim, bool keepdim) {

  AutoNoGIL no_gil;
  return self.norm(p, dim, keepdim);
}
inline Tensor dispatch_normal_(Tensor & self, double mean, double std, Generator * generator) {

  AutoNoGIL no_gil;
  return self.normal_(mean, std, generator);
}
inline int64_t dispatch_numel(Tensor & self) {

  AutoNoGIL no_gil;
  return self.numel();
}
inline Tensor dispatch_orgqr(Tensor & self, const Tensor & input2) {

  AutoNoGIL no_gil;
  return self.orgqr(input2);
}
inline Tensor dispatch_ormqr(Tensor & self, const Tensor & input2, const Tensor & input3, bool left, bool transpose) {

  AutoNoGIL no_gil;
  return self.ormqr(input2, input3, left, transpose);
}
inline Tensor dispatch_permute(Tensor & self, IntArrayRef dims) {

  AutoNoGIL no_gil;
  return self.permute(dims);
}
inline Tensor dispatch_pin_memory(Tensor & self) {

  AutoNoGIL no_gil;
  return self.pin_memory();
}
inline Tensor dispatch_pinverse(Tensor & self, double rcond) {

  AutoNoGIL no_gil;
  return self.pinverse(rcond);
}
inline Tensor dispatch_polygamma(int64_t n, Tensor & self) {

  AutoNoGIL no_gil;
  return self.polygamma(n);
}
inline Tensor dispatch_polygamma_(Tensor & self, int64_t n) {

  AutoNoGIL no_gil;
  return self.polygamma_(n);
}
inline Tensor dispatch_pow(Tensor & self, const Tensor & exponent) {

  AutoNoGIL no_gil;
  return self.pow(exponent);
}
inline Tensor dispatch_pow(Tensor & self, Scalar exponent) {

  AutoNoGIL no_gil;
  return self.pow(exponent);
}
inline Tensor dispatch_pow_(Tensor & self, const Tensor & exponent) {

  AutoNoGIL no_gil;
  return self.pow_(exponent);
}
inline Tensor dispatch_pow_(Tensor & self, Scalar exponent) {

  AutoNoGIL no_gil;
  return self.pow_(exponent);
}
inline Tensor dispatch_prelu(Tensor & self, const Tensor & weight) {

  AutoNoGIL no_gil;
  return self.prelu(weight);
}
inline Tensor dispatch_prod(Tensor & self, c10::optional<ScalarType> dtype) {

  AutoNoGIL no_gil;
  return self.prod(dtype);
}
inline Tensor dispatch_prod(Tensor & self, Dimname dim, bool keepdim, c10::optional<ScalarType> dtype) {

  AutoNoGIL no_gil;
  return self.prod(dim, keepdim, dtype);
}
inline Tensor dispatch_prod(Tensor & self, int64_t dim, bool keepdim, c10::optional<ScalarType> dtype) {

  AutoNoGIL no_gil;
  return self.prod(dim, keepdim, dtype);
}
inline Tensor dispatch_put_(Tensor & self, const Tensor & index, const Tensor & source, bool accumulate) {

  AutoNoGIL no_gil;
  return self.put_(index, source, accumulate);
}
inline int64_t dispatch_q_per_channel_axis(Tensor & self) {

  AutoNoGIL no_gil;
  return self.q_per_channel_axis();
}
inline Tensor dispatch_q_per_channel_scales(Tensor & self) {

  AutoNoGIL no_gil;
  return self.q_per_channel_scales();
}
inline Tensor dispatch_q_per_channel_zero_points(Tensor & self) {

  AutoNoGIL no_gil;
  return self.q_per_channel_zero_points();
}
inline double dispatch_q_scale(Tensor & self) {

  AutoNoGIL no_gil;
  return self.q_scale();
}
inline int64_t dispatch_q_zero_point(Tensor & self) {

  AutoNoGIL no_gil;
  return self.q_zero_point();
}
inline std::tuple<Tensor,Tensor> dispatch_qr(Tensor & self, bool some) {

  AutoNoGIL no_gil;
  return self.qr(some);
}
inline QScheme dispatch_qscheme(Tensor & self) {

  AutoNoGIL no_gil;
  return self.qscheme();
}
inline Tensor dispatch_random_(Tensor & self, Generator * generator) {

  AutoNoGIL no_gil;
  return self.random_(generator);
}
inline Tensor dispatch_random_(Tensor & self, int64_t from, int64_t to, Generator * generator) {

  AutoNoGIL no_gil;
  return self.random_(from, to, generator);
}
inline Tensor dispatch_random_(Tensor & self, int64_t to, Generator * generator) {

  AutoNoGIL no_gil;
  return self.random_(to, generator);
}
inline Tensor dispatch_reciprocal(Tensor & self) {

  AutoNoGIL no_gil;
  return self.reciprocal();
}
inline Tensor dispatch_reciprocal_(Tensor & self) {

  AutoNoGIL no_gil;
  return self.reciprocal_();
}
inline Tensor dispatch_refine_names(Tensor & self, DimnameList names) {

  AutoNoGIL no_gil;
  return self.refine_names(names);
}
inline Tensor dispatch_relu(Tensor & self) {

  AutoNoGIL no_gil;
  return self.relu();
}
inline Tensor dispatch_relu_(Tensor & self) {

  AutoNoGIL no_gil;
  return self.relu_();
}
inline Tensor dispatch_remainder(Tensor & self, const Tensor & other) {

  AutoNoGIL no_gil;
  return self.remainder(other);
}
inline Tensor dispatch_remainder(Tensor & self, Scalar other) {

  AutoNoGIL no_gil;
  return self.remainder(other);
}
inline Tensor dispatch_remainder_(Tensor & self, const Tensor & other) {

  AutoNoGIL no_gil;
  return self.remainder_(other);
}
inline Tensor dispatch_remainder_(Tensor & self, Scalar other) {

  AutoNoGIL no_gil;
  return self.remainder_(other);
}
inline Tensor dispatch_rename(Tensor & self, c10::optional<DimnameList> names) {

  AutoNoGIL no_gil;
  return self.rename(names);
}
inline Tensor dispatch_rename_(Tensor & self, c10::optional<DimnameList> names) {

  AutoNoGIL no_gil;
  return self.rename_(names);
}
inline Tensor dispatch_renorm(Tensor & self, Scalar p, int64_t dim, Scalar maxnorm) {

  AutoNoGIL no_gil;
  return self.renorm(p, dim, maxnorm);
}
inline Tensor dispatch_renorm_(Tensor & self, Scalar p, int64_t dim, Scalar maxnorm) {

  AutoNoGIL no_gil;
  return self.renorm_(p, dim, maxnorm);
}
inline Tensor dispatch_repeat(Tensor & self, IntArrayRef repeats) {

  AutoNoGIL no_gil;
  return self.repeat(repeats);
}
inline Tensor dispatch_repeat_interleave(Tensor & self, const Tensor & repeats, c10::optional<int64_t> dim) {

  AutoNoGIL no_gil;
  return self.repeat_interleave(repeats, dim);
}
inline Tensor dispatch_repeat_interleave(Tensor & self, int64_t repeats, c10::optional<int64_t> dim) {

  AutoNoGIL no_gil;
  return self.repeat_interleave(repeats, dim);
}
inline Tensor dispatch_reshape(Tensor & self, IntArrayRef shape) {

  AutoNoGIL no_gil;
  return self.reshape(shape);
}
inline Tensor dispatch_reshape_as(Tensor & self, const Tensor & other) {

  AutoNoGIL no_gil;
  return self.reshape_as(other);
}
inline Tensor dispatch_resize_(Tensor & self, IntArrayRef size) {

  AutoNoGIL no_gil;
  return self.resize_(size);
}
inline Tensor dispatch_resize_as_(Tensor & self, const Tensor & the_template) {

  AutoNoGIL no_gil;
  return self.resize_as_(the_template);
}
inline Tensor dispatch_rfft(Tensor & self, int64_t signal_ndim, bool normalized, bool onesided) {

  AutoNoGIL no_gil;
  return self.rfft(signal_ndim, normalized, onesided);
}
inline Tensor dispatch_roll(Tensor & self, IntArrayRef shifts, IntArrayRef dims) {

  AutoNoGIL no_gil;
  return self.roll(shifts, dims);
}
inline Tensor dispatch_rot90(Tensor & self, int64_t k, IntArrayRef dims) {

  AutoNoGIL no_gil;
  return self.rot90(k, dims);
}
inline Tensor dispatch_round(Tensor & self) {

  AutoNoGIL no_gil;
  return self.round();
}
inline Tensor dispatch_round_(Tensor & self) {

  AutoNoGIL no_gil;
  return self.round_();
}
inline Tensor dispatch_rsqrt(Tensor & self) {

  AutoNoGIL no_gil;
  return self.rsqrt();
}
inline Tensor dispatch_rsqrt_(Tensor & self) {

  AutoNoGIL no_gil;
  return self.rsqrt_();
}
inline Tensor dispatch_scatter(Tensor & self, Dimname dim, const Tensor & index, const Tensor & src) {

  AutoNoGIL no_gil;
  return self.scatter(dim, index, src);
}
inline Tensor dispatch_scatter(Tensor & self, int64_t dim, const Tensor & index, const Tensor & src) {

  AutoNoGIL no_gil;
  return self.scatter(dim, index, src);
}
inline Tensor dispatch_scatter(Tensor & self, Dimname dim, const Tensor & index, Scalar value) {

  AutoNoGIL no_gil;
  return self.scatter(dim, index, value);
}
inline Tensor dispatch_scatter(Tensor & self, int64_t dim, const Tensor & index, Scalar value) {

  AutoNoGIL no_gil;
  return self.scatter(dim, index, value);
}
inline Tensor dispatch_scatter_(Tensor & self, int64_t dim, const Tensor & index, const Tensor & src) {

  AutoNoGIL no_gil;
  return self.scatter_(dim, index, src);
}
inline Tensor dispatch_scatter_(Tensor & self, int64_t dim, const Tensor & index, Scalar value) {

  AutoNoGIL no_gil;
  return self.scatter_(dim, index, value);
}
inline Tensor dispatch_scatter_add(Tensor & self, Dimname dim, const Tensor & index, const Tensor & src) {

  AutoNoGIL no_gil;
  return self.scatter_add(dim, index, src);
}
inline Tensor dispatch_scatter_add(Tensor & self, int64_t dim, const Tensor & index, const Tensor & src) {

  AutoNoGIL no_gil;
  return self.scatter_add(dim, index, src);
}
inline Tensor dispatch_scatter_add_(Tensor & self, int64_t dim, const Tensor & index, const Tensor & src) {

  AutoNoGIL no_gil;
  return self.scatter_add_(dim, index, src);
}
inline Tensor dispatch_select(Tensor & self, Dimname dim, int64_t index) {

  AutoNoGIL no_gil;
  return self.select(dim, index);
}
inline Tensor dispatch_select(Tensor & self, int64_t dim, int64_t index) {

  AutoNoGIL no_gil;
  return self.select(dim, index);
}
inline Tensor dispatch_set_(Tensor & self) {

  AutoNoGIL no_gil;
  return self.set_();
}
inline Tensor dispatch_set_(Tensor & self, Storage source) {

  AutoNoGIL no_gil;
  return self.set_(source);
}
inline Tensor dispatch_set_(Tensor & self, Storage source, int64_t storage_offset, IntArrayRef size, IntArrayRef stride) {

  AutoNoGIL no_gil;
  return self.set_(source, storage_offset, size, stride);
}
inline Tensor dispatch_set_(Tensor & self, const Tensor & source) {

  AutoNoGIL no_gil;
  return self.set_(source);
}
inline Tensor dispatch_sigmoid(Tensor & self) {

  AutoNoGIL no_gil;
  return self.sigmoid();
}
inline Tensor dispatch_sigmoid_(Tensor & self) {

  AutoNoGIL no_gil;
  return self.sigmoid_();
}
inline Tensor dispatch_sign(Tensor & self) {

  AutoNoGIL no_gil;
  return self.sign();
}
inline Tensor dispatch_sign_(Tensor & self) {

  AutoNoGIL no_gil;
  return self.sign_();
}
inline Tensor dispatch_sin(Tensor & self) {

  AutoNoGIL no_gil;
  return self.sin();
}
inline Tensor dispatch_sin_(Tensor & self) {

  AutoNoGIL no_gil;
  return self.sin_();
}
inline Tensor dispatch_sinh(Tensor & self) {

  AutoNoGIL no_gil;
  return self.sinh();
}
inline Tensor dispatch_sinh_(Tensor & self) {

  AutoNoGIL no_gil;
  return self.sinh_();
}
inline std::tuple<Tensor,Tensor> dispatch_slogdet(Tensor & self) {

  AutoNoGIL no_gil;
  return self.slogdet();
}
inline Tensor dispatch_smm(Tensor & self, const Tensor & mat2) {

  AutoNoGIL no_gil;
  return self.smm(mat2);
}
inline Tensor dispatch_softmax(Tensor & self, Dimname dim, c10::optional<ScalarType> dtype) {

  AutoNoGIL no_gil;
  return self.softmax(dim, dtype);
}
inline Tensor dispatch_softmax(Tensor & self, int64_t dim, c10::optional<ScalarType> dtype) {

  AutoNoGIL no_gil;
  return self.softmax(dim, dtype);
}
inline std::tuple<Tensor,Tensor> dispatch_solve(Tensor & self, const Tensor & A) {

  AutoNoGIL no_gil;
  return self.solve(A);
}
inline std::tuple<Tensor,Tensor> dispatch_sort(Tensor & self, Dimname dim, bool descending) {

  AutoNoGIL no_gil;
  return self.sort(dim, descending);
}
inline std::tuple<Tensor,Tensor> dispatch_sort(Tensor & self, int64_t dim, bool descending) {

  AutoNoGIL no_gil;
  return self.sort(dim, descending);
}
inline int64_t dispatch_sparse_dim(Tensor & self) {

  AutoNoGIL no_gil;
  return self.sparse_dim();
}
inline Tensor dispatch_sparse_mask(Tensor & self, const Tensor & mask) {

  AutoNoGIL no_gil;
  return self.sparse_mask(mask);
}
inline Tensor dispatch_sparse_resize_(Tensor & self, IntArrayRef size, int64_t sparse_dim, int64_t dense_dim) {

  AutoNoGIL no_gil;
  return self.sparse_resize_(size, sparse_dim, dense_dim);
}
inline Tensor dispatch_sparse_resize_and_clear_(Tensor & self, IntArrayRef size, int64_t sparse_dim, int64_t dense_dim) {

  AutoNoGIL no_gil;
  return self.sparse_resize_and_clear_(size, sparse_dim, dense_dim);
}
inline std::vector<Tensor> dispatch_split(Tensor & self, int64_t split_size, int64_t dim) {

  AutoNoGIL no_gil;
  return self.split(split_size, dim);
}
inline std::vector<Tensor> dispatch_split_with_sizes(Tensor & self, IntArrayRef split_sizes, int64_t dim) {

  AutoNoGIL no_gil;
  return self.split_with_sizes(split_sizes, dim);
}
inline Tensor dispatch_sqrt(Tensor & self) {

  AutoNoGIL no_gil;
  return self.sqrt();
}
inline Tensor dispatch_sqrt_(Tensor & self) {

  AutoNoGIL no_gil;
  return self.sqrt_();
}
inline Tensor dispatch_squeeze(Tensor & self) {

  AutoNoGIL no_gil;
  return self.squeeze();
}
inline Tensor dispatch_squeeze(Tensor & self, Dimname dim) {

  AutoNoGIL no_gil;
  return self.squeeze(dim);
}
inline Tensor dispatch_squeeze(Tensor & self, int64_t dim) {

  AutoNoGIL no_gil;
  return self.squeeze(dim);
}
inline Tensor dispatch_squeeze_(Tensor & self) {

  AutoNoGIL no_gil;
  return self.squeeze_();
}
inline Tensor dispatch_squeeze_(Tensor & self, Dimname dim) {

  AutoNoGIL no_gil;
  return self.squeeze_(dim);
}
inline Tensor dispatch_squeeze_(Tensor & self, int64_t dim) {

  AutoNoGIL no_gil;
  return self.squeeze_(dim);
}
inline Tensor dispatch_sspaddmm(Scalar beta, Tensor & self, Scalar alpha, const Tensor & mat1, const Tensor & mat2) {

  AutoNoGIL no_gil;
  return self.sspaddmm(mat1, mat2, beta, alpha);
}
inline Tensor dispatch_sspaddmm(Scalar beta, Tensor & self, const Tensor & mat1, const Tensor & mat2) {

  AutoNoGIL no_gil;
  return self.sspaddmm(mat1, mat2, beta, 1);
}
inline Tensor dispatch_sspaddmm(Tensor & self, const Tensor & mat1, const Tensor & mat2, Scalar beta, Scalar alpha) {

  AutoNoGIL no_gil;
  return self.sspaddmm(mat1, mat2, beta, alpha);
}
inline Tensor dispatch_std(Tensor & self, DimnameList dim, bool unbiased, bool keepdim) {

  AutoNoGIL no_gil;
  return self.std(dim, unbiased, keepdim);
}
inline Tensor dispatch_std(Tensor & self, IntArrayRef dim, bool unbiased, bool keepdim) {

  AutoNoGIL no_gil;
  return self.std(dim, unbiased, keepdim);
}
inline Tensor dispatch_std(Tensor & self, bool unbiased) {

  AutoNoGIL no_gil;
  return self.std(unbiased);
}
inline Tensor dispatch_stft(Tensor & self, int64_t n_fft, c10::optional<int64_t> hop_length, c10::optional<int64_t> win_length, const Tensor & window, bool normalized, bool onesided) {

  AutoNoGIL no_gil;
  return self.stft(n_fft, hop_length, win_length, window, normalized, onesided);
}
inline Tensor dispatch_sub(Tensor & self, Scalar alpha, const Tensor & other) {

  AutoNoGIL no_gil;
  return self.sub(other, alpha);
}
inline Tensor dispatch_sub(Tensor & self, const Tensor & other, Scalar alpha) {

  AutoNoGIL no_gil;
  return self.sub(other, alpha);
}
inline Tensor dispatch_sub_(Tensor & self, Scalar alpha, const Tensor & other) {

  AutoNoGIL no_gil;
  return self.sub_(other, alpha);
}
inline Tensor dispatch_sub_(Tensor & self, const Tensor & other, Scalar alpha) {

  AutoNoGIL no_gil;
  return self.sub_(other, alpha);
}
inline Tensor dispatch_sum(Tensor & self, c10::optional<ScalarType> dtype) {

  AutoNoGIL no_gil;
  return self.sum(dtype);
}
inline Tensor dispatch_sum(Tensor & self, DimnameList dim, bool keepdim, c10::optional<ScalarType> dtype) {

  AutoNoGIL no_gil;
  return self.sum(dim, keepdim, dtype);
}
inline Tensor dispatch_sum(Tensor & self, IntArrayRef dim, bool keepdim, c10::optional<ScalarType> dtype) {

  AutoNoGIL no_gil;
  return self.sum(dim, keepdim, dtype);
}
inline Tensor dispatch_sum_to_size(Tensor & self, IntArrayRef size) {

  AutoNoGIL no_gil;
  return self.sum_to_size(size);
}
inline std::tuple<Tensor,Tensor,Tensor> dispatch_svd(Tensor & self, bool some, bool compute_uv) {

  AutoNoGIL no_gil;
  return self.svd(some, compute_uv);
}
inline std::tuple<Tensor,Tensor> dispatch_symeig(Tensor & self, bool eigenvectors, bool upper) {

  AutoNoGIL no_gil;
  return self.symeig(eigenvectors, upper);
}
inline Tensor dispatch_t(Tensor & self) {

  AutoNoGIL no_gil;
  return self.t();
}
inline Tensor dispatch_t_(Tensor & self) {

  AutoNoGIL no_gil;
  return self.t_();
}
inline Tensor dispatch_take(Tensor & self, const Tensor & index) {

  AutoNoGIL no_gil;
  return self.take(index);
}
inline Tensor dispatch_tan(Tensor & self) {

  AutoNoGIL no_gil;
  return self.tan();
}
inline Tensor dispatch_tan_(Tensor & self) {

  AutoNoGIL no_gil;
  return self.tan_();
}
inline Tensor dispatch_tanh(Tensor & self) {

  AutoNoGIL no_gil;
  return self.tanh();
}
inline Tensor dispatch_tanh_(Tensor & self) {

  AutoNoGIL no_gil;
  return self.tanh_();
}
inline Tensor dispatch_to_dense(Tensor & self) {

  AutoNoGIL no_gil;
  return self.to_dense();
}
inline Tensor dispatch_to_mkldnn(Tensor & self) {

  AutoNoGIL no_gil;
  return self.to_mkldnn();
}
inline Tensor dispatch_to_sparse(Tensor & self) {

  AutoNoGIL no_gil;
  return self.to_sparse();
}
inline Tensor dispatch_to_sparse(Tensor & self, int64_t sparse_dim) {

  AutoNoGIL no_gil;
  return self.to_sparse(sparse_dim);
}
inline std::tuple<Tensor,Tensor> dispatch_topk(Tensor & self, int64_t k, int64_t dim, bool largest, bool sorted) {

  AutoNoGIL no_gil;
  return self.topk(k, dim, largest, sorted);
}
inline Tensor dispatch_trace(Tensor & self) {

  AutoNoGIL no_gil;
  return self.trace();
}
inline Tensor dispatch_transpose(Tensor & self, Dimname dim0, Dimname dim1) {

  AutoNoGIL no_gil;
  return self.transpose(dim0, dim1);
}
inline Tensor dispatch_transpose(Tensor & self, int64_t dim0, int64_t dim1) {

  AutoNoGIL no_gil;
  return self.transpose(dim0, dim1);
}
inline Tensor dispatch_transpose_(Tensor & self, int64_t dim0, int64_t dim1) {

  AutoNoGIL no_gil;
  return self.transpose_(dim0, dim1);
}
inline std::tuple<Tensor,Tensor> dispatch_triangular_solve(Tensor & self, const Tensor & A, bool upper, bool transpose, bool unitriangular) {

  AutoNoGIL no_gil;
  return self.triangular_solve(A, upper, transpose, unitriangular);
}
inline Tensor dispatch_tril(Tensor & self, int64_t diagonal) {

  AutoNoGIL no_gil;
  return self.tril(diagonal);
}
inline Tensor dispatch_tril_(Tensor & self, int64_t diagonal) {

  AutoNoGIL no_gil;
  return self.tril_(diagonal);
}
inline Tensor dispatch_triu(Tensor & self, int64_t diagonal) {

  AutoNoGIL no_gil;
  return self.triu(diagonal);
}
inline Tensor dispatch_triu_(Tensor & self, int64_t diagonal) {

  AutoNoGIL no_gil;
  return self.triu_(diagonal);
}
inline Tensor dispatch_trunc(Tensor & self) {

  AutoNoGIL no_gil;
  return self.trunc();
}
inline Tensor dispatch_trunc_(Tensor & self) {

  AutoNoGIL no_gil;
  return self.trunc_();
}
inline Tensor dispatch_type_as(Tensor & self, const Tensor & other) {

  AutoNoGIL no_gil;
  return self.type_as(other);
}
inline std::vector<Tensor> dispatch_unbind(Tensor & self, Dimname dim) {

  AutoNoGIL no_gil;
  return self.unbind(dim);
}
inline std::vector<Tensor> dispatch_unbind(Tensor & self, int64_t dim) {

  AutoNoGIL no_gil;
  return self.unbind(dim);
}
inline Tensor dispatch_unflatten(Tensor & self, Dimname dim, IntArrayRef sizes, DimnameList names) {

  AutoNoGIL no_gil;
  return self.unflatten(dim, sizes, names);
}
inline Tensor dispatch_unflatten(Tensor & self, int64_t dim, IntArrayRef sizes, DimnameList names) {

  AutoNoGIL no_gil;
  return self.unflatten(dim, sizes, names);
}
inline Tensor dispatch_unfold(Tensor & self, int64_t dimension, int64_t size, int64_t step) {

  AutoNoGIL no_gil;
  return self.unfold(dimension, size, step);
}
inline Tensor dispatch_uniform_(Tensor & self, double from, double to, Generator * generator) {

  AutoNoGIL no_gil;
  return self.uniform_(from, to, generator);
}
inline Tensor dispatch_unsqueeze(Tensor & self, int64_t dim) {

  AutoNoGIL no_gil;
  return self.unsqueeze(dim);
}
inline Tensor dispatch_unsqueeze_(Tensor & self, int64_t dim) {

  AutoNoGIL no_gil;
  return self.unsqueeze_(dim);
}
inline Tensor dispatch_values(Tensor & self) {

  AutoNoGIL no_gil;
  return self.values();
}
inline Tensor dispatch_var(Tensor & self, DimnameList dim, bool unbiased, bool keepdim) {

  AutoNoGIL no_gil;
  return self.var(dim, unbiased, keepdim);
}
inline Tensor dispatch_var(Tensor & self, IntArrayRef dim, bool unbiased, bool keepdim) {

  AutoNoGIL no_gil;
  return self.var(dim, unbiased, keepdim);
}
inline Tensor dispatch_var(Tensor & self, bool unbiased) {

  AutoNoGIL no_gil;
  return self.var(unbiased);
}
inline Tensor dispatch_view(Tensor & self, IntArrayRef size) {

  AutoNoGIL no_gil;
  return self.view(size);
}
inline Tensor dispatch_view_as(Tensor & self, const Tensor & other) {

  AutoNoGIL no_gil;
  return self.view_as(other);
}
inline Tensor dispatch_where(const Tensor & condition, Tensor & self, const Tensor & other) {

  AutoNoGIL no_gil;
  return self.where(condition, other);
}
inline Tensor dispatch_zero_(Tensor & self) {

  AutoNoGIL no_gil;
  return self.zero_();
}

}} // namespace torch::autograd
