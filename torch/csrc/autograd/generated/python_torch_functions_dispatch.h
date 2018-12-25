#pragma once

// @generated from tools/autograd/templates/python_torch_functions_dispatch.h

#include "torch/csrc/autograd/generated/VariableType.h"
#include "torch/csrc/autograd/generated/variable_factories.h"
#include "torch/csrc/tensor/python_tensor.h"
#include "torch/csrc/utils/auto_gil.h"
#include "torch/csrc/utils/cuda_lazy_init.h"

#include <ATen/ATen.h>

// Contains inline wrappers around ATen functions that release the GIL and
// switch to the correct CUDA device.

namespace torch { namespace autograd {

using at::Tensor;
using at::Scalar;
using at::TensorList;
using at::IntList;
using at::Generator;
using at::SparseTensorRef;
using at::Storage;
using at::TensorOptions;

static at::Type& default_type() {
  return torch::tensors::get_default_tensor_type();
}

static void maybe_initialize_cuda(const at::TensorOptions& options) {
  if (options.device().is_cuda()) {
    torch::utils::cuda_lazy_init();
  }
}

inline Tensor dispatch___and__(const Tensor & self, const Tensor & other) {

  AutoNoGIL no_gil;
  return self.__and__(other);
}
inline Tensor dispatch___and__(const Tensor & self, Scalar other) {

  AutoNoGIL no_gil;
  return self.__and__(other);
}
inline Tensor dispatch___lshift__(const Tensor & self, const Tensor & other) {

  AutoNoGIL no_gil;
  return self.__lshift__(other);
}
inline Tensor dispatch___lshift__(const Tensor & self, Scalar other) {

  AutoNoGIL no_gil;
  return self.__lshift__(other);
}
inline Tensor dispatch___or__(const Tensor & self, const Tensor & other) {

  AutoNoGIL no_gil;
  return self.__or__(other);
}
inline Tensor dispatch___or__(const Tensor & self, Scalar other) {

  AutoNoGIL no_gil;
  return self.__or__(other);
}
inline Tensor dispatch___rshift__(const Tensor & self, const Tensor & other) {

  AutoNoGIL no_gil;
  return self.__rshift__(other);
}
inline Tensor dispatch___rshift__(const Tensor & self, Scalar other) {

  AutoNoGIL no_gil;
  return self.__rshift__(other);
}
inline Tensor dispatch___xor__(const Tensor & self, const Tensor & other) {

  AutoNoGIL no_gil;
  return self.__xor__(other);
}
inline Tensor dispatch___xor__(const Tensor & self, Scalar other) {

  AutoNoGIL no_gil;
  return self.__xor__(other);
}
inline Tensor dispatch__argmax(const Tensor & self, int64_t dim, bool keepdim) {

  AutoNoGIL no_gil;
  return at::_argmax(self, dim, keepdim);
}
inline Tensor dispatch__argmin(const Tensor & self, int64_t dim, bool keepdim) {

  AutoNoGIL no_gil;
  return at::_argmin(self, dim, keepdim);
}
inline Tensor dispatch__baddbmm_mkl_(Tensor self, const Tensor & batch1, const Tensor & batch2, Scalar beta, Scalar alpha) {

  AutoNoGIL no_gil;
  return at::_baddbmm_mkl_(self, batch1, batch2, beta, alpha);
}
inline Tensor dispatch__cast_Byte(const Tensor & self, bool non_blocking) {

  AutoNoGIL no_gil;
  return at::_cast_Byte(self, non_blocking);
}
inline Tensor dispatch__cast_Char(const Tensor & self, bool non_blocking) {

  AutoNoGIL no_gil;
  return at::_cast_Char(self, non_blocking);
}
inline Tensor dispatch__cast_Double(const Tensor & self, bool non_blocking) {

  AutoNoGIL no_gil;
  return at::_cast_Double(self, non_blocking);
}
inline Tensor dispatch__cast_Float(const Tensor & self, bool non_blocking) {

  AutoNoGIL no_gil;
  return at::_cast_Float(self, non_blocking);
}
inline Tensor dispatch__cast_Half(const Tensor & self, bool non_blocking) {

  AutoNoGIL no_gil;
  return at::_cast_Half(self, non_blocking);
}
inline Tensor dispatch__cast_Int(const Tensor & self, bool non_blocking) {

  AutoNoGIL no_gil;
  return at::_cast_Int(self, non_blocking);
}
inline Tensor dispatch__cast_Long(const Tensor & self, bool non_blocking) {

  AutoNoGIL no_gil;
  return at::_cast_Long(self, non_blocking);
}
inline Tensor dispatch__cast_Short(const Tensor & self, bool non_blocking) {

  AutoNoGIL no_gil;
  return at::_cast_Short(self, non_blocking);
}
inline Tensor dispatch__convolution(const Tensor & input, const Tensor & weight, const Tensor & bias, IntList stride, IntList padding, IntList dilation, bool transposed, IntList output_padding, int64_t groups, bool benchmark, bool deterministic, bool cudnn_enabled) {

  AutoNoGIL no_gil;
  return at::_convolution(input, weight, bias, stride, padding, dilation, transposed, output_padding, groups, benchmark, deterministic, cudnn_enabled);
}
inline Tensor dispatch__convolution_nogroup(const Tensor & input, const Tensor & weight, const Tensor & bias, IntList stride, IntList padding, IntList dilation, bool transposed, IntList output_padding) {

  AutoNoGIL no_gil;
  return at::_convolution_nogroup(input, weight, bias, stride, padding, dilation, transposed, output_padding);
}
inline void dispatch__copy_same_type_(Tensor self, const Tensor & src) {

  AutoNoGIL no_gil;
  return at::_copy_same_type_(self, src);
}
inline std::tuple<Tensor,Tensor> dispatch__ctc_loss(const Tensor & log_probs, const Tensor & targets, IntList input_lengths, IntList target_lengths, int64_t blank) {

  AutoNoGIL no_gil;
  return at::_ctc_loss(log_probs, targets, input_lengths, target_lengths, blank);
}
inline std::tuple<Tensor,Tensor> dispatch__cudnn_ctc_loss(const Tensor & log_probs, const Tensor & targets, IntList input_lengths, IntList target_lengths, int64_t blank, bool deterministic) {

  AutoNoGIL no_gil;
  return at::_cudnn_ctc_loss(log_probs, targets, input_lengths, target_lengths, blank, deterministic);
}
inline Tensor dispatch__cudnn_init_dropout_state(double dropout, bool train, int64_t dropout_seed, const TensorOptions & options) {
  maybe_initialize_cuda(options);
  AutoNoGIL no_gil;
  return torch::_cudnn_init_dropout_state(dropout, train, dropout_seed, options);
}
inline std::tuple<Tensor,Tensor,Tensor,Tensor,Tensor> dispatch__cudnn_rnn(const Tensor & input, TensorList weight, int64_t weight_stride0, const Tensor & weight_buf, const Tensor & hx, const Tensor & cx, int64_t mode, int64_t hidden_size, int64_t num_layers, bool batch_first, double dropout, bool train, bool bidirectional, IntList batch_sizes, const Tensor & dropout_state) {

  AutoNoGIL no_gil;
  return at::_cudnn_rnn(input, weight, weight_stride0, weight_buf, hx, cx, mode, hidden_size, num_layers, batch_first, dropout, train, bidirectional, batch_sizes, dropout_state);
}
inline Tensor dispatch__cudnn_rnn_flatten_weight(TensorList weight_arr, int64_t weight_stride0, int64_t input_size, int64_t mode, int64_t hidden_size, int64_t num_layers, bool batch_first, bool bidirectional) {

  AutoNoGIL no_gil;
  return at::_cudnn_rnn_flatten_weight(weight_arr, weight_stride0, input_size, mode, hidden_size, num_layers, batch_first, bidirectional);
}
inline void dispatch__cufft_clear_plan_cache() {

  AutoNoGIL no_gil;
  return at::_cufft_clear_plan_cache();
}
inline int64_t dispatch__cufft_get_plan_cache_max_size() {

  AutoNoGIL no_gil;
  return at::_cufft_get_plan_cache_max_size();
}
inline int64_t dispatch__cufft_get_plan_cache_size() {

  AutoNoGIL no_gil;
  return at::_cufft_get_plan_cache_size();
}
inline void dispatch__cufft_set_plan_cache_max_size(int64_t max_size) {

  AutoNoGIL no_gil;
  return at::_cufft_set_plan_cache_max_size(max_size);
}
inline Tensor dispatch__dim_arange(const Tensor & like, int64_t dim) {

  AutoNoGIL no_gil;
  return at::_dim_arange(like, dim);
}
inline Tensor dispatch__dirichlet_grad(const Tensor & x, const Tensor & alpha, const Tensor & total, Tensor output) {

  AutoNoGIL no_gil;
  return at::_dirichlet_grad_out(output, x, alpha, total);
}
inline Tensor dispatch__dirichlet_grad(const Tensor & x, const Tensor & alpha, const Tensor & total) {

  AutoNoGIL no_gil;
  return at::_dirichlet_grad(x, alpha, total);
}
inline std::tuple<Tensor,Tensor,Tensor,Tensor> dispatch__embedding_bag(const Tensor & weight, const Tensor & indices, const Tensor & offsets, bool scale_grad_by_freq, int64_t mode, bool sparse) {

  AutoNoGIL no_gil;
  return at::_embedding_bag(weight, indices, offsets, scale_grad_by_freq, mode, sparse);
}
inline Tensor dispatch__fft_with_size(const Tensor & self, int64_t signal_ndim, bool complex_input, bool complex_output, bool inverse, IntList checked_signal_sizes, bool normalized, bool onesided, IntList output_sizes) {

  AutoNoGIL no_gil;
  return at::_fft_with_size(self, signal_ndim, complex_input, complex_output, inverse, checked_signal_sizes, normalized, onesided, output_sizes);
}
inline std::tuple<Tensor,Tensor> dispatch__fused_dropout(const Tensor & self, double p, Generator * generator) {

  AutoNoGIL no_gil;
  return at::_fused_dropout(self, p, generator);
}
inline Tensor dispatch__log_softmax(const Tensor & self, int64_t dim, bool half_to_float) {

  AutoNoGIL no_gil;
  return at::_log_softmax(self, dim, half_to_float);
}
inline Tensor dispatch__log_softmax_backward_data(const Tensor & grad_output, const Tensor & output, int64_t dim, const Tensor & self) {

  AutoNoGIL no_gil;
  return at::_log_softmax_backward_data(grad_output, output, dim, self);
}
inline Tensor dispatch__masked_scale(const Tensor & self, const Tensor & mask, double scale) {

  AutoNoGIL no_gil;
  return at::_masked_scale(self, mask, scale);
}
inline std::tuple<Tensor,Tensor> dispatch__pack_padded_sequence(const Tensor & input, const Tensor & lengths, bool batch_first) {

  AutoNoGIL no_gil;
  return at::_pack_padded_sequence(input, lengths, batch_first);
}
inline std::tuple<Tensor,Tensor> dispatch__pad_packed_sequence(const Tensor & data, const Tensor & batch_sizes, bool batch_first, Scalar padding_value, int64_t total_length) {

  AutoNoGIL no_gil;
  return at::_pad_packed_sequence(data, batch_sizes, batch_first, padding_value, total_length);
}
inline Tensor dispatch__reshape_from_tensor(const Tensor & self, const Tensor & shape) {

  AutoNoGIL no_gil;
  return at::_reshape_from_tensor(self, shape);
}
inline Tensor dispatch__s_copy_from(const Tensor & self, const Tensor & dst, bool non_blocking) {

  AutoNoGIL no_gil;
  return at::_s_copy_from(self, dst, non_blocking);
}
inline Tensor dispatch__s_where(const Tensor & condition, const Tensor & self, const Tensor & other) {

  AutoNoGIL no_gil;
  return at::_s_where(condition, self, other);
}
inline Tensor dispatch__shape_as_tensor(const Tensor & self) {

  AutoNoGIL no_gil;
  return at::_shape_as_tensor(self);
}
inline Tensor dispatch__softmax(const Tensor & self, int64_t dim, bool half_to_float) {

  AutoNoGIL no_gil;
  return at::_softmax(self, dim, half_to_float);
}
inline Tensor dispatch__softmax_backward_data(const Tensor & grad_output, const Tensor & output, int64_t dim, const Tensor & self) {

  AutoNoGIL no_gil;
  return at::_softmax_backward_data(grad_output, output, dim, self);
}
inline Tensor dispatch__sparse_addmm(const Tensor & self, const Tensor & sparse, const Tensor & dense, Scalar beta, Scalar alpha) {

  AutoNoGIL no_gil;
  return at::_sparse_addmm(self, sparse, dense, beta, alpha);
}
inline Tensor dispatch__sparse_mm(const Tensor & sparse, const Tensor & dense) {

  AutoNoGIL no_gil;
  return at::_sparse_mm(sparse, dense);
}
inline Tensor dispatch__sparse_sum(const Tensor & self) {

  AutoNoGIL no_gil;
  return at::_sparse_sum(self);
}
inline Tensor dispatch__sparse_sum(const Tensor & self, ScalarType dtype) {

  AutoNoGIL no_gil;
  return at::_sparse_sum(self, dtype);
}
inline Tensor dispatch__sparse_sum(const Tensor & self, IntList dim) {

  AutoNoGIL no_gil;
  return at::_sparse_sum(self, dim);
}
inline Tensor dispatch__sparse_sum(const Tensor & self, IntList dim, ScalarType dtype) {

  AutoNoGIL no_gil;
  return at::_sparse_sum(self, dim, dtype);
}
inline Tensor dispatch__standard_gamma(const Tensor & self, Generator * generator) {

  AutoNoGIL no_gil;
  return at::_standard_gamma(self, generator);
}
inline Tensor dispatch__standard_gamma_grad(const Tensor & self, const Tensor & output) {

  AutoNoGIL no_gil;
  return at::_standard_gamma_grad(self, output);
}
inline Tensor dispatch__trilinear(const Tensor & i1, const Tensor & i2, const Tensor & i3, IntList expand1, IntList expand2, IntList expand3, IntList sumdim, int64_t unroll_dim) {

  AutoNoGIL no_gil;
  return at::_trilinear(i1, i2, i3, expand1, expand2, expand3, sumdim, unroll_dim);
}
inline std::tuple<Tensor,Tensor> dispatch__unique(const Tensor & self, bool sorted, bool return_inverse) {

  AutoNoGIL no_gil;
  return at::_unique(self, sorted, return_inverse);
}
inline std::tuple<Tensor,Tensor> dispatch__unique_dim(const Tensor & self, int64_t dim, bool sorted, bool return_inverse) {

  AutoNoGIL no_gil;
  return at::_unique_dim(self, dim, sorted, return_inverse);
}
inline Tensor dispatch__weight_norm(const Tensor & v, const Tensor & g, int64_t dim) {

  AutoNoGIL no_gil;
  return at::_weight_norm(v, g, dim);
}
inline std::tuple<Tensor,Tensor> dispatch__weight_norm_cuda_interface(const Tensor & v, const Tensor & g, int64_t dim) {

  AutoNoGIL no_gil;
  return at::_weight_norm_cuda_interface(v, g, dim);
}
inline Tensor dispatch_abs(const Tensor & self, Tensor result) {

  AutoNoGIL no_gil;
  return at::abs_out(result, self);
}
inline Tensor dispatch_abs(const Tensor & self) {

  AutoNoGIL no_gil;
  return self.abs();
}
inline Tensor dispatch_abs_(Tensor self) {

  AutoNoGIL no_gil;
  return self.abs_();
}
inline Tensor dispatch_acos(const Tensor & self, Tensor result) {

  AutoNoGIL no_gil;
  return at::acos_out(result, self);
}
inline Tensor dispatch_acos(const Tensor & self) {

  AutoNoGIL no_gil;
  return self.acos();
}
inline Tensor dispatch_acos_(Tensor self) {

  AutoNoGIL no_gil;
  return self.acos_();
}
inline Tensor dispatch_adaptive_avg_pool1d(const Tensor & self, IntList output_size) {

  AutoNoGIL no_gil;
  return at::adaptive_avg_pool1d(self, output_size);
}
inline std::tuple<Tensor,Tensor> dispatch_adaptive_max_pool1d(const Tensor & self, IntList output_size) {

  AutoNoGIL no_gil;
  return at::adaptive_max_pool1d(self, output_size);
}
inline Tensor dispatch_add(const Tensor & self, Scalar alpha, const Tensor & other, Tensor out) {

  AutoNoGIL no_gil;
  return at::add_out(out, self, other, alpha);
}
inline Tensor dispatch_add(const Tensor & self, Scalar alpha, const Tensor & other) {

  AutoNoGIL no_gil;
  return self.add(other, alpha);
}
inline Tensor dispatch_add(const Tensor & self, const Tensor & other, Scalar alpha, Tensor result) {

  AutoNoGIL no_gil;
  return at::add_out(result, self, other, alpha);
}
inline Tensor dispatch_add(const Tensor & self, const Tensor & other, Scalar alpha) {

  AutoNoGIL no_gil;
  return self.add(other, alpha);
}
inline Tensor dispatch_addbmm(Scalar beta, const Tensor & self, Scalar alpha, const Tensor & batch1, const Tensor & batch2, Tensor out) {

  AutoNoGIL no_gil;
  return at::addbmm_out(out, self, batch1, batch2, beta, alpha);
}
inline Tensor dispatch_addbmm(Scalar beta, const Tensor & self, Scalar alpha, const Tensor & batch1, const Tensor & batch2) {

  AutoNoGIL no_gil;
  return self.addbmm(batch1, batch2, beta, alpha);
}
inline Tensor dispatch_addbmm(Scalar beta, const Tensor & self, const Tensor & batch1, const Tensor & batch2, Tensor out) {

  AutoNoGIL no_gil;
  return at::addbmm_out(out, self, batch1, batch2, beta, 1);
}
inline Tensor dispatch_addbmm(Scalar beta, const Tensor & self, const Tensor & batch1, const Tensor & batch2) {

  AutoNoGIL no_gil;
  return self.addbmm(batch1, batch2, beta, 1);
}
inline Tensor dispatch_addbmm(const Tensor & self, const Tensor & batch1, const Tensor & batch2, Scalar beta, Scalar alpha, Tensor result) {

  AutoNoGIL no_gil;
  return at::addbmm_out(result, self, batch1, batch2, beta, alpha);
}
inline Tensor dispatch_addbmm(const Tensor & self, const Tensor & batch1, const Tensor & batch2, Scalar beta, Scalar alpha) {

  AutoNoGIL no_gil;
  return self.addbmm(batch1, batch2, beta, alpha);
}
inline Tensor dispatch_addcdiv(const Tensor & self, Scalar value, const Tensor & tensor1, const Tensor & tensor2, Tensor out) {

  AutoNoGIL no_gil;
  return at::addcdiv_out(out, self, tensor1, tensor2, value);
}
inline Tensor dispatch_addcdiv(const Tensor & self, Scalar value, const Tensor & tensor1, const Tensor & tensor2) {

  AutoNoGIL no_gil;
  return self.addcdiv(tensor1, tensor2, value);
}
inline Tensor dispatch_addcdiv(const Tensor & self, const Tensor & tensor1, const Tensor & tensor2, Scalar value, Tensor result) {

  AutoNoGIL no_gil;
  return at::addcdiv_out(result, self, tensor1, tensor2, value);
}
inline Tensor dispatch_addcdiv(const Tensor & self, const Tensor & tensor1, const Tensor & tensor2, Scalar value) {

  AutoNoGIL no_gil;
  return self.addcdiv(tensor1, tensor2, value);
}
inline Tensor dispatch_addcmul(const Tensor & self, Scalar value, const Tensor & tensor1, const Tensor & tensor2, Tensor out) {

  AutoNoGIL no_gil;
  return at::addcmul_out(out, self, tensor1, tensor2, value);
}
inline Tensor dispatch_addcmul(const Tensor & self, Scalar value, const Tensor & tensor1, const Tensor & tensor2) {

  AutoNoGIL no_gil;
  return self.addcmul(tensor1, tensor2, value);
}
inline Tensor dispatch_addcmul(const Tensor & self, const Tensor & tensor1, const Tensor & tensor2, Scalar value, Tensor result) {

  AutoNoGIL no_gil;
  return at::addcmul_out(result, self, tensor1, tensor2, value);
}
inline Tensor dispatch_addcmul(const Tensor & self, const Tensor & tensor1, const Tensor & tensor2, Scalar value) {

  AutoNoGIL no_gil;
  return self.addcmul(tensor1, tensor2, value);
}
inline Tensor dispatch_addmm(Scalar beta, const Tensor & self, Scalar alpha, const Tensor & mat1, const Tensor & mat2, Tensor out) {

  AutoNoGIL no_gil;
  return at::addmm_out(out, self, mat1, mat2, beta, alpha);
}
inline Tensor dispatch_addmm(Scalar beta, const Tensor & self, Scalar alpha, const Tensor & mat1, const Tensor & mat2) {

  AutoNoGIL no_gil;
  return self.addmm(mat1, mat2, beta, alpha);
}
inline Tensor dispatch_addmm(Scalar beta, const Tensor & self, const Tensor & mat1, const Tensor & mat2, Tensor out) {

  AutoNoGIL no_gil;
  return at::addmm_out(out, self, mat1, mat2, beta, 1);
}
inline Tensor dispatch_addmm(Scalar beta, const Tensor & self, const Tensor & mat1, const Tensor & mat2) {

  AutoNoGIL no_gil;
  return self.addmm(mat1, mat2, beta, 1);
}
inline Tensor dispatch_addmm(const Tensor & self, const Tensor & mat1, const Tensor & mat2, Scalar beta, Scalar alpha, Tensor result) {

  AutoNoGIL no_gil;
  return at::addmm_out(result, self, mat1, mat2, beta, alpha);
}
inline Tensor dispatch_addmm(const Tensor & self, const Tensor & mat1, const Tensor & mat2, Scalar beta, Scalar alpha) {

  AutoNoGIL no_gil;
  return self.addmm(mat1, mat2, beta, alpha);
}
inline Tensor dispatch_addmv(Scalar beta, const Tensor & self, Scalar alpha, const Tensor & mat, const Tensor & vec, Tensor out) {

  AutoNoGIL no_gil;
  return at::addmv_out(out, self, mat, vec, beta, alpha);
}
inline Tensor dispatch_addmv(Scalar beta, const Tensor & self, Scalar alpha, const Tensor & mat, const Tensor & vec) {

  AutoNoGIL no_gil;
  return self.addmv(mat, vec, beta, alpha);
}
inline Tensor dispatch_addmv(Scalar beta, const Tensor & self, const Tensor & mat, const Tensor & vec, Tensor out) {

  AutoNoGIL no_gil;
  return at::addmv_out(out, self, mat, vec, beta, 1);
}
inline Tensor dispatch_addmv(Scalar beta, const Tensor & self, const Tensor & mat, const Tensor & vec) {

  AutoNoGIL no_gil;
  return self.addmv(mat, vec, beta, 1);
}
inline Tensor dispatch_addmv(const Tensor & self, const Tensor & mat, const Tensor & vec, Scalar beta, Scalar alpha, Tensor result) {

  AutoNoGIL no_gil;
  return at::addmv_out(result, self, mat, vec, beta, alpha);
}
inline Tensor dispatch_addmv(const Tensor & self, const Tensor & mat, const Tensor & vec, Scalar beta, Scalar alpha) {

  AutoNoGIL no_gil;
  return self.addmv(mat, vec, beta, alpha);
}
inline Tensor dispatch_addmv_(Scalar beta, Tensor self, Scalar alpha, const Tensor & mat, const Tensor & vec) {

  AutoNoGIL no_gil;
  return self.addmv_(mat, vec, beta, alpha);
}
inline Tensor dispatch_addmv_(Scalar beta, Tensor self, const Tensor & mat, const Tensor & vec) {

  AutoNoGIL no_gil;
  return self.addmv_(mat, vec, beta, 1);
}
inline Tensor dispatch_addmv_(Tensor self, const Tensor & mat, const Tensor & vec, Scalar beta, Scalar alpha) {

  AutoNoGIL no_gil;
  return self.addmv_(mat, vec, beta, alpha);
}
inline Tensor dispatch_addr(Scalar beta, const Tensor & self, Scalar alpha, const Tensor & vec1, const Tensor & vec2, Tensor out) {

  AutoNoGIL no_gil;
  return at::addr_out(out, self, vec1, vec2, beta, alpha);
}
inline Tensor dispatch_addr(Scalar beta, const Tensor & self, Scalar alpha, const Tensor & vec1, const Tensor & vec2) {

  AutoNoGIL no_gil;
  return self.addr(vec1, vec2, beta, alpha);
}
inline Tensor dispatch_addr(Scalar beta, const Tensor & self, const Tensor & vec1, const Tensor & vec2, Tensor out) {

  AutoNoGIL no_gil;
  return at::addr_out(out, self, vec1, vec2, beta, 1);
}
inline Tensor dispatch_addr(Scalar beta, const Tensor & self, const Tensor & vec1, const Tensor & vec2) {

  AutoNoGIL no_gil;
  return self.addr(vec1, vec2, beta, 1);
}
inline Tensor dispatch_addr(const Tensor & self, const Tensor & vec1, const Tensor & vec2, Scalar beta, Scalar alpha, Tensor result) {

  AutoNoGIL no_gil;
  return at::addr_out(result, self, vec1, vec2, beta, alpha);
}
inline Tensor dispatch_addr(const Tensor & self, const Tensor & vec1, const Tensor & vec2, Scalar beta, Scalar alpha) {

  AutoNoGIL no_gil;
  return self.addr(vec1, vec2, beta, alpha);
}
inline Tensor dispatch_affine_grid_generator(const Tensor & theta, IntList size) {

  AutoNoGIL no_gil;
  return at::affine_grid_generator(theta, size);
}
inline Tensor dispatch_all(const Tensor & self) {

  AutoNoGIL no_gil;
  return self.all();
}
inline Tensor dispatch_all(const Tensor & self, int64_t dim, bool keepdim, Tensor result) {

  AutoNoGIL no_gil;
  return at::all_out(result, self, dim, keepdim);
}
inline Tensor dispatch_all(const Tensor & self, int64_t dim, bool keepdim) {

  AutoNoGIL no_gil;
  return self.all(dim, keepdim);
}
inline bool dispatch_allclose(const Tensor & self, const Tensor & other, double rtol, double atol, bool equal_nan) {

  AutoNoGIL no_gil;
  return self.allclose(other, rtol, atol, equal_nan);
}
inline Tensor dispatch_alpha_dropout(const Tensor & input, double p, bool train) {

  AutoNoGIL no_gil;
  return at::alpha_dropout(input, p, train);
}
inline Tensor dispatch_alpha_dropout_(Tensor self, double p, bool train) {

  AutoNoGIL no_gil;
  return at::alpha_dropout_(self, p, train);
}
inline Tensor dispatch_any(const Tensor & self) {

  AutoNoGIL no_gil;
  return self.any();
}
inline Tensor dispatch_any(const Tensor & self, int64_t dim, bool keepdim, Tensor result) {

  AutoNoGIL no_gil;
  return at::any_out(result, self, dim, keepdim);
}
inline Tensor dispatch_any(const Tensor & self, int64_t dim, bool keepdim) {

  AutoNoGIL no_gil;
  return self.any(dim, keepdim);
}
inline Tensor dispatch_as_strided(const Tensor & self, IntList size, IntList stride) {

  AutoNoGIL no_gil;
  return self.as_strided(size, stride);
}
inline Tensor dispatch_as_strided(const Tensor & self, IntList size, IntList stride, int64_t storage_offset) {

  AutoNoGIL no_gil;
  return self.as_strided(size, stride, storage_offset);
}
inline Tensor dispatch_as_strided_(Tensor self, IntList size, IntList stride) {

  AutoNoGIL no_gil;
  return self.as_strided_(size, stride);
}
inline Tensor dispatch_as_strided_(Tensor self, IntList size, IntList stride, int64_t storage_offset) {

  AutoNoGIL no_gil;
  return self.as_strided_(size, stride, storage_offset);
}
inline Tensor dispatch_asin(const Tensor & self, Tensor result) {

  AutoNoGIL no_gil;
  return at::asin_out(result, self);
}
inline Tensor dispatch_asin(const Tensor & self) {

  AutoNoGIL no_gil;
  return self.asin();
}
inline Tensor dispatch_asin_(Tensor self) {

  AutoNoGIL no_gil;
  return self.asin_();
}
inline Tensor dispatch_atan(const Tensor & self, Tensor result) {

  AutoNoGIL no_gil;
  return at::atan_out(result, self);
}
inline Tensor dispatch_atan(const Tensor & self) {

  AutoNoGIL no_gil;
  return self.atan();
}
inline Tensor dispatch_atan2(const Tensor & self, const Tensor & other, Tensor result) {

  AutoNoGIL no_gil;
  return at::atan2_out(result, self, other);
}
inline Tensor dispatch_atan2(const Tensor & self, const Tensor & other) {

  AutoNoGIL no_gil;
  return self.atan2(other);
}
inline Tensor dispatch_atan_(Tensor self) {

  AutoNoGIL no_gil;
  return self.atan_();
}
inline Tensor dispatch_avg_pool1d(const Tensor & self, IntList kernel_size, IntList stride, IntList padding, bool ceil_mode, bool count_include_pad) {

  AutoNoGIL no_gil;
  return at::avg_pool1d(self, kernel_size, stride, padding, ceil_mode, count_include_pad);
}
inline Tensor dispatch_baddbmm(Scalar beta, const Tensor & self, Scalar alpha, const Tensor & batch1, const Tensor & batch2, Tensor out) {

  AutoNoGIL no_gil;
  return at::baddbmm_out(out, self, batch1, batch2, beta, alpha);
}
inline Tensor dispatch_baddbmm(Scalar beta, const Tensor & self, Scalar alpha, const Tensor & batch1, const Tensor & batch2) {

  AutoNoGIL no_gil;
  return self.baddbmm(batch1, batch2, beta, alpha);
}
inline Tensor dispatch_baddbmm(Scalar beta, const Tensor & self, const Tensor & batch1, const Tensor & batch2, Tensor out) {

  AutoNoGIL no_gil;
  return at::baddbmm_out(out, self, batch1, batch2, beta, 1);
}
inline Tensor dispatch_baddbmm(Scalar beta, const Tensor & self, const Tensor & batch1, const Tensor & batch2) {

  AutoNoGIL no_gil;
  return self.baddbmm(batch1, batch2, beta, 1);
}
inline Tensor dispatch_baddbmm(const Tensor & self, const Tensor & batch1, const Tensor & batch2, Scalar beta, Scalar alpha, Tensor result) {

  AutoNoGIL no_gil;
  return at::baddbmm_out(result, self, batch1, batch2, beta, alpha);
}
inline Tensor dispatch_baddbmm(const Tensor & self, const Tensor & batch1, const Tensor & batch2, Scalar beta, Scalar alpha) {

  AutoNoGIL no_gil;
  return self.baddbmm(batch1, batch2, beta, alpha);
}
inline Tensor dispatch_bartlett_window(int64_t window_length, const TensorOptions & options) {
  maybe_initialize_cuda(options);
  AutoNoGIL no_gil;
  return torch::bartlett_window(window_length, options);
}
inline Tensor dispatch_bartlett_window(int64_t window_length, bool periodic, const TensorOptions & options) {
  maybe_initialize_cuda(options);
  AutoNoGIL no_gil;
  return torch::bartlett_window(window_length, periodic, options);
}
inline Tensor dispatch_batch_norm(const Tensor & input, const Tensor & weight, const Tensor & bias, const Tensor & running_mean, const Tensor & running_var, bool training, double momentum, double eps, bool cudnn_enabled) {

  AutoNoGIL no_gil;
  return at::batch_norm(input, weight, bias, running_mean, running_var, training, momentum, eps, cudnn_enabled);
}
inline Tensor dispatch_bernoulli(const Tensor & self, Generator * generator, Tensor result) {

  AutoNoGIL no_gil;
  return at::bernoulli_out(result, self, generator);
}
inline Tensor dispatch_bernoulli(const Tensor & self, Generator * generator) {

  AutoNoGIL no_gil;
  return self.bernoulli(generator);
}
inline Tensor dispatch_bernoulli(const Tensor & self, double p, Generator * generator) {

  AutoNoGIL no_gil;
  return self.bernoulli(p, generator);
}
inline Tensor dispatch_bilinear(const Tensor & input1, const Tensor & input2, const Tensor & weight, const Tensor & bias) {

  AutoNoGIL no_gil;
  return at::bilinear(input1, input2, weight, bias);
}
inline Tensor dispatch_binary_cross_entropy_with_logits(const Tensor & self, const Tensor & target, const Tensor & weight, const Tensor & pos_weight, int64_t reduction) {

  AutoNoGIL no_gil;
  return at::binary_cross_entropy_with_logits(self, target, weight, pos_weight, reduction);
}
inline Tensor dispatch_bincount(const Tensor & self, const Tensor & weights, int64_t minlength) {

  AutoNoGIL no_gil;
  return self.bincount(weights, minlength);
}
inline Tensor dispatch_blackman_window(int64_t window_length, const TensorOptions & options) {
  maybe_initialize_cuda(options);
  AutoNoGIL no_gil;
  return torch::blackman_window(window_length, options);
}
inline Tensor dispatch_blackman_window(int64_t window_length, bool periodic, const TensorOptions & options) {
  maybe_initialize_cuda(options);
  AutoNoGIL no_gil;
  return torch::blackman_window(window_length, periodic, options);
}
inline Tensor dispatch_bmm(const Tensor & self, const Tensor & mat2, Tensor result) {

  AutoNoGIL no_gil;
  return at::bmm_out(result, self, mat2);
}
inline Tensor dispatch_bmm(const Tensor & self, const Tensor & mat2) {

  AutoNoGIL no_gil;
  return self.bmm(mat2);
}
inline std::vector<Tensor> dispatch_broadcast_tensors(TensorList tensors) {

  AutoNoGIL no_gil;
  return at::broadcast_tensors(tensors);
}
inline std::tuple<Tensor,Tensor> dispatch_btrifact(const Tensor & self, bool pivot, Tensor & A_LU, Tensor & pivots) {

  AutoNoGIL no_gil;
  return at::btrifact_out(A_LU, pivots, self, pivot);
}
inline std::tuple<Tensor,Tensor> dispatch_btrifact(const Tensor & self, bool pivot) {

  AutoNoGIL no_gil;
  return self.btrifact(pivot);
}
inline std::tuple<Tensor,Tensor,Tensor> dispatch_btrifact_with_info(const Tensor & self, bool pivot, Tensor & A_LU, Tensor & pivots, Tensor & info) {

  AutoNoGIL no_gil;
  return at::btrifact_with_info_out(A_LU, pivots, info, self, pivot);
}
inline std::tuple<Tensor,Tensor,Tensor> dispatch_btrifact_with_info(const Tensor & self, bool pivot) {

  AutoNoGIL no_gil;
  return self.btrifact_with_info(pivot);
}
inline Tensor dispatch_btrisolve(const Tensor & self, const Tensor & LU_data, const Tensor & LU_pivots, Tensor result) {

  AutoNoGIL no_gil;
  return at::btrisolve_out(result, self, LU_data, LU_pivots);
}
inline Tensor dispatch_btrisolve(const Tensor & self, const Tensor & LU_data, const Tensor & LU_pivots) {

  AutoNoGIL no_gil;
  return self.btrisolve(LU_data, LU_pivots);
}
inline Tensor dispatch_cat(TensorList tensors, int64_t dim, Tensor result) {

  AutoNoGIL no_gil;
  return at::cat_out(result, tensors, dim);
}
inline Tensor dispatch_cat(TensorList tensors, int64_t dim) {

  AutoNoGIL no_gil;
  return at::cat(tensors, dim);
}
inline Tensor dispatch_ceil(const Tensor & self, Tensor result) {

  AutoNoGIL no_gil;
  return at::ceil_out(result, self);
}
inline Tensor dispatch_ceil(const Tensor & self) {

  AutoNoGIL no_gil;
  return self.ceil();
}
inline Tensor dispatch_ceil_(Tensor self) {

  AutoNoGIL no_gil;
  return self.ceil_();
}
inline Tensor dispatch_celu(const Tensor & self, Scalar alpha) {

  AutoNoGIL no_gil;
  return at::celu(self, alpha);
}
inline Tensor dispatch_celu_(Tensor self, Scalar alpha) {

  AutoNoGIL no_gil;
  return at::celu_(self, alpha);
}
inline Tensor dispatch_chain_matmul(TensorList matrices) {

  AutoNoGIL no_gil;
  return at::chain_matmul(matrices);
}
inline Tensor dispatch_cholesky(const Tensor & self, bool upper, Tensor result) {

  AutoNoGIL no_gil;
  return at::cholesky_out(result, self, upper);
}
inline Tensor dispatch_cholesky(const Tensor & self, bool upper) {

  AutoNoGIL no_gil;
  return self.cholesky(upper);
}
inline std::vector<Tensor> dispatch_chunk(const Tensor & self, int64_t chunks, int64_t dim) {

  AutoNoGIL no_gil;
  return self.chunk(chunks, dim);
}
inline Tensor dispatch_clamp(const Tensor & self, c10::optional<Scalar> min, c10::optional<Scalar> max, Tensor result) {

  AutoNoGIL no_gil;
  return at::clamp_out(result, self, min, max);
}
inline Tensor dispatch_clamp(const Tensor & self, c10::optional<Scalar> min, c10::optional<Scalar> max) {

  AutoNoGIL no_gil;
  return self.clamp(min, max);
}
inline Tensor dispatch_clamp_(Tensor self, c10::optional<Scalar> min, c10::optional<Scalar> max) {

  AutoNoGIL no_gil;
  return self.clamp_(min, max);
}
inline Tensor dispatch_clamp_max(const Tensor & self, Scalar max, Tensor result) {

  AutoNoGIL no_gil;
  return at::clamp_max_out(result, self, max);
}
inline Tensor dispatch_clamp_max(const Tensor & self, Scalar max) {

  AutoNoGIL no_gil;
  return self.clamp_max(max);
}
inline Tensor dispatch_clamp_max_(Tensor self, Scalar max) {

  AutoNoGIL no_gil;
  return self.clamp_max_(max);
}
inline Tensor dispatch_clamp_min(const Tensor & self, Scalar min, Tensor result) {

  AutoNoGIL no_gil;
  return at::clamp_min_out(result, self, min);
}
inline Tensor dispatch_clamp_min(const Tensor & self, Scalar min) {

  AutoNoGIL no_gil;
  return self.clamp_min(min);
}
inline Tensor dispatch_clamp_min_(Tensor self, Scalar min) {

  AutoNoGIL no_gil;
  return self.clamp_min_(min);
}
inline Tensor dispatch_clone(const Tensor & self) {

  AutoNoGIL no_gil;
  return self.clone();
}
inline Tensor dispatch_constant_pad_nd(const Tensor & self, IntList pad, Scalar value) {

  AutoNoGIL no_gil;
  return at::constant_pad_nd(self, pad, value);
}
inline Tensor dispatch_conv1d(const Tensor & input, const Tensor & weight, const Tensor & bias, IntList stride, IntList padding, IntList dilation, int64_t groups) {

  AutoNoGIL no_gil;
  return at::conv1d(input, weight, bias, stride, padding, dilation, groups);
}
inline Tensor dispatch_conv2d(const Tensor & input, const Tensor & weight, const Tensor & bias, IntList stride, IntList padding, IntList dilation, int64_t groups) {

  AutoNoGIL no_gil;
  return at::conv2d(input, weight, bias, stride, padding, dilation, groups);
}
inline Tensor dispatch_conv3d(const Tensor & input, const Tensor & weight, const Tensor & bias, IntList stride, IntList padding, IntList dilation, int64_t groups) {

  AutoNoGIL no_gil;
  return at::conv3d(input, weight, bias, stride, padding, dilation, groups);
}
inline Tensor dispatch_conv_tbc(const Tensor & self, const Tensor & weight, const Tensor & bias, int64_t pad) {

  AutoNoGIL no_gil;
  return at::conv_tbc(self, weight, bias, pad);
}
inline Tensor dispatch_conv_transpose1d(const Tensor & input, const Tensor & weight, const Tensor & bias, IntList stride, IntList padding, IntList output_padding, int64_t groups, IntList dilation) {

  AutoNoGIL no_gil;
  return at::conv_transpose1d(input, weight, bias, stride, padding, output_padding, groups, dilation);
}
inline Tensor dispatch_conv_transpose2d(const Tensor & input, const Tensor & weight, const Tensor & bias, IntList stride, IntList padding, IntList output_padding, int64_t groups, IntList dilation) {

  AutoNoGIL no_gil;
  return at::conv_transpose2d(input, weight, bias, stride, padding, output_padding, groups, dilation);
}
inline Tensor dispatch_conv_transpose3d(const Tensor & input, const Tensor & weight, const Tensor & bias, IntList stride, IntList padding, IntList output_padding, int64_t groups, IntList dilation) {

  AutoNoGIL no_gil;
  return at::conv_transpose3d(input, weight, bias, stride, padding, output_padding, groups, dilation);
}
inline Tensor dispatch_convolution(const Tensor & input, const Tensor & weight, const Tensor & bias, IntList stride, IntList padding, IntList dilation, bool transposed, IntList output_padding, int64_t groups) {

  AutoNoGIL no_gil;
  return at::convolution(input, weight, bias, stride, padding, dilation, transposed, output_padding, groups);
}
inline Tensor dispatch_cos(const Tensor & self, Tensor result) {

  AutoNoGIL no_gil;
  return at::cos_out(result, self);
}
inline Tensor dispatch_cos(const Tensor & self) {

  AutoNoGIL no_gil;
  return self.cos();
}
inline Tensor dispatch_cos_(Tensor self) {

  AutoNoGIL no_gil;
  return self.cos_();
}
inline Tensor dispatch_cosh(const Tensor & self, Tensor result) {

  AutoNoGIL no_gil;
  return at::cosh_out(result, self);
}
inline Tensor dispatch_cosh(const Tensor & self) {

  AutoNoGIL no_gil;
  return self.cosh();
}
inline Tensor dispatch_cosh_(Tensor self) {

  AutoNoGIL no_gil;
  return self.cosh_();
}
inline Tensor dispatch_cosine_embedding_loss(const Tensor & input1, const Tensor & input2, const Tensor & target, double margin, int64_t reduction) {

  AutoNoGIL no_gil;
  return at::cosine_embedding_loss(input1, input2, target, margin, reduction);
}
inline Tensor dispatch_cosine_similarity(const Tensor & x1, const Tensor & x2, int64_t dim, double eps) {

  AutoNoGIL no_gil;
  return at::cosine_similarity(x1, x2, dim, eps);
}
inline Tensor dispatch_cross(const Tensor & self, const Tensor & other, int64_t dim, Tensor result) {

  AutoNoGIL no_gil;
  return at::cross_out(result, self, other, dim);
}
inline Tensor dispatch_cross(const Tensor & self, const Tensor & other, int64_t dim) {

  AutoNoGIL no_gil;
  return self.cross(other, dim);
}
inline Tensor dispatch_ctc_loss(const Tensor & log_probs, const Tensor & targets, IntList input_lengths, IntList target_lengths, int64_t blank, int64_t reduction) {

  AutoNoGIL no_gil;
  return at::ctc_loss(log_probs, targets, input_lengths, target_lengths, blank, reduction);
}
inline Tensor dispatch_ctc_loss(const Tensor & log_probs, const Tensor & targets, const Tensor & input_lengths, const Tensor & target_lengths, int64_t blank, int64_t reduction) {

  AutoNoGIL no_gil;
  return at::ctc_loss(log_probs, targets, input_lengths, target_lengths, blank, reduction);
}
inline Tensor dispatch_cudnn_affine_grid_generator(const Tensor & theta, int64_t N, int64_t C, int64_t H, int64_t W) {

  AutoNoGIL no_gil;
  return at::cudnn_affine_grid_generator(theta, N, C, H, W);
}
inline std::tuple<Tensor,Tensor,Tensor> dispatch_cudnn_batch_norm(const Tensor & input, const Tensor & weight, const Tensor & bias, const Tensor & running_mean, const Tensor & running_var, bool training, double exponential_average_factor, double epsilon) {

  AutoNoGIL no_gil;
  return at::cudnn_batch_norm(input, weight, bias, running_mean, running_var, training, exponential_average_factor, epsilon);
}
inline Tensor dispatch_cudnn_convolution(const Tensor & self, const Tensor & weight, const Tensor & bias, IntList padding, IntList stride, IntList dilation, int64_t groups, bool benchmark, bool deterministic) {

  AutoNoGIL no_gil;
  return at::cudnn_convolution(self, weight, bias, padding, stride, dilation, groups, benchmark, deterministic);
}
inline Tensor dispatch_cudnn_convolution_transpose(const Tensor & self, const Tensor & weight, const Tensor & bias, IntList padding, IntList output_padding, IntList stride, IntList dilation, int64_t groups, bool benchmark, bool deterministic) {

  AutoNoGIL no_gil;
  return at::cudnn_convolution_transpose(self, weight, bias, padding, output_padding, stride, dilation, groups, benchmark, deterministic);
}
inline Tensor dispatch_cudnn_grid_sampler(const Tensor & self, const Tensor & grid) {

  AutoNoGIL no_gil;
  return at::cudnn_grid_sampler(self, grid);
}
inline bool dispatch_cudnn_is_acceptable(const Tensor & self) {

  AutoNoGIL no_gil;
  return at::cudnn_is_acceptable(self);
}
inline Tensor dispatch_cumprod(const Tensor & self, int64_t dim, Tensor result) {

  AutoNoGIL no_gil;
  return at::cumprod_out(result, self, dim);
}
inline Tensor dispatch_cumprod(const Tensor & self, int64_t dim) {

  AutoNoGIL no_gil;
  return self.cumprod(dim);
}
inline Tensor dispatch_cumprod(const Tensor & self, int64_t dim, ScalarType dtype, Tensor result) {

  AutoNoGIL no_gil;
  return at::cumprod_out(result, self, dim, dtype);
}
inline Tensor dispatch_cumprod(const Tensor & self, int64_t dim, ScalarType dtype) {

  AutoNoGIL no_gil;
  return self.cumprod(dim, dtype);
}
inline Tensor dispatch_cumsum(const Tensor & self, int64_t dim, Tensor result) {

  AutoNoGIL no_gil;
  return at::cumsum_out(result, self, dim);
}
inline Tensor dispatch_cumsum(const Tensor & self, int64_t dim) {

  AutoNoGIL no_gil;
  return self.cumsum(dim);
}
inline Tensor dispatch_cumsum(const Tensor & self, int64_t dim, ScalarType dtype, Tensor result) {

  AutoNoGIL no_gil;
  return at::cumsum_out(result, self, dim, dtype);
}
inline Tensor dispatch_cumsum(const Tensor & self, int64_t dim, ScalarType dtype) {

  AutoNoGIL no_gil;
  return self.cumsum(dim, dtype);
}
inline Tensor dispatch_det(const Tensor & self) {

  AutoNoGIL no_gil;
  return self.det();
}
inline Tensor dispatch_detach(const Tensor & self) {

  AutoNoGIL no_gil;
  return self.detach();
}
inline Tensor dispatch_detach_(Tensor self) {

  AutoNoGIL no_gil;
  return self.detach_();
}
inline Tensor dispatch_diag(const Tensor & self, int64_t diagonal, Tensor result) {

  AutoNoGIL no_gil;
  return at::diag_out(result, self, diagonal);
}
inline Tensor dispatch_diag(const Tensor & self, int64_t diagonal) {

  AutoNoGIL no_gil;
  return self.diag(diagonal);
}
inline Tensor dispatch_diag_embed(const Tensor & self, int64_t offset, int64_t dim1, int64_t dim2) {

  AutoNoGIL no_gil;
  return self.diag_embed(offset, dim1, dim2);
}
inline Tensor dispatch_diagflat(const Tensor & self, int64_t offset) {

  AutoNoGIL no_gil;
  return self.diagflat(offset);
}
inline Tensor dispatch_diagonal(const Tensor & self, int64_t offset, int64_t dim1, int64_t dim2) {

  AutoNoGIL no_gil;
  return self.diagonal(offset, dim1, dim2);
}
inline Tensor dispatch_digamma(const Tensor & self, Tensor result) {

  AutoNoGIL no_gil;
  return at::digamma_out(result, self);
}
inline Tensor dispatch_digamma(const Tensor & self) {

  AutoNoGIL no_gil;
  return self.digamma();
}
inline Tensor dispatch_dist(const Tensor & self, const Tensor & other, Scalar p) {

  AutoNoGIL no_gil;
  return self.dist(other, p);
}
inline Tensor dispatch_div(const Tensor & self, const Tensor & other, Tensor result) {

  AutoNoGIL no_gil;
  return at::div_out(result, self, other);
}
inline Tensor dispatch_div(const Tensor & self, const Tensor & other) {

  AutoNoGIL no_gil;
  return self.div(other);
}
inline Tensor dispatch_dot(const Tensor & self, const Tensor & tensor, Tensor result) {

  AutoNoGIL no_gil;
  return at::dot_out(result, self, tensor);
}
inline Tensor dispatch_dot(const Tensor & self, const Tensor & tensor) {

  AutoNoGIL no_gil;
  return self.dot(tensor);
}
inline Tensor dispatch_dropout(const Tensor & input, double p, bool train) {

  AutoNoGIL no_gil;
  return at::dropout(input, p, train);
}
inline Tensor dispatch_dropout_(Tensor self, double p, bool train) {

  AutoNoGIL no_gil;
  return at::dropout_(self, p, train);
}
inline std::tuple<Tensor,Tensor> dispatch_eig(const Tensor & self, bool eigenvectors, Tensor & e, Tensor & v) {

  AutoNoGIL no_gil;
  return at::eig_out(e, v, self, eigenvectors);
}
inline std::tuple<Tensor,Tensor> dispatch_eig(const Tensor & self, bool eigenvectors) {

  AutoNoGIL no_gil;
  return self.eig(eigenvectors);
}
inline Tensor dispatch_einsum(std::string equation, TensorList tensors) {

  AutoNoGIL no_gil;
  return at::einsum(equation, tensors);
}
inline Tensor dispatch_embedding(const Tensor & weight, const Tensor & indices, int64_t padding_idx, bool scale_grad_by_freq, bool sparse) {

  AutoNoGIL no_gil;
  return at::embedding(weight, indices, padding_idx, scale_grad_by_freq, sparse);
}
inline std::tuple<Tensor,Tensor,Tensor,Tensor> dispatch_embedding_bag(const Tensor & weight, const Tensor & indices, const Tensor & offsets, bool scale_grad_by_freq, int64_t mode, bool sparse) {

  AutoNoGIL no_gil;
  return at::embedding_bag(weight, indices, offsets, scale_grad_by_freq, mode, sparse);
}
inline Tensor dispatch_embedding_renorm_(Tensor self, const Tensor & indices, double max_norm, double norm_type) {

  AutoNoGIL no_gil;
  return at::embedding_renorm_(self, indices, max_norm, norm_type);
}
inline Tensor dispatch_empty(IntList size, Tensor result) {

  AutoNoGIL no_gil;
  return at::empty_out(result, size);
}
inline Tensor dispatch_empty(IntList size, const TensorOptions & options) {
  maybe_initialize_cuda(options);
  AutoNoGIL no_gil;
  return torch::empty(size, options);
}
inline Tensor dispatch_empty_like(const Tensor & self, const TensorOptions & options) {
  maybe_initialize_cuda(options);
  AutoNoGIL no_gil;
  return torch::empty_like(self, options);
}
inline Tensor dispatch_empty_like(const Tensor & self) {

  AutoNoGIL no_gil;
  return torch::empty_like(self);
}
inline Tensor dispatch_empty_strided(IntList size, IntList stride, const TensorOptions & options) {
  maybe_initialize_cuda(options);
  AutoNoGIL no_gil;
  return torch::empty_strided(size, stride, options);
}
inline Tensor dispatch_eq(const Tensor & self, const Tensor & other, Tensor result) {

  AutoNoGIL no_gil;
  return at::eq_out(result, self, other);
}
inline Tensor dispatch_eq(const Tensor & self, const Tensor & other) {

  AutoNoGIL no_gil;
  return self.eq(other);
}
inline Tensor dispatch_eq(const Tensor & self, Scalar other, Tensor result) {

  AutoNoGIL no_gil;
  return at::eq_out(result, self, other);
}
inline Tensor dispatch_eq(const Tensor & self, Scalar other) {

  AutoNoGIL no_gil;
  return self.eq(other);
}
inline bool dispatch_equal(const Tensor & self, const Tensor & other) {

  AutoNoGIL no_gil;
  return self.equal(other);
}
inline Tensor dispatch_erf(const Tensor & self, Tensor result) {

  AutoNoGIL no_gil;
  return at::erf_out(result, self);
}
inline Tensor dispatch_erf(const Tensor & self) {

  AutoNoGIL no_gil;
  return self.erf();
}
inline Tensor dispatch_erf_(Tensor self) {

  AutoNoGIL no_gil;
  return self.erf_();
}
inline Tensor dispatch_erfc(const Tensor & self, Tensor result) {

  AutoNoGIL no_gil;
  return at::erfc_out(result, self);
}
inline Tensor dispatch_erfc(const Tensor & self) {

  AutoNoGIL no_gil;
  return self.erfc();
}
inline Tensor dispatch_erfc_(Tensor self) {

  AutoNoGIL no_gil;
  return self.erfc_();
}
inline Tensor dispatch_erfinv(const Tensor & self, Tensor result) {

  AutoNoGIL no_gil;
  return at::erfinv_out(result, self);
}
inline Tensor dispatch_erfinv(const Tensor & self) {

  AutoNoGIL no_gil;
  return self.erfinv();
}
inline Tensor dispatch_exp(const Tensor & self, Tensor result) {

  AutoNoGIL no_gil;
  return at::exp_out(result, self);
}
inline Tensor dispatch_exp(const Tensor & self) {

  AutoNoGIL no_gil;
  return self.exp();
}
inline Tensor dispatch_exp_(Tensor self) {

  AutoNoGIL no_gil;
  return self.exp_();
}
inline Tensor dispatch_expm1(const Tensor & self, Tensor result) {

  AutoNoGIL no_gil;
  return at::expm1_out(result, self);
}
inline Tensor dispatch_expm1(const Tensor & self) {

  AutoNoGIL no_gil;
  return self.expm1();
}
inline Tensor dispatch_expm1_(Tensor self) {

  AutoNoGIL no_gil;
  return self.expm1_();
}
inline Tensor dispatch_eye(int64_t n, Tensor result) {

  AutoNoGIL no_gil;
  return at::eye_out(result, n);
}
inline Tensor dispatch_eye(int64_t n, const TensorOptions & options) {
  maybe_initialize_cuda(options);
  AutoNoGIL no_gil;
  return torch::eye(n, options);
}
inline Tensor dispatch_eye(int64_t n, int64_t m, Tensor result) {

  AutoNoGIL no_gil;
  return at::eye_out(result, n, m);
}
inline Tensor dispatch_eye(int64_t n, int64_t m, const TensorOptions & options) {
  maybe_initialize_cuda(options);
  AutoNoGIL no_gil;
  return torch::eye(n, m, options);
}
inline Tensor dispatch_feature_alpha_dropout(const Tensor & input, double p, bool train) {

  AutoNoGIL no_gil;
  return at::feature_alpha_dropout(input, p, train);
}
inline Tensor dispatch_feature_alpha_dropout_(Tensor self, double p, bool train) {

  AutoNoGIL no_gil;
  return at::feature_alpha_dropout_(self, p, train);
}
inline Tensor dispatch_feature_dropout(const Tensor & input, double p, bool train) {

  AutoNoGIL no_gil;
  return at::feature_dropout(input, p, train);
}
inline Tensor dispatch_feature_dropout_(Tensor self, double p, bool train) {

  AutoNoGIL no_gil;
  return at::feature_dropout_(self, p, train);
}
inline Tensor dispatch_fft(const Tensor & self, int64_t signal_ndim, bool normalized) {

  AutoNoGIL no_gil;
  return self.fft(signal_ndim, normalized);
}
inline Tensor dispatch_fill_(Tensor self, const Tensor & value) {

  AutoNoGIL no_gil;
  return self.fill_(value);
}
inline Tensor dispatch_fill_(Tensor self, Scalar value) {

  AutoNoGIL no_gil;
  return self.fill_(value);
}
inline Tensor dispatch_flatten(const Tensor & self, int64_t start_dim, int64_t end_dim) {

  AutoNoGIL no_gil;
  return self.flatten(start_dim, end_dim);
}
inline Tensor dispatch_flip(const Tensor & self, IntList dims) {

  AutoNoGIL no_gil;
  return self.flip(dims);
}
inline Tensor dispatch_floor(const Tensor & self, Tensor result) {

  AutoNoGIL no_gil;
  return at::floor_out(result, self);
}
inline Tensor dispatch_floor(const Tensor & self) {

  AutoNoGIL no_gil;
  return self.floor();
}
inline Tensor dispatch_floor_(Tensor self) {

  AutoNoGIL no_gil;
  return self.floor_();
}
inline Tensor dispatch_fmod(const Tensor & self, const Tensor & other, Tensor result) {

  AutoNoGIL no_gil;
  return at::fmod_out(result, self, other);
}
inline Tensor dispatch_fmod(const Tensor & self, const Tensor & other) {

  AutoNoGIL no_gil;
  return self.fmod(other);
}
inline Tensor dispatch_fmod(const Tensor & self, Scalar other, Tensor result) {

  AutoNoGIL no_gil;
  return at::fmod_out(result, self, other);
}
inline Tensor dispatch_fmod(const Tensor & self, Scalar other) {

  AutoNoGIL no_gil;
  return self.fmod(other);
}
inline Tensor dispatch_frac(const Tensor & self, Tensor result) {

  AutoNoGIL no_gil;
  return at::frac_out(result, self);
}
inline Tensor dispatch_frac(const Tensor & self) {

  AutoNoGIL no_gil;
  return self.frac();
}
inline Tensor dispatch_frobenius_norm(const Tensor & self) {

  AutoNoGIL no_gil;
  return at::frobenius_norm(self);
}
inline Tensor dispatch_frobenius_norm(const Tensor & self, IntList dim, bool keepdim, Tensor result) {

  AutoNoGIL no_gil;
  return at::frobenius_norm_out(result, self, dim, keepdim);
}
inline Tensor dispatch_frobenius_norm(const Tensor & self, IntList dim, bool keepdim) {

  AutoNoGIL no_gil;
  return at::frobenius_norm(self, dim, keepdim);
}
inline Tensor dispatch_full(IntList size, Scalar fill_value, Tensor result) {

  AutoNoGIL no_gil;
  return at::full_out(result, size, fill_value);
}
inline Tensor dispatch_full(IntList size, Scalar fill_value, const TensorOptions & options) {
  maybe_initialize_cuda(options);
  AutoNoGIL no_gil;
  return torch::full(size, fill_value, options);
}
inline Tensor dispatch_full_like(const Tensor & self, Scalar fill_value, const TensorOptions & options) {
  maybe_initialize_cuda(options);
  AutoNoGIL no_gil;
  return torch::full_like(self, fill_value, options);
}
inline Tensor dispatch_full_like(const Tensor & self, Scalar fill_value) {

  AutoNoGIL no_gil;
  return torch::full_like(self, fill_value);
}
inline Tensor dispatch_gather(const Tensor & self, int64_t dim, const Tensor & index, Tensor result) {

  AutoNoGIL no_gil;
  return at::gather_out(result, self, dim, index);
}
inline Tensor dispatch_gather(const Tensor & self, int64_t dim, const Tensor & index) {

  AutoNoGIL no_gil;
  return self.gather(dim, index);
}
inline Tensor dispatch_ge(const Tensor & self, const Tensor & other, Tensor result) {

  AutoNoGIL no_gil;
  return at::ge_out(result, self, other);
}
inline Tensor dispatch_ge(const Tensor & self, const Tensor & other) {

  AutoNoGIL no_gil;
  return self.ge(other);
}
inline Tensor dispatch_ge(const Tensor & self, Scalar other, Tensor result) {

  AutoNoGIL no_gil;
  return at::ge_out(result, self, other);
}
inline Tensor dispatch_ge(const Tensor & self, Scalar other) {

  AutoNoGIL no_gil;
  return self.ge(other);
}
inline std::tuple<Tensor,Tensor> dispatch_gels(const Tensor & self, const Tensor & A, Tensor & X, Tensor & qr) {

  AutoNoGIL no_gil;
  return at::gels_out(X, qr, self, A);
}
inline std::tuple<Tensor,Tensor> dispatch_gels(const Tensor & self, const Tensor & A) {

  AutoNoGIL no_gil;
  return self.gels(A);
}
inline std::tuple<Tensor,Tensor> dispatch_geqrf(const Tensor & self, Tensor & result0, Tensor & result1) {

  AutoNoGIL no_gil;
  return at::geqrf_out(result0, result1, self);
}
inline std::tuple<Tensor,Tensor> dispatch_geqrf(const Tensor & self) {

  AutoNoGIL no_gil;
  return self.geqrf();
}
inline Tensor dispatch_ger(const Tensor & self, const Tensor & vec2, Tensor result) {

  AutoNoGIL no_gil;
  return at::ger_out(result, self, vec2);
}
inline Tensor dispatch_ger(const Tensor & self, const Tensor & vec2) {

  AutoNoGIL no_gil;
  return self.ger(vec2);
}
inline std::tuple<Tensor,Tensor> dispatch_gesv(const Tensor & self, const Tensor & A, Tensor & solution, Tensor & lu) {

  AutoNoGIL no_gil;
  return at::gesv_out(solution, lu, self, A);
}
inline std::tuple<Tensor,Tensor> dispatch_gesv(const Tensor & self, const Tensor & A) {

  AutoNoGIL no_gil;
  return self.gesv(A);
}
inline Tensor dispatch_grid_sampler(const Tensor & input, const Tensor & grid, int64_t interpolation_mode, int64_t padding_mode) {

  AutoNoGIL no_gil;
  return at::grid_sampler(input, grid, interpolation_mode, padding_mode);
}
inline Tensor dispatch_grid_sampler_2d(const Tensor & input, const Tensor & grid, int64_t interpolation_mode, int64_t padding_mode) {

  AutoNoGIL no_gil;
  return at::grid_sampler_2d(input, grid, interpolation_mode, padding_mode);
}
inline Tensor dispatch_grid_sampler_3d(const Tensor & input, const Tensor & grid, int64_t interpolation_mode, int64_t padding_mode) {

  AutoNoGIL no_gil;
  return at::grid_sampler_3d(input, grid, interpolation_mode, padding_mode);
}
inline Tensor dispatch_group_norm(const Tensor & input, int64_t num_groups, const Tensor & weight, const Tensor & bias, double eps, bool cudnn_enabled) {

  AutoNoGIL no_gil;
  return at::group_norm(input, num_groups, weight, bias, eps, cudnn_enabled);
}
inline std::tuple<Tensor,Tensor> dispatch_gru(const Tensor & data, const Tensor & batch_sizes, const Tensor & hx, TensorList params, bool has_biases, int64_t num_layers, double dropout, bool train, bool bidirectional) {

  AutoNoGIL no_gil;
  return at::gru(data, batch_sizes, hx, params, has_biases, num_layers, dropout, train, bidirectional);
}
inline std::tuple<Tensor,Tensor> dispatch_gru(const Tensor & input, const Tensor & hx, TensorList params, bool has_biases, int64_t num_layers, double dropout, bool train, bool bidirectional, bool batch_first) {

  AutoNoGIL no_gil;
  return at::gru(input, hx, params, has_biases, num_layers, dropout, train, bidirectional, batch_first);
}
inline Tensor dispatch_gru_cell(const Tensor & input, const Tensor & hx, const Tensor & w_ih, const Tensor & w_hh, const Tensor & b_ih, const Tensor & b_hh) {

  AutoNoGIL no_gil;
  return at::gru_cell(input, hx, w_ih, w_hh, b_ih, b_hh);
}
inline Tensor dispatch_gt(const Tensor & self, const Tensor & other, Tensor result) {

  AutoNoGIL no_gil;
  return at::gt_out(result, self, other);
}
inline Tensor dispatch_gt(const Tensor & self, const Tensor & other) {

  AutoNoGIL no_gil;
  return self.gt(other);
}
inline Tensor dispatch_gt(const Tensor & self, Scalar other, Tensor result) {

  AutoNoGIL no_gil;
  return at::gt_out(result, self, other);
}
inline Tensor dispatch_gt(const Tensor & self, Scalar other) {

  AutoNoGIL no_gil;
  return self.gt(other);
}
inline Tensor dispatch_hamming_window(int64_t window_length, const TensorOptions & options) {
  maybe_initialize_cuda(options);
  AutoNoGIL no_gil;
  return torch::hamming_window(window_length, options);
}
inline Tensor dispatch_hamming_window(int64_t window_length, bool periodic, const TensorOptions & options) {
  maybe_initialize_cuda(options);
  AutoNoGIL no_gil;
  return torch::hamming_window(window_length, periodic, options);
}
inline Tensor dispatch_hamming_window(int64_t window_length, bool periodic, double alpha, const TensorOptions & options) {
  maybe_initialize_cuda(options);
  AutoNoGIL no_gil;
  return torch::hamming_window(window_length, periodic, alpha, options);
}
inline Tensor dispatch_hamming_window(int64_t window_length, bool periodic, double alpha, double beta, const TensorOptions & options) {
  maybe_initialize_cuda(options);
  AutoNoGIL no_gil;
  return torch::hamming_window(window_length, periodic, alpha, beta, options);
}
inline Tensor dispatch_hann_window(int64_t window_length, const TensorOptions & options) {
  maybe_initialize_cuda(options);
  AutoNoGIL no_gil;
  return torch::hann_window(window_length, options);
}
inline Tensor dispatch_hann_window(int64_t window_length, bool periodic, const TensorOptions & options) {
  maybe_initialize_cuda(options);
  AutoNoGIL no_gil;
  return torch::hann_window(window_length, periodic, options);
}
inline Tensor dispatch_hardshrink(const Tensor & self, Scalar lambd) {

  AutoNoGIL no_gil;
  return self.hardshrink(lambd);
}
inline Tensor dispatch_hinge_embedding_loss(const Tensor & self, const Tensor & target, double margin, int64_t reduction) {

  AutoNoGIL no_gil;
  return at::hinge_embedding_loss(self, target, margin, reduction);
}
inline Tensor dispatch_histc(const Tensor & self, int64_t bins, Scalar min, Scalar max, Tensor result) {

  AutoNoGIL no_gil;
  return at::histc_out(result, self, bins, min, max);
}
inline Tensor dispatch_histc(const Tensor & self, int64_t bins, Scalar min, Scalar max) {

  AutoNoGIL no_gil;
  return self.histc(bins, min, max);
}
inline Tensor dispatch_hspmm(const Tensor & mat1, const Tensor & mat2, Tensor result) {

  AutoNoGIL no_gil;
  return at::hspmm_out(result, mat1, mat2);
}
inline Tensor dispatch_hspmm(const Tensor & mat1, const Tensor & mat2) {

  AutoNoGIL no_gil;
  return at::hspmm(mat1, mat2);
}
inline Tensor dispatch_ifft(const Tensor & self, int64_t signal_ndim, bool normalized) {

  AutoNoGIL no_gil;
  return self.ifft(signal_ndim, normalized);
}
inline Tensor dispatch_index_put(const Tensor & self, TensorList indices, const Tensor & values, bool accumulate) {

  AutoNoGIL no_gil;
  return self.index_put(indices, values, accumulate);
}
inline Tensor dispatch_index_put_(Tensor self, TensorList indices, const Tensor & values, bool accumulate) {

  AutoNoGIL no_gil;
  return self.index_put_(indices, values, accumulate);
}
inline Tensor dispatch_index_select(const Tensor & self, int64_t dim, const Tensor & index, Tensor result) {

  AutoNoGIL no_gil;
  return at::index_select_out(result, self, dim, index);
}
inline Tensor dispatch_index_select(const Tensor & self, int64_t dim, const Tensor & index) {

  AutoNoGIL no_gil;
  return self.index_select(dim, index);
}
inline Tensor dispatch_instance_norm(const Tensor & input, const Tensor & weight, const Tensor & bias, const Tensor & running_mean, const Tensor & running_var, bool use_input_stats, double momentum, double eps, bool cudnn_enabled) {

  AutoNoGIL no_gil;
  return at::instance_norm(input, weight, bias, running_mean, running_var, use_input_stats, momentum, eps, cudnn_enabled);
}
inline Tensor dispatch_inverse(const Tensor & self, Tensor result) {

  AutoNoGIL no_gil;
  return at::inverse_out(result, self);
}
inline Tensor dispatch_inverse(const Tensor & self) {

  AutoNoGIL no_gil;
  return self.inverse();
}
inline Tensor dispatch_irfft(const Tensor & self, int64_t signal_ndim, bool normalized, bool onesided, IntList signal_sizes) {

  AutoNoGIL no_gil;
  return self.irfft(signal_ndim, normalized, onesided, signal_sizes);
}
inline bool dispatch_is_complex(const Tensor & self) {

  AutoNoGIL no_gil;
  return self.is_complex();
}
inline bool dispatch_is_distributed(const Tensor & self) {

  AutoNoGIL no_gil;
  return self.is_distributed();
}
inline bool dispatch_is_floating_point(const Tensor & self) {

  AutoNoGIL no_gil;
  return self.is_floating_point();
}
inline bool dispatch_is_nonzero(const Tensor & self) {

  AutoNoGIL no_gil;
  return self.is_nonzero();
}
inline bool dispatch_is_same_size(const Tensor & self, const Tensor & other) {

  AutoNoGIL no_gil;
  return self.is_same_size(other);
}
inline bool dispatch_is_signed(const Tensor & self) {

  AutoNoGIL no_gil;
  return self.is_signed();
}
inline Tensor dispatch_isclose(const Tensor & self, const Tensor & other, double rtol, double atol, bool equal_nan) {

  AutoNoGIL no_gil;
  return self.isclose(other, rtol, atol, equal_nan);
}
inline Tensor dispatch_kl_div(const Tensor & self, const Tensor & target, int64_t reduction) {

  AutoNoGIL no_gil;
  return at::kl_div(self, target, reduction);
}
inline std::tuple<Tensor,Tensor> dispatch_kthvalue(const Tensor & self, int64_t k, int64_t dim, bool keepdim, Tensor & values, Tensor & indices) {

  AutoNoGIL no_gil;
  return at::kthvalue_out(values, indices, self, k, dim, keepdim);
}
inline std::tuple<Tensor,Tensor> dispatch_kthvalue(const Tensor & self, int64_t k, int64_t dim, bool keepdim) {

  AutoNoGIL no_gil;
  return self.kthvalue(k, dim, keepdim);
}
inline Tensor dispatch_layer_norm(const Tensor & input, IntList normalized_shape, const Tensor & weight, const Tensor & bias, double eps, bool cudnn_enable) {

  AutoNoGIL no_gil;
  return at::layer_norm(input, normalized_shape, weight, bias, eps, cudnn_enable);
}
inline Tensor dispatch_le(const Tensor & self, const Tensor & other, Tensor result) {

  AutoNoGIL no_gil;
  return at::le_out(result, self, other);
}
inline Tensor dispatch_le(const Tensor & self, const Tensor & other) {

  AutoNoGIL no_gil;
  return self.le(other);
}
inline Tensor dispatch_le(const Tensor & self, Scalar other, Tensor result) {

  AutoNoGIL no_gil;
  return at::le_out(result, self, other);
}
inline Tensor dispatch_le(const Tensor & self, Scalar other) {

  AutoNoGIL no_gil;
  return self.le(other);
}
inline Tensor dispatch_lerp(const Tensor & self, const Tensor & end, Scalar weight, Tensor result) {

  AutoNoGIL no_gil;
  return at::lerp_out(result, self, end, weight);
}
inline Tensor dispatch_lerp(const Tensor & self, const Tensor & end, Scalar weight) {

  AutoNoGIL no_gil;
  return self.lerp(end, weight);
}
inline Tensor dispatch_lgamma(const Tensor & self, Tensor result) {

  AutoNoGIL no_gil;
  return at::lgamma_out(result, self);
}
inline Tensor dispatch_lgamma(const Tensor & self) {

  AutoNoGIL no_gil;
  return self.lgamma();
}
inline Tensor dispatch_linspace(Scalar start, Scalar end, int64_t steps, Tensor result) {

  AutoNoGIL no_gil;
  return at::linspace_out(result, start, end, steps);
}
inline Tensor dispatch_linspace(Scalar start, Scalar end, int64_t steps, const TensorOptions & options) {
  maybe_initialize_cuda(options);
  AutoNoGIL no_gil;
  return torch::linspace(start, end, steps, options);
}
inline Tensor dispatch_log(const Tensor & self, Tensor result) {

  AutoNoGIL no_gil;
  return at::log_out(result, self);
}
inline Tensor dispatch_log(const Tensor & self) {

  AutoNoGIL no_gil;
  return self.log();
}
inline Tensor dispatch_log10(const Tensor & self, Tensor result) {

  AutoNoGIL no_gil;
  return at::log10_out(result, self);
}
inline Tensor dispatch_log10(const Tensor & self) {

  AutoNoGIL no_gil;
  return self.log10();
}
inline Tensor dispatch_log10_(Tensor self) {

  AutoNoGIL no_gil;
  return self.log10_();
}
inline Tensor dispatch_log1p(const Tensor & self, Tensor result) {

  AutoNoGIL no_gil;
  return at::log1p_out(result, self);
}
inline Tensor dispatch_log1p(const Tensor & self) {

  AutoNoGIL no_gil;
  return self.log1p();
}
inline Tensor dispatch_log1p_(Tensor self) {

  AutoNoGIL no_gil;
  return self.log1p_();
}
inline Tensor dispatch_log2(const Tensor & self, Tensor result) {

  AutoNoGIL no_gil;
  return at::log2_out(result, self);
}
inline Tensor dispatch_log2(const Tensor & self) {

  AutoNoGIL no_gil;
  return self.log2();
}
inline Tensor dispatch_log2_(Tensor self) {

  AutoNoGIL no_gil;
  return self.log2_();
}
inline Tensor dispatch_log_(Tensor self) {

  AutoNoGIL no_gil;
  return self.log_();
}
inline Tensor dispatch_log_softmax(const Tensor & self, int64_t dim) {

  AutoNoGIL no_gil;
  return self.log_softmax(dim);
}
inline Tensor dispatch_log_softmax(const Tensor & self, int64_t dim, ScalarType dtype) {

  AutoNoGIL no_gil;
  return self.log_softmax(dim, dtype);
}
inline Tensor dispatch_logdet(const Tensor & self) {

  AutoNoGIL no_gil;
  return self.logdet();
}
inline Tensor dispatch_logspace(Scalar start, Scalar end, Tensor result) {

  AutoNoGIL no_gil;
  return at::logspace_out(result, start, end);
}
inline Tensor dispatch_logspace(Scalar start, Scalar end, const TensorOptions & options) {
  maybe_initialize_cuda(options);
  AutoNoGIL no_gil;
  return torch::logspace(start, end, options);
}
inline Tensor dispatch_logspace(Scalar start, Scalar end, int64_t steps, Tensor result) {

  AutoNoGIL no_gil;
  return at::logspace_out(result, start, end, steps);
}
inline Tensor dispatch_logspace(Scalar start, Scalar end, int64_t steps, const TensorOptions & options) {
  maybe_initialize_cuda(options);
  AutoNoGIL no_gil;
  return torch::logspace(start, end, steps, options);
}
inline Tensor dispatch_logsumexp(const Tensor & self, int64_t dim, bool keepdim, Tensor result) {

  AutoNoGIL no_gil;
  return at::logsumexp_out(result, self, dim, keepdim);
}
inline Tensor dispatch_logsumexp(const Tensor & self, int64_t dim, bool keepdim) {

  AutoNoGIL no_gil;
  return self.logsumexp(dim, keepdim);
}
inline std::tuple<Tensor,Tensor,Tensor> dispatch_lstm(const Tensor & data, const Tensor & batch_sizes, TensorList hx, TensorList params, bool has_biases, int64_t num_layers, double dropout, bool train, bool bidirectional) {

  AutoNoGIL no_gil;
  return at::lstm(data, batch_sizes, hx, params, has_biases, num_layers, dropout, train, bidirectional);
}
inline std::tuple<Tensor,Tensor,Tensor> dispatch_lstm(const Tensor & input, TensorList hx, TensorList params, bool has_biases, int64_t num_layers, double dropout, bool train, bool bidirectional, bool batch_first) {

  AutoNoGIL no_gil;
  return at::lstm(input, hx, params, has_biases, num_layers, dropout, train, bidirectional, batch_first);
}
inline std::tuple<Tensor,Tensor> dispatch_lstm_cell(const Tensor & input, TensorList hx, const Tensor & w_ih, const Tensor & w_hh, const Tensor & b_ih, const Tensor & b_hh) {

  AutoNoGIL no_gil;
  return at::lstm_cell(input, hx, w_ih, w_hh, b_ih, b_hh);
}
inline Tensor dispatch_lt(const Tensor & self, const Tensor & other, Tensor result) {

  AutoNoGIL no_gil;
  return at::lt_out(result, self, other);
}
inline Tensor dispatch_lt(const Tensor & self, const Tensor & other) {

  AutoNoGIL no_gil;
  return self.lt(other);
}
inline Tensor dispatch_lt(const Tensor & self, Scalar other, Tensor result) {

  AutoNoGIL no_gil;
  return at::lt_out(result, self, other);
}
inline Tensor dispatch_lt(const Tensor & self, Scalar other) {

  AutoNoGIL no_gil;
  return self.lt(other);
}
inline Tensor dispatch_margin_ranking_loss(const Tensor & input1, const Tensor & input2, const Tensor & target, double margin, int64_t reduction) {

  AutoNoGIL no_gil;
  return at::margin_ranking_loss(input1, input2, target, margin, reduction);
}
inline Tensor dispatch_masked_select(const Tensor & self, const Tensor & mask, Tensor result) {

  AutoNoGIL no_gil;
  return at::masked_select_out(result, self, mask);
}
inline Tensor dispatch_masked_select(const Tensor & self, const Tensor & mask) {

  AutoNoGIL no_gil;
  return self.masked_select(mask);
}
inline Tensor dispatch_matmul(const Tensor & self, const Tensor & other, Tensor result) {

  AutoNoGIL no_gil;
  return at::matmul_out(result, self, other);
}
inline Tensor dispatch_matmul(const Tensor & self, const Tensor & other) {

  AutoNoGIL no_gil;
  return self.matmul(other);
}
inline Tensor dispatch_matrix_power(const Tensor & self, int64_t n) {

  AutoNoGIL no_gil;
  return self.matrix_power(n);
}
inline Tensor dispatch_matrix_rank(const Tensor & self, bool symmetric) {

  AutoNoGIL no_gil;
  return at::matrix_rank(self, symmetric);
}
inline Tensor dispatch_matrix_rank(const Tensor & self, double tol, bool symmetric) {

  AutoNoGIL no_gil;
  return at::matrix_rank(self, tol, symmetric);
}
inline Tensor dispatch_max(const Tensor & self) {

  AutoNoGIL no_gil;
  return self.max();
}
inline Tensor dispatch_max(const Tensor & self, const Tensor & other, Tensor result) {

  AutoNoGIL no_gil;
  return at::max_out(result, self, other);
}
inline Tensor dispatch_max(const Tensor & self, const Tensor & other) {

  AutoNoGIL no_gil;
  return self.max(other);
}
inline std::tuple<Tensor,Tensor> dispatch_max(const Tensor & self, int64_t dim, bool keepdim, Tensor & max, Tensor & max_values) {

  AutoNoGIL no_gil;
  return at::max_out(max, max_values, self, dim, keepdim);
}
inline std::tuple<Tensor,Tensor> dispatch_max(const Tensor & self, int64_t dim, bool keepdim) {

  AutoNoGIL no_gil;
  return self.max(dim, keepdim);
}
inline std::tuple<Tensor,Tensor> dispatch_max_pool1d_with_indices(const Tensor & self, IntList kernel_size, IntList stride, IntList padding, IntList dilation, bool ceil_mode) {

  AutoNoGIL no_gil;
  return at::max_pool1d_with_indices(self, kernel_size, stride, padding, dilation, ceil_mode);
}
inline Tensor dispatch_mean(const Tensor & self) {

  AutoNoGIL no_gil;
  return self.mean();
}
inline Tensor dispatch_mean(const Tensor & self, ScalarType dtype) {

  AutoNoGIL no_gil;
  return self.mean(dtype);
}
inline Tensor dispatch_mean(const Tensor & self, IntList dim, ScalarType dtype, Tensor result) {

  AutoNoGIL no_gil;
  return at::mean_out(result, self, dim, dtype);
}
inline Tensor dispatch_mean(const Tensor & self, IntList dim, ScalarType dtype) {

  AutoNoGIL no_gil;
  return self.mean(dim, dtype);
}
inline Tensor dispatch_mean(const Tensor & self, IntList dim, bool keepdim, ScalarType dtype, Tensor result) {

  AutoNoGIL no_gil;
  return at::mean_out(result, self, dim, keepdim, dtype);
}
inline Tensor dispatch_mean(const Tensor & self, IntList dim, bool keepdim, ScalarType dtype) {

  AutoNoGIL no_gil;
  return self.mean(dim, keepdim, dtype);
}
inline Tensor dispatch_mean(const Tensor & self, IntList dim, bool keepdim, Tensor result) {

  AutoNoGIL no_gil;
  return at::mean_out(result, self, dim, keepdim);
}
inline Tensor dispatch_mean(const Tensor & self, IntList dim, bool keepdim) {

  AutoNoGIL no_gil;
  return self.mean(dim, keepdim);
}
inline Tensor dispatch_median(const Tensor & self) {

  AutoNoGIL no_gil;
  return self.median();
}
inline std::tuple<Tensor,Tensor> dispatch_median(const Tensor & self, int64_t dim, bool keepdim, Tensor & values, Tensor & indices) {

  AutoNoGIL no_gil;
  return at::median_out(values, indices, self, dim, keepdim);
}
inline std::tuple<Tensor,Tensor> dispatch_median(const Tensor & self, int64_t dim, bool keepdim) {

  AutoNoGIL no_gil;
  return self.median(dim, keepdim);
}
inline std::vector<Tensor> dispatch_meshgrid(TensorList tensors) {

  AutoNoGIL no_gil;
  return at::meshgrid(tensors);
}
inline Tensor dispatch_min(const Tensor & self) {

  AutoNoGIL no_gil;
  return self.min();
}
inline Tensor dispatch_min(const Tensor & self, const Tensor & other, Tensor result) {

  AutoNoGIL no_gil;
  return at::min_out(result, self, other);
}
inline Tensor dispatch_min(const Tensor & self, const Tensor & other) {

  AutoNoGIL no_gil;
  return self.min(other);
}
inline std::tuple<Tensor,Tensor> dispatch_min(const Tensor & self, int64_t dim, bool keepdim, Tensor & min, Tensor & min_indices) {

  AutoNoGIL no_gil;
  return at::min_out(min, min_indices, self, dim, keepdim);
}
inline std::tuple<Tensor,Tensor> dispatch_min(const Tensor & self, int64_t dim, bool keepdim) {

  AutoNoGIL no_gil;
  return self.min(dim, keepdim);
}
inline std::tuple<Tensor,Tensor,Tensor> dispatch_miopen_batch_norm(const Tensor & input, const Tensor & weight, const Tensor & bias, const Tensor & running_mean, const Tensor & running_var, bool training, double exponential_average_factor, double epsilon) {

  AutoNoGIL no_gil;
  return at::miopen_batch_norm(input, weight, bias, running_mean, running_var, training, exponential_average_factor, epsilon);
}
inline Tensor dispatch_miopen_convolution(const Tensor & self, const Tensor & weight, const Tensor & bias, IntList padding, IntList stride, IntList dilation, int64_t groups, bool benchmark, bool deterministic) {

  AutoNoGIL no_gil;
  return at::miopen_convolution(self, weight, bias, padding, stride, dilation, groups, benchmark, deterministic);
}
inline Tensor dispatch_miopen_convolution_transpose(const Tensor & self, const Tensor & weight, const Tensor & bias, IntList padding, IntList output_padding, IntList stride, IntList dilation, int64_t groups, bool benchmark, bool deterministic) {

  AutoNoGIL no_gil;
  return at::miopen_convolution_transpose(self, weight, bias, padding, output_padding, stride, dilation, groups, benchmark, deterministic);
}
inline Tensor dispatch_mkldnn_convolution(const Tensor & self, const Tensor & weight, const Tensor & bias, IntList padding, IntList stride, IntList dilation, int64_t groups) {

  AutoNoGIL no_gil;
  return at::mkldnn_convolution(self, weight, bias, padding, stride, dilation, groups);
}
inline std::tuple<Tensor,Tensor> dispatch_mkldnn_convolution_backward_weights(IntList weight_size, const Tensor & grad_output, const Tensor & self, IntList padding, IntList stride, IntList dilation, int64_t groups, bool bias_defined) {

  AutoNoGIL no_gil;
  return at::mkldnn_convolution_backward_weights(weight_size, grad_output, self, padding, stride, dilation, groups, bias_defined);
}
inline Tensor dispatch_mm(const Tensor & self, const Tensor & mat2, Tensor result) {

  AutoNoGIL no_gil;
  return at::mm_out(result, self, mat2);
}
inline Tensor dispatch_mm(const Tensor & self, const Tensor & mat2) {

  AutoNoGIL no_gil;
  return self.mm(mat2);
}
inline std::tuple<Tensor,Tensor> dispatch_mode(const Tensor & self, int64_t dim, bool keepdim, Tensor & values, Tensor & indices) {

  AutoNoGIL no_gil;
  return at::mode_out(values, indices, self, dim, keepdim);
}
inline std::tuple<Tensor,Tensor> dispatch_mode(const Tensor & self, int64_t dim, bool keepdim) {

  AutoNoGIL no_gil;
  return self.mode(dim, keepdim);
}
inline Tensor dispatch_mul(const Tensor & self, const Tensor & other, Tensor result) {

  AutoNoGIL no_gil;
  return at::mul_out(result, self, other);
}
inline Tensor dispatch_mul(const Tensor & self, const Tensor & other) {

  AutoNoGIL no_gil;
  return self.mul(other);
}
inline Tensor dispatch_multinomial(const Tensor & self, int64_t num_samples, bool replacement, Generator * generator, Tensor result) {

  AutoNoGIL no_gil;
  return at::multinomial_out(result, self, num_samples, replacement, generator);
}
inline Tensor dispatch_multinomial(const Tensor & self, int64_t num_samples, bool replacement, Generator * generator) {

  AutoNoGIL no_gil;
  return self.multinomial(num_samples, replacement, generator);
}
inline Tensor dispatch_mv(const Tensor & self, const Tensor & vec, Tensor result) {

  AutoNoGIL no_gil;
  return at::mv_out(result, self, vec);
}
inline Tensor dispatch_mv(const Tensor & self, const Tensor & vec) {

  AutoNoGIL no_gil;
  return self.mv(vec);
}
inline Tensor dispatch_mvlgamma(const Tensor & self, int64_t p) {

  AutoNoGIL no_gil;
  return self.mvlgamma(p);
}
inline Tensor dispatch_narrow(const Tensor & self, int64_t dim, int64_t start, int64_t length) {

  AutoNoGIL no_gil;
  return self.narrow(dim, start, length);
}
inline std::tuple<Tensor,Tensor,Tensor> dispatch_native_batch_norm(const Tensor & input, const Tensor & weight, const Tensor & bias, const Tensor & running_mean, const Tensor & running_var, bool training, double momentum, double eps) {

  AutoNoGIL no_gil;
  return at::native_batch_norm(input, weight, bias, running_mean, running_var, training, momentum, eps);
}
inline Tensor dispatch_native_clone(const Tensor & self) {

  AutoNoGIL no_gil;
  return at::native_clone(self);
}
inline Tensor dispatch_native_norm(const Tensor & self, Scalar p) {

  AutoNoGIL no_gil;
  return at::native_norm(self, p);
}
inline Tensor dispatch_native_pow(const Tensor & self, Scalar exponent, Tensor result) {

  AutoNoGIL no_gil;
  return at::native_pow_out(result, self, exponent);
}
inline Tensor dispatch_native_pow(const Tensor & self, Scalar exponent) {

  AutoNoGIL no_gil;
  return at::native_pow(self, exponent);
}
inline Tensor dispatch_native_resize_as_(Tensor self, const Tensor & the_template) {

  AutoNoGIL no_gil;
  return at::native_resize_as_(self, the_template);
}
inline Tensor dispatch_native_zero_(Tensor self) {

  AutoNoGIL no_gil;
  return at::native_zero_(self);
}
inline Tensor dispatch_ne(const Tensor & self, const Tensor & other, Tensor result) {

  AutoNoGIL no_gil;
  return at::ne_out(result, self, other);
}
inline Tensor dispatch_ne(const Tensor & self, const Tensor & other) {

  AutoNoGIL no_gil;
  return self.ne(other);
}
inline Tensor dispatch_ne(const Tensor & self, Scalar other, Tensor result) {

  AutoNoGIL no_gil;
  return at::ne_out(result, self, other);
}
inline Tensor dispatch_ne(const Tensor & self, Scalar other) {

  AutoNoGIL no_gil;
  return self.ne(other);
}
inline Tensor dispatch_neg(const Tensor & self, Tensor result) {

  AutoNoGIL no_gil;
  return at::neg_out(result, self);
}
inline Tensor dispatch_neg(const Tensor & self) {

  AutoNoGIL no_gil;
  return self.neg();
}
inline Tensor dispatch_nonzero(const Tensor & self, Tensor result) {

  AutoNoGIL no_gil;
  return at::nonzero_out(result, self);
}
inline Tensor dispatch_nonzero(const Tensor & self) {

  AutoNoGIL no_gil;
  return self.nonzero();
}
inline Tensor dispatch_norm(const Tensor & self, Scalar p) {

  AutoNoGIL no_gil;
  return self.norm(p);
}
inline Tensor dispatch_norm(const Tensor & self, Scalar p, int64_t dim, bool keepdim, Tensor result) {

  AutoNoGIL no_gil;
  return at::norm_out(result, self, p, dim, keepdim);
}
inline Tensor dispatch_norm(const Tensor & self, Scalar p, int64_t dim, bool keepdim) {

  AutoNoGIL no_gil;
  return self.norm(p, dim, keepdim);
}
inline Tensor dispatch_norm_except_dim(const Tensor & v, int64_t pow, int64_t dim) {

  AutoNoGIL no_gil;
  return at::norm_except_dim(v, pow, dim);
}
inline Tensor dispatch_normal(const Tensor & mean, const Tensor & std, Generator * generator, Tensor output) {

  AutoNoGIL no_gil;
  return at::normal_out(output, mean, std, generator);
}
inline Tensor dispatch_normal(const Tensor & mean, const Tensor & std, Generator * generator) {

  AutoNoGIL no_gil;
  return at::normal(mean, std, generator);
}
inline Tensor dispatch_normal(const Tensor & mean, double std, Generator * generator, Tensor output) {

  AutoNoGIL no_gil;
  return at::normal_out(output, mean, std, generator);
}
inline Tensor dispatch_normal(const Tensor & mean, double std, Generator * generator) {

  AutoNoGIL no_gil;
  return at::normal(mean, std, generator);
}
inline Tensor dispatch_normal(double mean, const Tensor & std, Generator * generator, Tensor output) {

  AutoNoGIL no_gil;
  return at::normal_out(output, mean, std, generator);
}
inline Tensor dispatch_normal(double mean, const Tensor & std, Generator * generator) {

  AutoNoGIL no_gil;
  return at::normal(mean, std, generator);
}
inline Tensor dispatch_nuclear_norm(const Tensor & self, bool keepdim, Tensor result) {

  AutoNoGIL no_gil;
  return at::nuclear_norm_out(result, self, keepdim);
}
inline Tensor dispatch_nuclear_norm(const Tensor & self, bool keepdim) {

  AutoNoGIL no_gil;
  return at::nuclear_norm(self, keepdim);
}
inline int64_t dispatch_numel(const Tensor & self) {

  AutoNoGIL no_gil;
  return self.numel();
}
inline Tensor dispatch_ones(IntList size, Tensor result) {

  AutoNoGIL no_gil;
  return at::ones_out(result, size);
}
inline Tensor dispatch_ones(IntList size, const TensorOptions & options) {
  maybe_initialize_cuda(options);
  AutoNoGIL no_gil;
  return torch::ones(size, options);
}
inline Tensor dispatch_ones_like(const Tensor & self, const TensorOptions & options) {
  maybe_initialize_cuda(options);
  AutoNoGIL no_gil;
  return torch::ones_like(self, options);
}
inline Tensor dispatch_ones_like(const Tensor & self) {

  AutoNoGIL no_gil;
  return torch::ones_like(self);
}
inline Tensor dispatch_orgqr(const Tensor & self, const Tensor & input2, Tensor result) {

  AutoNoGIL no_gil;
  return at::orgqr_out(result, self, input2);
}
inline Tensor dispatch_orgqr(const Tensor & self, const Tensor & input2) {

  AutoNoGIL no_gil;
  return self.orgqr(input2);
}
inline Tensor dispatch_ormqr(const Tensor & self, const Tensor & input2, const Tensor & input3, bool left, bool transpose, Tensor result) {

  AutoNoGIL no_gil;
  return at::ormqr_out(result, self, input2, input3, left, transpose);
}
inline Tensor dispatch_ormqr(const Tensor & self, const Tensor & input2, const Tensor & input3, bool left, bool transpose) {

  AutoNoGIL no_gil;
  return self.ormqr(input2, input3, left, transpose);
}
inline Tensor dispatch_pairwise_distance(const Tensor & x1, const Tensor & x2, double p, double eps, bool keepdim) {

  AutoNoGIL no_gil;
  return at::pairwise_distance(x1, x2, p, eps, keepdim);
}
inline Tensor dispatch_pdist(const Tensor & self, double p) {

  AutoNoGIL no_gil;
  return at::pdist(self, p);
}
inline Tensor dispatch_pin_memory(const Tensor & self) {

  AutoNoGIL no_gil;
  return self.pin_memory();
}
inline Tensor dispatch_pinverse(const Tensor & self, double rcond) {

  AutoNoGIL no_gil;
  return self.pinverse(rcond);
}
inline Tensor dispatch_pixel_shuffle(const Tensor & self, int64_t upscale_factor) {

  AutoNoGIL no_gil;
  return at::pixel_shuffle(self, upscale_factor);
}
inline Tensor dispatch_poisson(const Tensor & self, Generator * generator) {

  AutoNoGIL no_gil;
  return at::poisson(self, generator);
}
inline Tensor dispatch_polygamma(int64_t n, const Tensor & self, Tensor result) {

  AutoNoGIL no_gil;
  return at::polygamma_out(result, n, self);
}
inline Tensor dispatch_polygamma(int64_t n, const Tensor & self) {

  AutoNoGIL no_gil;
  return self.polygamma(n);
}
inline Tensor dispatch_potri(const Tensor & self, bool upper, Tensor result) {

  AutoNoGIL no_gil;
  return at::potri_out(result, self, upper);
}
inline Tensor dispatch_potri(const Tensor & self, bool upper) {

  AutoNoGIL no_gil;
  return self.potri(upper);
}
inline Tensor dispatch_potrs(const Tensor & self, const Tensor & input2, bool upper, Tensor result) {

  AutoNoGIL no_gil;
  return at::potrs_out(result, self, input2, upper);
}
inline Tensor dispatch_potrs(const Tensor & self, const Tensor & input2, bool upper) {

  AutoNoGIL no_gil;
  return self.potrs(input2, upper);
}
inline Tensor dispatch_pow(const Tensor & self, const Tensor & exponent, Tensor result) {

  AutoNoGIL no_gil;
  return at::pow_out(result, self, exponent);
}
inline Tensor dispatch_pow(const Tensor & self, const Tensor & exponent) {

  AutoNoGIL no_gil;
  return self.pow(exponent);
}
inline Tensor dispatch_pow(Scalar self, const Tensor & exponent, Tensor result) {

  AutoNoGIL no_gil;
  return at::pow_out(result, self, exponent);
}
inline Tensor dispatch_pow(Scalar self, const Tensor & exponent) {

  AutoNoGIL no_gil;
  return at::pow(self, exponent);
}
inline Tensor dispatch_pow(const Tensor & self, Scalar exponent, Tensor result) {

  AutoNoGIL no_gil;
  return at::pow_out(result, self, exponent);
}
inline Tensor dispatch_pow(const Tensor & self, Scalar exponent) {

  AutoNoGIL no_gil;
  return self.pow(exponent);
}
inline Tensor dispatch_prelu(const Tensor & self, const Tensor & weight) {

  AutoNoGIL no_gil;
  return self.prelu(weight);
}
inline Tensor dispatch_prod(const Tensor & self) {

  AutoNoGIL no_gil;
  return self.prod();
}
inline Tensor dispatch_prod(const Tensor & self, ScalarType dtype) {

  AutoNoGIL no_gil;
  return self.prod(dtype);
}
inline Tensor dispatch_prod(const Tensor & self, int64_t dim, ScalarType dtype, Tensor result) {

  AutoNoGIL no_gil;
  return at::prod_out(result, self, dim, dtype);
}
inline Tensor dispatch_prod(const Tensor & self, int64_t dim, ScalarType dtype) {

  AutoNoGIL no_gil;
  return self.prod(dim, dtype);
}
inline Tensor dispatch_prod(const Tensor & self, int64_t dim, bool keepdim, ScalarType dtype, Tensor result) {

  AutoNoGIL no_gil;
  return at::prod_out(result, self, dim, keepdim, dtype);
}
inline Tensor dispatch_prod(const Tensor & self, int64_t dim, bool keepdim, ScalarType dtype) {

  AutoNoGIL no_gil;
  return self.prod(dim, keepdim, dtype);
}
inline Tensor dispatch_prod(const Tensor & self, int64_t dim, bool keepdim, Tensor result) {

  AutoNoGIL no_gil;
  return at::prod_out(result, self, dim, keepdim);
}
inline Tensor dispatch_prod(const Tensor & self, int64_t dim, bool keepdim) {

  AutoNoGIL no_gil;
  return self.prod(dim, keepdim);
}
inline std::tuple<Tensor,Tensor> dispatch_pstrf(const Tensor & self, bool upper, Scalar tol, Tensor & u, Tensor & piv) {

  AutoNoGIL no_gil;
  return at::pstrf_out(u, piv, self, upper, tol);
}
inline std::tuple<Tensor,Tensor> dispatch_pstrf(const Tensor & self, bool upper, Scalar tol) {

  AutoNoGIL no_gil;
  return self.pstrf(upper, tol);
}
inline std::tuple<Tensor,Tensor> dispatch_qr(const Tensor & self, Tensor & Q, Tensor & R) {

  AutoNoGIL no_gil;
  return at::qr_out(Q, R, self);
}
inline std::tuple<Tensor,Tensor> dispatch_qr(const Tensor & self) {

  AutoNoGIL no_gil;
  return self.qr();
}
inline Tensor dispatch_rand(IntList size, Generator * generator, Tensor result) {

  AutoNoGIL no_gil;
  return at::rand_out(result, size, generator);
}
inline Tensor dispatch_rand(IntList size, Generator * generator, const TensorOptions & options) {
  maybe_initialize_cuda(options);
  AutoNoGIL no_gil;
  return torch::rand(size, generator, options);
}
inline Tensor dispatch_rand(IntList size, Tensor result) {

  AutoNoGIL no_gil;
  return at::rand_out(result, size);
}
inline Tensor dispatch_rand(IntList size, const TensorOptions & options) {
  maybe_initialize_cuda(options);
  AutoNoGIL no_gil;
  return torch::rand(size, options);
}
inline Tensor dispatch_rand_like(const Tensor & self, const TensorOptions & options) {
  maybe_initialize_cuda(options);
  AutoNoGIL no_gil;
  return torch::rand_like(self, options);
}
inline Tensor dispatch_rand_like(const Tensor & self) {

  AutoNoGIL no_gil;
  return torch::rand_like(self);
}
inline Tensor dispatch_randint_like(const Tensor & self, int64_t high, const TensorOptions & options) {
  maybe_initialize_cuda(options);
  AutoNoGIL no_gil;
  return torch::randint_like(self, high, options);
}
inline Tensor dispatch_randint_like(const Tensor & self, int64_t high) {

  AutoNoGIL no_gil;
  return torch::randint_like(self, high);
}
inline Tensor dispatch_randint_like(const Tensor & self, int64_t low, int64_t high, const TensorOptions & options) {
  maybe_initialize_cuda(options);
  AutoNoGIL no_gil;
  return torch::randint_like(self, low, high, options);
}
inline Tensor dispatch_randint_like(const Tensor & self, int64_t low, int64_t high) {

  AutoNoGIL no_gil;
  return torch::randint_like(self, low, high);
}
inline Tensor dispatch_randn(IntList size, Generator * generator, Tensor result) {

  AutoNoGIL no_gil;
  return at::randn_out(result, size, generator);
}
inline Tensor dispatch_randn(IntList size, Generator * generator, const TensorOptions & options) {
  maybe_initialize_cuda(options);
  AutoNoGIL no_gil;
  return torch::randn(size, generator, options);
}
inline Tensor dispatch_randn(IntList size, Tensor result) {

  AutoNoGIL no_gil;
  return at::randn_out(result, size);
}
inline Tensor dispatch_randn(IntList size, const TensorOptions & options) {
  maybe_initialize_cuda(options);
  AutoNoGIL no_gil;
  return torch::randn(size, options);
}
inline Tensor dispatch_randn_like(const Tensor & self, const TensorOptions & options) {
  maybe_initialize_cuda(options);
  AutoNoGIL no_gil;
  return torch::randn_like(self, options);
}
inline Tensor dispatch_randn_like(const Tensor & self) {

  AutoNoGIL no_gil;
  return torch::randn_like(self);
}
inline Tensor dispatch_randperm(int64_t n, Generator * generator, Tensor result) {

  AutoNoGIL no_gil;
  return at::randperm_out(result, n, generator);
}
inline Tensor dispatch_randperm(int64_t n, Generator * generator, const TensorOptions & options) {
  maybe_initialize_cuda(options);
  AutoNoGIL no_gil;
  return torch::randperm(n, generator, options);
}
inline Tensor dispatch_randperm(int64_t n, Tensor result) {

  AutoNoGIL no_gil;
  return at::randperm_out(result, n);
}
inline Tensor dispatch_randperm(int64_t n, const TensorOptions & options) {
  maybe_initialize_cuda(options);
  AutoNoGIL no_gil;
  return torch::randperm(n, options);
}
inline Tensor dispatch_reciprocal(const Tensor & self, Tensor result) {

  AutoNoGIL no_gil;
  return at::reciprocal_out(result, self);
}
inline Tensor dispatch_reciprocal(const Tensor & self) {

  AutoNoGIL no_gil;
  return self.reciprocal();
}
inline Tensor dispatch_relu(const Tensor & self) {

  AutoNoGIL no_gil;
  return self.relu();
}
inline Tensor dispatch_relu_(Tensor self) {

  AutoNoGIL no_gil;
  return self.relu_();
}
inline Tensor dispatch_remainder(const Tensor & self, const Tensor & other, Tensor result) {

  AutoNoGIL no_gil;
  return at::remainder_out(result, self, other);
}
inline Tensor dispatch_remainder(const Tensor & self, const Tensor & other) {

  AutoNoGIL no_gil;
  return self.remainder(other);
}
inline Tensor dispatch_remainder(const Tensor & self, Scalar other, Tensor result) {

  AutoNoGIL no_gil;
  return at::remainder_out(result, self, other);
}
inline Tensor dispatch_remainder(const Tensor & self, Scalar other) {

  AutoNoGIL no_gil;
  return self.remainder(other);
}
inline Tensor dispatch_renorm(const Tensor & self, Scalar p, int64_t dim, Scalar maxnorm, Tensor result) {

  AutoNoGIL no_gil;
  return at::renorm_out(result, self, p, dim, maxnorm);
}
inline Tensor dispatch_renorm(const Tensor & self, Scalar p, int64_t dim, Scalar maxnorm) {

  AutoNoGIL no_gil;
  return self.renorm(p, dim, maxnorm);
}
inline Tensor dispatch_reshape(const Tensor & self, IntList shape) {

  AutoNoGIL no_gil;
  return self.reshape(shape);
}
inline Tensor dispatch_resize_as_(Tensor self, const Tensor & the_template) {

  AutoNoGIL no_gil;
  return self.resize_as_(the_template);
}
inline Tensor dispatch_rfft(const Tensor & self, int64_t signal_ndim, bool normalized, bool onesided) {

  AutoNoGIL no_gil;
  return self.rfft(signal_ndim, normalized, onesided);
}
inline std::tuple<Tensor,Tensor> dispatch_rnn_relu(const Tensor & data, const Tensor & batch_sizes, const Tensor & hx, TensorList params, bool has_biases, int64_t num_layers, double dropout, bool train, bool bidirectional) {

  AutoNoGIL no_gil;
  return at::rnn_relu(data, batch_sizes, hx, params, has_biases, num_layers, dropout, train, bidirectional);
}
inline std::tuple<Tensor,Tensor> dispatch_rnn_relu(const Tensor & input, const Tensor & hx, TensorList params, bool has_biases, int64_t num_layers, double dropout, bool train, bool bidirectional, bool batch_first) {

  AutoNoGIL no_gil;
  return at::rnn_relu(input, hx, params, has_biases, num_layers, dropout, train, bidirectional, batch_first);
}
inline Tensor dispatch_rnn_relu_cell(const Tensor & input, const Tensor & hx, const Tensor & w_ih, const Tensor & w_hh, const Tensor & b_ih, const Tensor & b_hh) {

  AutoNoGIL no_gil;
  return at::rnn_relu_cell(input, hx, w_ih, w_hh, b_ih, b_hh);
}
inline std::tuple<Tensor,Tensor> dispatch_rnn_tanh(const Tensor & data, const Tensor & batch_sizes, const Tensor & hx, TensorList params, bool has_biases, int64_t num_layers, double dropout, bool train, bool bidirectional) {

  AutoNoGIL no_gil;
  return at::rnn_tanh(data, batch_sizes, hx, params, has_biases, num_layers, dropout, train, bidirectional);
}
inline std::tuple<Tensor,Tensor> dispatch_rnn_tanh(const Tensor & input, const Tensor & hx, TensorList params, bool has_biases, int64_t num_layers, double dropout, bool train, bool bidirectional, bool batch_first) {

  AutoNoGIL no_gil;
  return at::rnn_tanh(input, hx, params, has_biases, num_layers, dropout, train, bidirectional, batch_first);
}
inline Tensor dispatch_rnn_tanh_cell(const Tensor & input, const Tensor & hx, const Tensor & w_ih, const Tensor & w_hh, const Tensor & b_ih, const Tensor & b_hh) {

  AutoNoGIL no_gil;
  return at::rnn_tanh_cell(input, hx, w_ih, w_hh, b_ih, b_hh);
}
inline Tensor dispatch_roll(const Tensor & self, IntList shifts, IntList dims) {

  AutoNoGIL no_gil;
  return self.roll(shifts, dims);
}
inline Tensor dispatch_rot90(const Tensor & self, int64_t k, IntList dims) {

  AutoNoGIL no_gil;
  return self.rot90(k, dims);
}
inline Tensor dispatch_round(const Tensor & self, Tensor result) {

  AutoNoGIL no_gil;
  return at::round_out(result, self);
}
inline Tensor dispatch_round(const Tensor & self) {

  AutoNoGIL no_gil;
  return self.round();
}
inline Tensor dispatch_round_(Tensor self) {

  AutoNoGIL no_gil;
  return self.round_();
}
inline Tensor dispatch_rrelu(const Tensor & self, Scalar lower, Scalar upper, bool training, Generator * generator) {

  AutoNoGIL no_gil;
  return at::rrelu(self, lower, upper, training, generator);
}
inline Tensor dispatch_rrelu_(Tensor self, Scalar lower, Scalar upper, bool training, Generator * generator) {

  AutoNoGIL no_gil;
  return at::rrelu_(self, lower, upper, training, generator);
}
inline Tensor dispatch_rsqrt(const Tensor & self, Tensor result) {

  AutoNoGIL no_gil;
  return at::rsqrt_out(result, self);
}
inline Tensor dispatch_rsqrt(const Tensor & self) {

  AutoNoGIL no_gil;
  return self.rsqrt();
}
inline Tensor dispatch_rsqrt_(Tensor self) {

  AutoNoGIL no_gil;
  return self.rsqrt_();
}
inline Tensor dispatch_rsub(const Tensor & self, const Tensor & other, Scalar alpha) {

  AutoNoGIL no_gil;
  return at::rsub(self, other, alpha);
}
inline Tensor dispatch_rsub(const Tensor & self, Scalar other, Scalar alpha) {

  AutoNoGIL no_gil;
  return at::rsub(self, other, alpha);
}
inline Tensor dispatch_s_copy_(Tensor self, const Tensor & src, bool non_blocking) {

  AutoNoGIL no_gil;
  return at::s_copy_(self, src, non_blocking);
}
inline Tensor dispatch_s_native_addmm(const Tensor & self, const Tensor & mat1, const Tensor & mat2, Scalar beta, Scalar alpha, Tensor result) {

  AutoNoGIL no_gil;
  return at::s_native_addmm_out(result, self, mat1, mat2, beta, alpha);
}
inline Tensor dispatch_s_native_addmm(const Tensor & self, const Tensor & mat1, const Tensor & mat2, Scalar beta, Scalar alpha) {

  AutoNoGIL no_gil;
  return at::s_native_addmm(self, mat1, mat2, beta, alpha);
}
inline Tensor dispatch_s_native_addmm_(Tensor self, const Tensor & mat1, const Tensor & mat2, Scalar beta, Scalar alpha) {

  AutoNoGIL no_gil;
  return at::s_native_addmm_(self, mat1, mat2, beta, alpha);
}
inline Tensor dispatch_scalar_tensor(Scalar s, const TensorOptions & options) {
  maybe_initialize_cuda(options);
  AutoNoGIL no_gil;
  return torch::scalar_tensor(s, options);
}
inline Tensor dispatch_select(const Tensor & self, int64_t dim, int64_t index) {

  AutoNoGIL no_gil;
  return self.select(dim, index);
}
inline Tensor dispatch_selu(const Tensor & self) {

  AutoNoGIL no_gil;
  return at::selu(self);
}
inline Tensor dispatch_selu_(Tensor self) {

  AutoNoGIL no_gil;
  return at::selu_(self);
}
inline Tensor dispatch_sigmoid(const Tensor & self, Tensor result) {

  AutoNoGIL no_gil;
  return at::sigmoid_out(result, self);
}
inline Tensor dispatch_sigmoid(const Tensor & self) {

  AutoNoGIL no_gil;
  return self.sigmoid();
}
inline Tensor dispatch_sigmoid_(Tensor self) {

  AutoNoGIL no_gil;
  return self.sigmoid_();
}
inline Tensor dispatch_sign(const Tensor & self, Tensor result) {

  AutoNoGIL no_gil;
  return at::sign_out(result, self);
}
inline Tensor dispatch_sign(const Tensor & self) {

  AutoNoGIL no_gil;
  return self.sign();
}
inline Tensor dispatch_sin(const Tensor & self, Tensor result) {

  AutoNoGIL no_gil;
  return at::sin_out(result, self);
}
inline Tensor dispatch_sin(const Tensor & self) {

  AutoNoGIL no_gil;
  return self.sin();
}
inline Tensor dispatch_sin_(Tensor self) {

  AutoNoGIL no_gil;
  return self.sin_();
}
inline Tensor dispatch_sinh(const Tensor & self, Tensor result) {

  AutoNoGIL no_gil;
  return at::sinh_out(result, self);
}
inline Tensor dispatch_sinh(const Tensor & self) {

  AutoNoGIL no_gil;
  return self.sinh();
}
inline Tensor dispatch_sinh_(Tensor self) {

  AutoNoGIL no_gil;
  return self.sinh_();
}
inline std::tuple<Tensor,Tensor> dispatch_slogdet(const Tensor & self) {

  AutoNoGIL no_gil;
  return self.slogdet();
}
inline Tensor dispatch_smm(const Tensor & self, const Tensor & mat2) {

  AutoNoGIL no_gil;
  return self.smm(mat2);
}
inline Tensor dispatch_softmax(const Tensor & self, int64_t dim) {

  AutoNoGIL no_gil;
  return self.softmax(dim);
}
inline Tensor dispatch_softmax(const Tensor & self, int64_t dim, ScalarType dtype) {

  AutoNoGIL no_gil;
  return self.softmax(dim, dtype);
}
inline std::tuple<Tensor,Tensor> dispatch_sort(const Tensor & self, int64_t dim, bool descending, Tensor & values, Tensor & indices) {

  AutoNoGIL no_gil;
  return at::sort_out(values, indices, self, dim, descending);
}
inline std::tuple<Tensor,Tensor> dispatch_sort(const Tensor & self, int64_t dim, bool descending) {

  AutoNoGIL no_gil;
  return self.sort(dim, descending);
}
inline std::vector<Tensor> dispatch_split(const Tensor & self, int64_t split_size, int64_t dim) {

  AutoNoGIL no_gil;
  return self.split(split_size, dim);
}
inline std::vector<Tensor> dispatch_split_with_sizes(const Tensor & self, IntList split_sizes, int64_t dim) {

  AutoNoGIL no_gil;
  return self.split_with_sizes(split_sizes, dim);
}
inline Tensor dispatch_sqrt(const Tensor & self, Tensor result) {

  AutoNoGIL no_gil;
  return at::sqrt_out(result, self);
}
inline Tensor dispatch_sqrt(const Tensor & self) {

  AutoNoGIL no_gil;
  return self.sqrt();
}
inline Tensor dispatch_sqrt_(Tensor self) {

  AutoNoGIL no_gil;
  return self.sqrt_();
}
inline Tensor dispatch_squeeze(const Tensor & self) {

  AutoNoGIL no_gil;
  return self.squeeze();
}
inline Tensor dispatch_squeeze(const Tensor & self, int64_t dim) {

  AutoNoGIL no_gil;
  return self.squeeze(dim);
}
inline Tensor dispatch_sspaddmm(Scalar beta, const Tensor & self, Scalar alpha, const Tensor & mat1, const Tensor & mat2) {

  AutoNoGIL no_gil;
  return self.sspaddmm(mat1, mat2, beta, alpha);
}
inline Tensor dispatch_sspaddmm(Scalar beta, const Tensor & self, const Tensor & mat1, const Tensor & mat2) {

  AutoNoGIL no_gil;
  return self.sspaddmm(mat1, mat2, beta, 1);
}
inline Tensor dispatch_sspaddmm(const Tensor & self, const Tensor & mat1, const Tensor & mat2, Scalar beta, Scalar alpha, Tensor result) {

  AutoNoGIL no_gil;
  return at::sspaddmm_out(result, self, mat1, mat2, beta, alpha);
}
inline Tensor dispatch_sspaddmm(const Tensor & self, const Tensor & mat1, const Tensor & mat2, Scalar beta, Scalar alpha) {

  AutoNoGIL no_gil;
  return self.sspaddmm(mat1, mat2, beta, alpha);
}
inline Tensor dispatch_stack(TensorList tensors, int64_t dim, Tensor result) {

  AutoNoGIL no_gil;
  return at::stack_out(result, tensors, dim);
}
inline Tensor dispatch_stack(TensorList tensors, int64_t dim) {

  AutoNoGIL no_gil;
  return at::stack(tensors, dim);
}
inline Tensor dispatch_std(const Tensor & self, IntList dim, bool unbiased, bool keepdim, Tensor result) {

  AutoNoGIL no_gil;
  return at::std_out(result, self, dim, unbiased, keepdim);
}
inline Tensor dispatch_std(const Tensor & self, IntList dim, bool unbiased, bool keepdim) {

  AutoNoGIL no_gil;
  return self.std(dim, unbiased, keepdim);
}
inline Tensor dispatch_std(const Tensor & self, bool unbiased) {

  AutoNoGIL no_gil;
  return self.std(unbiased);
}
inline Tensor dispatch_stft(const Tensor & self, int64_t n_fft, int64_t hop_length, int64_t win_length, const Tensor & window, bool normalized, bool onesided) {

  AutoNoGIL no_gil;
  return self.stft(n_fft, hop_length, win_length, window, normalized, onesided);
}
inline Tensor dispatch_sub(const Tensor & self, Scalar alpha, const Tensor & other, Tensor out) {

  AutoNoGIL no_gil;
  return at::sub_out(out, self, other, alpha);
}
inline Tensor dispatch_sub(const Tensor & self, Scalar alpha, const Tensor & other) {

  AutoNoGIL no_gil;
  return self.sub(other, alpha);
}
inline Tensor dispatch_sub(const Tensor & self, const Tensor & other, Scalar alpha, Tensor result) {

  AutoNoGIL no_gil;
  return at::sub_out(result, self, other, alpha);
}
inline Tensor dispatch_sub(const Tensor & self, const Tensor & other, Scalar alpha) {

  AutoNoGIL no_gil;
  return self.sub(other, alpha);
}
inline Tensor dispatch_sum(const Tensor & self) {

  AutoNoGIL no_gil;
  return self.sum();
}
inline Tensor dispatch_sum(const Tensor & self, ScalarType dtype) {

  AutoNoGIL no_gil;
  return self.sum(dtype);
}
inline Tensor dispatch_sum(const Tensor & self, IntList dim, ScalarType dtype, Tensor result) {

  AutoNoGIL no_gil;
  return at::sum_out(result, self, dim, dtype);
}
inline Tensor dispatch_sum(const Tensor & self, IntList dim, ScalarType dtype) {

  AutoNoGIL no_gil;
  return self.sum(dim, dtype);
}
inline Tensor dispatch_sum(const Tensor & self, IntList dim, bool keepdim, ScalarType dtype, Tensor result) {

  AutoNoGIL no_gil;
  return at::sum_out(result, self, dim, keepdim, dtype);
}
inline Tensor dispatch_sum(const Tensor & self, IntList dim, bool keepdim, ScalarType dtype) {

  AutoNoGIL no_gil;
  return self.sum(dim, keepdim, dtype);
}
inline Tensor dispatch_sum(const Tensor & self, IntList dim, bool keepdim, Tensor result) {

  AutoNoGIL no_gil;
  return at::sum_out(result, self, dim, keepdim);
}
inline Tensor dispatch_sum(const Tensor & self, IntList dim, bool keepdim) {

  AutoNoGIL no_gil;
  return self.sum(dim, keepdim);
}
inline std::tuple<Tensor,Tensor,Tensor> dispatch_svd(const Tensor & self, bool some, bool compute_uv, Tensor & U, Tensor & S, Tensor & V) {

  AutoNoGIL no_gil;
  return at::svd_out(U, S, V, self, some, compute_uv);
}
inline std::tuple<Tensor,Tensor,Tensor> dispatch_svd(const Tensor & self, bool some, bool compute_uv) {

  AutoNoGIL no_gil;
  return self.svd(some, compute_uv);
}
inline std::tuple<Tensor,Tensor> dispatch_symeig(const Tensor & self, bool eigenvectors, bool upper, Tensor & e, Tensor & V) {

  AutoNoGIL no_gil;
  return at::symeig_out(e, V, self, eigenvectors, upper);
}
inline std::tuple<Tensor,Tensor> dispatch_symeig(const Tensor & self, bool eigenvectors, bool upper) {

  AutoNoGIL no_gil;
  return self.symeig(eigenvectors, upper);
}
inline Tensor dispatch_t(const Tensor & self) {

  AutoNoGIL no_gil;
  return self.t();
}
inline Tensor dispatch_take(const Tensor & self, const Tensor & index, Tensor result) {

  AutoNoGIL no_gil;
  return at::take_out(result, self, index);
}
inline Tensor dispatch_take(const Tensor & self, const Tensor & index) {

  AutoNoGIL no_gil;
  return self.take(index);
}
inline Tensor dispatch_tan(const Tensor & self, Tensor result) {

  AutoNoGIL no_gil;
  return at::tan_out(result, self);
}
inline Tensor dispatch_tan(const Tensor & self) {

  AutoNoGIL no_gil;
  return self.tan();
}
inline Tensor dispatch_tan_(Tensor self) {

  AutoNoGIL no_gil;
  return self.tan_();
}
inline Tensor dispatch_tanh(const Tensor & self, Tensor result) {

  AutoNoGIL no_gil;
  return at::tanh_out(result, self);
}
inline Tensor dispatch_tanh(const Tensor & self) {

  AutoNoGIL no_gil;
  return self.tanh();
}
inline Tensor dispatch_tanh_(Tensor self) {

  AutoNoGIL no_gil;
  return self.tanh_();
}
inline Tensor dispatch_tensordot(const Tensor & self, const Tensor & other, IntList dims_self, IntList dims_other) {

  AutoNoGIL no_gil;
  return at::tensordot(self, other, dims_self, dims_other);
}
inline Tensor dispatch_threshold(const Tensor & self, Scalar threshold, Scalar value, Tensor result) {

  AutoNoGIL no_gil;
  return at::threshold_out(result, self, threshold, value);
}
inline Tensor dispatch_threshold(const Tensor & self, Scalar threshold, Scalar value) {

  AutoNoGIL no_gil;
  return at::threshold(self, threshold, value);
}
inline Tensor dispatch_threshold_(Tensor self, Scalar threshold, Scalar value) {

  AutoNoGIL no_gil;
  return at::threshold_(self, threshold, value);
}
inline std::tuple<Tensor,Tensor> dispatch_topk(const Tensor & self, int64_t k, int64_t dim, bool largest, bool sorted, Tensor & values, Tensor & indices) {

  AutoNoGIL no_gil;
  return at::topk_out(values, indices, self, k, dim, largest, sorted);
}
inline std::tuple<Tensor,Tensor> dispatch_topk(const Tensor & self, int64_t k, int64_t dim, bool largest, bool sorted) {

  AutoNoGIL no_gil;
  return self.topk(k, dim, largest, sorted);
}
inline Tensor dispatch_trace(const Tensor & self) {

  AutoNoGIL no_gil;
  return self.trace();
}
inline Tensor dispatch_transpose(const Tensor & self, int64_t dim0, int64_t dim1) {

  AutoNoGIL no_gil;
  return self.transpose(dim0, dim1);
}
inline Tensor dispatch_tril(const Tensor & self, int64_t diagonal, Tensor result) {

  AutoNoGIL no_gil;
  return at::tril_out(result, self, diagonal);
}
inline Tensor dispatch_tril(const Tensor & self, int64_t diagonal) {

  AutoNoGIL no_gil;
  return self.tril(diagonal);
}
inline Tensor dispatch_tril_indices(int64_t row, int64_t col, int64_t offset, const TensorOptions & options) {
  maybe_initialize_cuda(options);
  AutoNoGIL no_gil;
  return torch::tril_indices(row, col, offset, options);
}
inline Tensor dispatch_triplet_margin_loss(const Tensor & anchor, const Tensor & positive, const Tensor & negative, double margin, double p, double eps, bool swap, int64_t reduction) {

  AutoNoGIL no_gil;
  return at::triplet_margin_loss(anchor, positive, negative, margin, p, eps, swap, reduction);
}
inline Tensor dispatch_triu(const Tensor & self, int64_t diagonal, Tensor result) {

  AutoNoGIL no_gil;
  return at::triu_out(result, self, diagonal);
}
inline Tensor dispatch_triu(const Tensor & self, int64_t diagonal) {

  AutoNoGIL no_gil;
  return self.triu(diagonal);
}
inline Tensor dispatch_triu_indices(int64_t row, int64_t col, int64_t offset, const TensorOptions & options) {
  maybe_initialize_cuda(options);
  AutoNoGIL no_gil;
  return torch::triu_indices(row, col, offset, options);
}
inline std::tuple<Tensor,Tensor> dispatch_trtrs(const Tensor & self, const Tensor & A, bool upper, bool transpose, bool unitriangular, Tensor & X, Tensor & M) {

  AutoNoGIL no_gil;
  return at::trtrs_out(X, M, self, A, upper, transpose, unitriangular);
}
inline std::tuple<Tensor,Tensor> dispatch_trtrs(const Tensor & self, const Tensor & A, bool upper, bool transpose, bool unitriangular) {

  AutoNoGIL no_gil;
  return self.trtrs(A, upper, transpose, unitriangular);
}
inline Tensor dispatch_trunc(const Tensor & self, Tensor result) {

  AutoNoGIL no_gil;
  return at::trunc_out(result, self);
}
inline Tensor dispatch_trunc(const Tensor & self) {

  AutoNoGIL no_gil;
  return self.trunc();
}
inline Tensor dispatch_trunc_(Tensor self) {

  AutoNoGIL no_gil;
  return self.trunc_();
}
inline std::vector<Tensor> dispatch_unbind(const Tensor & self, int64_t dim) {

  AutoNoGIL no_gil;
  return self.unbind(dim);
}
inline Tensor dispatch_unsqueeze(const Tensor & self, int64_t dim) {

  AutoNoGIL no_gil;
  return self.unsqueeze(dim);
}
inline Tensor dispatch_var(const Tensor & self, bool unbiased) {

  AutoNoGIL no_gil;
  return self.var(unbiased);
}
inline Tensor dispatch_var(const Tensor & self, int64_t dim, bool unbiased, bool keepdim, Tensor result) {

  AutoNoGIL no_gil;
  return at::var_out(result, self, dim, unbiased, keepdim);
}
inline Tensor dispatch_var(const Tensor & self, int64_t dim, bool unbiased, bool keepdim) {

  AutoNoGIL no_gil;
  return self.var(dim, unbiased, keepdim);
}
inline Tensor dispatch_where(const Tensor & condition, const Tensor & self, const Tensor & other) {

  AutoNoGIL no_gil;
  return self.where(condition, other);
}
inline Tensor dispatch_zero_(Tensor self) {

  AutoNoGIL no_gil;
  return self.zero_();
}
inline Tensor dispatch_zeros(IntList size, Tensor result) {

  AutoNoGIL no_gil;
  return at::zeros_out(result, size);
}
inline Tensor dispatch_zeros(IntList size, const TensorOptions & options) {
  maybe_initialize_cuda(options);
  AutoNoGIL no_gil;
  return torch::zeros(size, options);
}
inline Tensor dispatch_zeros_like(const Tensor & self, const TensorOptions & options) {
  maybe_initialize_cuda(options);
  AutoNoGIL no_gil;
  return torch::zeros_like(self, options);
}
inline Tensor dispatch_zeros_like(const Tensor & self) {

  AutoNoGIL no_gil;
  return torch::zeros_like(self);
}

}} // namespace torch::autograd
