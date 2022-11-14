#pragma once

#include <ATen/Tensor.h>
#include <c10/core/ScalarType.h>
#include <c10/core/SymInt.h>
#include <c10/core/SymIntArrayRef.h>
#include <c10/core/SymNodeImpl.h>
#include <c10/macros/Export.h>
#include <c10/util/Optional.h>
#include <torch/csrc/lazy/backend/backend_data.h>
#include <torch/csrc/lazy/core/ir.h>
#include <torch/csrc/lazy/core/shape.h>
#include <torch/csrc/lazy/core/tensor.h>
#include <vector>

namespace torch {
namespace lazy {
// Turn clang-format off, as we rely on the whole signature being on one line
// for codegen.
// clang-format off
TORCH_API std::vector<torch::lazy::Shape> compute_shape__adaptive_avg_pool2d(const at::Tensor & self, at::IntArrayRef output_size);
TORCH_API std::vector<torch::lazy::Shape> compute_shape__adaptive_avg_pool2d_backward(const at::Tensor & grad_output, const at::Tensor & self);
TORCH_API std::vector<torch::lazy::Shape> compute_shape__adaptive_avg_pool3d(const at::Tensor & self, at::IntArrayRef output_size);
TORCH_API std::vector<torch::lazy::Shape> compute_shape__adaptive_avg_pool3d_backward(const at::Tensor & grad_output, const at::Tensor & self);
TORCH_API std::vector<torch::lazy::Shape> compute_shape_abs(const at::Tensor & self);
TORCH_API std::vector<torch::lazy::Shape> compute_shape_arange_out(const at::Scalar & start, const at::Scalar & end, const at::Scalar & step, at::Tensor & out);
TORCH_API std::vector<torch::lazy::Shape> compute_shape_bernoulli(const at::Tensor & self, c10::optional<at::Generator> generator);
TORCH_API std::vector<torch::lazy::Shape> compute_shape_bernoulli(const at::Tensor & self, double p, c10::optional<at::Generator> generator);
TORCH_API std::vector<torch::lazy::Shape> compute_shape_binary_cross_entropy(const at::Tensor & self, const at::Tensor & target, const c10::optional<at::Tensor> & weight, int64_t reduction);
TORCH_API std::vector<torch::lazy::Shape> compute_shape_binary_cross_entropy_backward(const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & target, const c10::optional<at::Tensor> & weight, int64_t reduction);
TORCH_API std::vector<torch::lazy::Shape> compute_shape_cat(at::TensorList tensors, int64_t dim);
TORCH_API std::vector<torch::lazy::Shape> compute_shape_cholesky(const at::Tensor & self, bool upper);
TORCH_API std::vector<torch::lazy::Shape> compute_shape_clamp_min(const at::Tensor & self, const at::Scalar & min);
TORCH_API std::vector<torch::lazy::Shape> compute_shape_clone(const at::Tensor & self, c10::optional<at::MemoryFormat> memory_format);
TORCH_API std::vector<torch::lazy::Shape> compute_shape_constant_pad_nd(const at::Tensor & self, at::IntArrayRef pad, const at::Scalar & value);
TORCH_API std::vector<torch::lazy::Shape> compute_shape_convolution(const at::Tensor & input, const at::Tensor & weight, const c10::optional<at::Tensor> & bias, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, bool transposed, at::IntArrayRef output_padding, int64_t groups);
TORCH_API std::vector<torch::lazy::Shape> compute_shape_convolution_backward(const at::Tensor & grad_output, const at::Tensor & input, const at::Tensor & weight, at::OptionalIntArrayRef bias_sizes, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, bool transposed, at::IntArrayRef output_padding, int64_t groups, ::std::array<bool,3> output_mask);
TORCH_API std::vector<torch::lazy::Shape> compute_shape_embedding(const at::Tensor & weight, const at::Tensor & indices, int64_t padding_idx, bool scale_grad_by_freq, bool sparse);
TORCH_API std::vector<torch::lazy::Shape> compute_shape_embedding_dense_backward(const at::Tensor & grad_output, const at::Tensor & indices, int64_t num_weights, int64_t padding_idx, bool scale_grad_by_freq);
TORCH_API std::vector<torch::lazy::Shape> compute_shape_expand(const at::Tensor & self, at::IntArrayRef size, bool implicit);
TORCH_API std::vector<torch::lazy::Shape> compute_shape_expand(const at::Tensor & self, c10::SymIntArrayRef size, bool implicit);
TORCH_API std::vector<torch::lazy::Shape> compute_shape_flip(const at::Tensor & self, at::IntArrayRef dims);
TORCH_API std::vector<torch::lazy::Shape> compute_shape_glu_backward(const at::Tensor & grad_output, const at::Tensor & self, int64_t dim);
TORCH_API std::vector<torch::lazy::Shape> compute_shape_glu_jvp(const at::Tensor & glu, const at::Tensor & x, const at::Tensor & dx, int64_t dim);
TORCH_API std::vector<torch::lazy::Shape> compute_shape_grid_sampler_2d(const at::Tensor & input, const at::Tensor & grid, int64_t interpolation_mode, int64_t padding_mode, bool align_corners);
TORCH_API std::vector<torch::lazy::Shape> compute_shape_grid_sampler_2d_backward(const at::Tensor & grad_output, const at::Tensor & input, const at::Tensor & grid, int64_t interpolation_mode, int64_t padding_mode, bool align_corners, ::std::array<bool,2> output_mask);
TORCH_API std::vector<torch::lazy::Shape> compute_shape_index_select(const at::Tensor & self, int64_t dim, const at::Tensor & index);
TORCH_API std::vector<torch::lazy::Shape> compute_shape_inverse(const at::Tensor & self);
TORCH_API std::vector<torch::lazy::Shape> compute_shape_isnan(const at::Tensor & self);
TORCH_API std::vector<torch::lazy::Shape> compute_shape_log_sigmoid_backward(const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & buffer);
TORCH_API std::vector<torch::lazy::Shape> compute_shape_log_sigmoid_forward(const at::Tensor & self);
TORCH_API std::vector<torch::lazy::Shape> compute_shape_logdet(const at::Tensor & self);
TORCH_API std::vector<torch::lazy::Shape> compute_shape_logical_and(const at::Tensor & self, const at::Tensor & other);
TORCH_API std::vector<torch::lazy::Shape> compute_shape_logical_not(const at::Tensor & self);
TORCH_API std::vector<torch::lazy::Shape> compute_shape_logical_or(const at::Tensor & self, const at::Tensor & other);
TORCH_API std::vector<torch::lazy::Shape> compute_shape_logical_xor(const at::Tensor & self, const at::Tensor & other);
TORCH_API std::vector<torch::lazy::Shape> compute_shape_masked_fill(const at::Tensor & self, const at::Tensor & mask, const at::Scalar & value);
TORCH_API std::vector<torch::lazy::Shape> compute_shape_masked_fill(const at::Tensor & self, const at::Tensor & mask, const at::Tensor & value);
TORCH_API std::vector<torch::lazy::Shape> compute_shape_max(const at::Tensor & self);
TORCH_API std::vector<torch::lazy::Shape> compute_shape_mean(const at::Tensor & self, c10::optional<at::ScalarType> dtype);
TORCH_API std::vector<torch::lazy::Shape> compute_shape_min(const at::Tensor & self);
TORCH_API std::vector<torch::lazy::Shape> compute_shape_mv(const at::Tensor & self, const at::Tensor & vec);
TORCH_API std::vector<torch::lazy::Shape> compute_shape_native_batch_norm(const at::Tensor & input, const c10::optional<at::Tensor> & weight, const c10::optional<at::Tensor> & bias, const c10::optional<at::Tensor> & running_mean, const c10::optional<at::Tensor> & running_var, bool training, double momentum, double eps);
TORCH_API std::vector<torch::lazy::Shape> compute_shape_native_batch_norm_backward(const at::Tensor & grad_out, const at::Tensor & input, const c10::optional<at::Tensor> & weight, const c10::optional<at::Tensor> & running_mean, const c10::optional<at::Tensor> & running_var, const c10::optional<at::Tensor> & save_mean, const c10::optional<at::Tensor> & save_invstd, bool train, double eps, ::std::array<bool,3> output_mask);
TORCH_API std::vector<torch::lazy::Shape> compute_shape_native_dropout(const at::Tensor & input, double p, c10::optional<bool> train);
TORCH_API std::vector<torch::lazy::Shape> compute_shape_native_dropout_backward(const at::Tensor & grad_output, const at::Tensor & mask, double scale);
TORCH_API std::vector<torch::lazy::Shape> compute_shape_native_layer_norm(const at::Tensor & input, at::IntArrayRef normalized_shape, const c10::optional<at::Tensor> & weight, const c10::optional<at::Tensor> & bias, double eps);
TORCH_API std::vector<torch::lazy::Shape> compute_shape_native_layer_norm_backward(const at::Tensor & grad_out, const at::Tensor & input, at::IntArrayRef normalized_shape, const at::Tensor & mean, const at::Tensor & rstd, const c10::optional<at::Tensor> & weight, const c10::optional<at::Tensor> & bias, ::std::array<bool,3> output_mask);
TORCH_API std::vector<torch::lazy::Shape> compute_shape_new_empty_strided(const at::Tensor & self, at::IntArrayRef size, at::IntArrayRef stride, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory);
TORCH_API std::vector<torch::lazy::Shape> compute_shape_nll_loss2d_backward(const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & target, const c10::optional<at::Tensor> & weight, int64_t reduction, int64_t ignore_index, const at::Tensor & total_weight);
TORCH_API std::vector<torch::lazy::Shape> compute_shape_nll_loss2d_forward(const at::Tensor & self, const at::Tensor & target, const c10::optional<at::Tensor> & weight, int64_t reduction, int64_t ignore_index);
TORCH_API std::vector<torch::lazy::Shape> compute_shape_nonzero(const at::Tensor & self);
TORCH_API std::vector<torch::lazy::Shape> compute_shape_random(const at::Tensor & self, c10::optional<at::Generator> generator);
TORCH_API std::vector<torch::lazy::Shape> compute_shape_random(const at::Tensor & self, int64_t to, c10::optional<at::Generator> generator);
TORCH_API std::vector<torch::lazy::Shape> compute_shape_random(const at::Tensor & self, int64_t from, c10::optional<int64_t> to, c10::optional<at::Generator> generator);
TORCH_API std::vector<torch::lazy::Shape> compute_shape_relu(const at::Tensor & self);
TORCH_API std::vector<torch::lazy::Shape> compute_shape_repeat(const at::Tensor & self, at::IntArrayRef repeats);
TORCH_API std::vector<torch::lazy::Shape> compute_shape_slogdet(const at::Tensor & self);
TORCH_API std::vector<torch::lazy::Shape> compute_shape_smooth_l1_loss_backward(const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & target, int64_t reduction, double beta);
TORCH_API std::vector<torch::lazy::Shape> compute_shape_sort(const at::Tensor & self, int64_t dim, bool descending);
TORCH_API std::vector<torch::lazy::Shape> compute_shape_stack(at::TensorList tensors, int64_t dim);
TORCH_API std::vector<torch::lazy::Shape> compute_shape_std(const at::Tensor & self, bool unbiased);
TORCH_API std::vector<torch::lazy::Shape> compute_shape_std(const at::Tensor & self, at::OptionalIntArrayRef dim, bool unbiased, bool keepdim);
TORCH_API std::vector<torch::lazy::Shape> compute_shape_std(const at::Tensor & self, at::OptionalIntArrayRef dim, c10::optional<int64_t> correction, bool keepdim);
TORCH_API std::vector<torch::lazy::Shape> compute_shape_sum(const at::Tensor & self, c10::optional<at::ScalarType> dtype);
TORCH_API std::vector<torch::lazy::Shape> compute_shape__to_copy(const at::Tensor & self, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory, bool non_blocking, c10::optional<at::MemoryFormat> memory_format);
TORCH_API std::vector<torch::lazy::Shape> compute_shape_take(const at::Tensor & self, const at::Tensor & index);
TORCH_API std::vector<torch::lazy::Shape> compute_shape_trace(const at::Tensor & self);
TORCH_API std::vector<torch::lazy::Shape> compute_shape_zero(const at::Tensor & self);
TORCH_API std::vector<torch::lazy::Shape> compute_shape_narrow_copy_symint(const at::Tensor & self, int64_t dim, int64_t start, c10::SymInt length);
TORCH_API std::vector<torch::lazy::Shape> compute_shape_hardswish(const at::Tensor & self);
TORCH_API std::vector<torch::lazy::Shape> compute_shape_hardswish_backward(const at::Tensor & grad_output, const at::Tensor & self);
TORCH_API std::vector<torch::lazy::Shape> compute_shape_selu(const at::Tensor & self);

// Non-Native ops
TORCH_API std::vector<Shape> compute_shape_scalar(const at::Scalar& value, const at::ScalarType& type);
TORCH_API std::vector<Shape> compute_shape_expand(const Output& input0, const std::vector<int64_t>& size, const bool& is_scalar_expand);
TORCH_API std::vector<Shape> compute_shape_view(const Output& input0, const std::vector<int64_t>& output_sizes);
TORCH_API std::vector<Shape> compute_shape_cast(const Output& input0, const at::ScalarType& dtype, const c10::optional<at::ScalarType>& stype);

// View Ops
// (Now that functionalization pass is used, we should kill these in a later PR)
TORCH_API std::vector<Shape> compute_shape_as_strided_view_update(const Output& target, const Output& input, const std::vector<int64_t>& size, const std::vector<int64_t>& stride, const int64_t& storage_offset);
TORCH_API std::vector<Shape> compute_shape_as_strided(const Output& input, const std::vector<int64_t>& size, const std::vector<int64_t>& stride, const int64_t& storage_offset);
TORCH_API std::vector<Shape> compute_shape_diagonal_view_update(const Output& target, const Output& input, const int64_t& offset, const int64_t& dim1, const int64_t& dim2);
TORCH_API std::vector<Shape> compute_shape_diagonal(const Output& input, const int64_t& offset, const int64_t& dim1, const int64_t& dim2);
TORCH_API std::vector<Shape> compute_shape_narrow_view_update(const Output& input, const Output& source, const std::vector<int64_t>& base_indices);
TORCH_API std::vector<Shape> compute_shape_narrow(const Output& input, const std::vector<int64_t>& base_indices, const std::vector<int64_t>& sizes);
TORCH_API std::vector<Shape> compute_shape_permute(const Output& input, const std::vector<int64_t>& dims);
TORCH_API std::vector<Shape> compute_shape_resize(const Output& input, const std::vector<int64_t>& size);
TORCH_API std::vector<Shape> compute_shape_select_view_update(const Output& target, const Output& source, const int64_t& dim, const int64_t& start, const int64_t& end, const int64_t& stride);
TORCH_API std::vector<Shape> compute_shape_select(const Output& input, const int64_t& dim, const int64_t& start, const int64_t& end, const int64_t& stride);
TORCH_API std::vector<Shape> compute_shape_squeeze(const Output& input, const int& dim);
TORCH_API std::vector<Shape> compute_shape_unsqueeze(const Output& input, const int& dim);

TORCH_API std::vector<torch::lazy::Shape> compute_shape_select_scatter(const at::Tensor & self, const at::Tensor & src, int64_t dim, int64_t index);
TORCH_API std::vector<torch::lazy::Shape> compute_shape_diagonal_scatter(const at::Tensor & self, const at::Tensor & src, int64_t offset, int64_t dim1, int64_t dim2);
TORCH_API std::vector<torch::lazy::Shape> compute_shape_slice_scatter_symint(const at::Tensor & self, const at::Tensor & src, int64_t dim, c10::optional<c10::SymInt> start, c10::optional<c10::SymInt> end, c10::SymInt step);
TORCH_API std::vector<torch::lazy::Shape> compute_shape_as_strided_scatter_symint(const at::Tensor & self, const at::Tensor & src, c10::SymIntArrayRef size, c10::SymIntArrayRef stride, c10::optional<c10::SymInt> storage_offset);
// clang-format on
} // namespace lazy
} // namespace torch
