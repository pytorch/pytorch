#pragma once

#include <ATen/Tensor.h>
#include <c10/core/ScalarType.h>
#include <c10/macros/Export.h>
#include <c10/util/Optional.h>
#include <torch/csrc/lazy/core/ir.h>
#include <torch/csrc/lazy/core/shape.h>
#include <vector>

namespace torch{
namespace lazy {

TORCH_API std::vector<Shape> compute_shape__adaptive_avg_pool2d(const at::Tensor & self, at::IntArrayRef output_size);
TORCH_API std::vector<Shape> compute_shape__adaptive_avg_pool2d_backward(const at::Tensor & grad_output, const at::Tensor & self);
TORCH_API std::vector<Shape> compute_shape_abs(const at::Tensor & self);
TORCH_API std::vector<Shape> compute_shape_arange_out(const at::Scalar & start, const at::Scalar & end, const at::Scalar & step, at::Tensor & out);
TORCH_API std::vector<Shape> compute_shape_binary_cross_entropy(const at::Tensor & self, const at::Tensor & target, const c10::optional<at::Tensor> & weight, int64_t reduction);
TORCH_API std::vector<Shape> compute_shape_binary_cross_entropy_backward(const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & target, const c10::optional<at::Tensor> & weight, int64_t reduction);
TORCH_API std::vector<Shape> compute_shape_cat(at::TensorList tensors, int64_t dim);
TORCH_API std::vector<Shape> compute_shape_clamp_min(const at::Tensor & self, const at::Scalar & min);
TORCH_API std::vector<Shape> compute_shape_constant_pad_nd(const at::Tensor & self, at::IntArrayRef pad, const at::Scalar & value);
TORCH_API std::vector<Shape> compute_shape_convolution(const at::Tensor & input, const at::Tensor & weight, const c10::optional<at::Tensor> & bias, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, bool transposed, at::IntArrayRef output_padding, int64_t groups);
TORCH_API std::vector<Shape> compute_shape_convolution_backward(const at::Tensor & grad_output, const at::Tensor & input, const at::Tensor & weight, c10::optional<at::IntArrayRef> bias_sizes, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, bool transposed, at::IntArrayRef output_padding, int64_t groups, ::std::array<bool,3> output_mask);
TORCH_API std::vector<Shape> compute_shape_embedding(const at::Tensor & weight, const at::Tensor & indices, int64_t padding_idx, bool scale_grad_by_freq, bool sparse);
TORCH_API std::vector<Shape> compute_shape_embedding_dense_backward(const at::Tensor & grad_output, const at::Tensor & indices, int64_t num_weights, int64_t padding_idx, bool scale_grad_by_freq);
TORCH_API std::vector<Shape> compute_shape_flip(const at::Tensor & self, at::IntArrayRef dims);
TORCH_API std::vector<Shape> compute_shape_glu_backward(const at::Tensor & grad_output, const at::Tensor & self, int64_t dim);
TORCH_API std::vector<Shape> compute_shape_grid_sampler_2d(const at::Tensor & input, const at::Tensor & grid, int64_t interpolation_mode, int64_t padding_mode, bool align_corners);
TORCH_API std::vector<Shape> compute_shape_grid_sampler_2d_backward(const at::Tensor & grad_output, const at::Tensor & input, const at::Tensor & grid, int64_t interpolation_mode, int64_t padding_mode, bool align_corners, ::std::array<bool,2> output_mask);
TORCH_API std::vector<Shape> compute_shape_index_select(const at::Tensor & self, int64_t dim, const at::Tensor & index);
TORCH_API std::vector<Shape> compute_shape_kl_div_backward(const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & target, int64_t reduction, bool log_target);
TORCH_API std::vector<Shape> compute_shape_l1_loss_backward(const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & target, int64_t reduction);
TORCH_API std::vector<Shape> compute_shape_log_sigmoid_backward(const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & buffer);
TORCH_API std::vector<Shape> compute_shape_log_sigmoid_forward(const at::Tensor & self);
TORCH_API std::vector<Shape> compute_shape_logdet(const at::Tensor & self);
TORCH_API std::vector<Shape> compute_shape_masked_fill_(at::Tensor & self, const at::Tensor & mask, const at::Scalar & value);
TORCH_API std::vector<Shape> compute_shape_masked_fill_(at::Tensor & self, const at::Tensor & mask, const at::Tensor & value);
TORCH_API std::vector<Shape> compute_shape_max(const at::Tensor & self);
TORCH_API std::vector<Shape> compute_shape_mean(const at::Tensor & self, c10::optional<at::ScalarType> dtype);
TORCH_API std::vector<Shape> compute_shape_min(const at::Tensor & self);
TORCH_API std::vector<Shape> compute_shape_mv(const at::Tensor & self, const at::Tensor & vec);
TORCH_API std::vector<Shape> compute_shape_native_dropout(const at::Tensor & input, double p, c10::optional<bool> train);
TORCH_API std::vector<Shape> compute_shape_native_dropout_backward(const at::Tensor & grad_output, const at::Tensor & mask, double scale);
TORCH_API std::vector<Shape> compute_shape_native_layer_norm(const at::Tensor & input, at::IntArrayRef normalized_shape, const c10::optional<at::Tensor> & weight, const c10::optional<at::Tensor> & bias, double eps);
TORCH_API std::vector<Shape> compute_shape_native_layer_norm_backward(const at::Tensor & grad_out, const at::Tensor & input, at::IntArrayRef normalized_shape, const at::Tensor & mean, const at::Tensor & rstd, const c10::optional<at::Tensor> & weight, const c10::optional<at::Tensor> & bias, ::std::array<bool,3> output_mask);
TORCH_API std::vector<Shape> compute_shape_nll_loss2d_backward(const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & target, const c10::optional<at::Tensor> & weight, int64_t reduction, int64_t ignore_index, const at::Tensor & total_weight);
TORCH_API std::vector<Shape> compute_shape_nll_loss2d_forward(const at::Tensor & self, const at::Tensor & target, const c10::optional<at::Tensor> & weight, int64_t reduction, int64_t ignore_index);
TORCH_API std::vector<Shape> compute_shape_relu(const at::Tensor & self);
TORCH_API std::vector<Shape> compute_shape_relu_(at::Tensor & self);
TORCH_API std::vector<Shape> compute_shape_repeat(const at::Tensor & self, at::IntArrayRef repeats);
TORCH_API std::vector<Shape> compute_shape_smooth_l1_loss_backward(const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & target, int64_t reduction, double beta);
TORCH_API std::vector<Shape> compute_shape_sort(const at::Tensor & self, int64_t dim, bool descending);
TORCH_API std::vector<Shape> compute_shape_std(const at::Tensor & self, bool unbiased);
TORCH_API std::vector<Shape> compute_shape_std(const at::Tensor & self, at::IntArrayRef dim, bool unbiased, bool keepdim);
TORCH_API std::vector<Shape> compute_shape_std(const at::Tensor & self, c10::optional<at::IntArrayRef> dim, c10::optional<int64_t> correction, bool keepdim);
TORCH_API std::vector<Shape> compute_shape_sum(const at::Tensor & self, c10::optional<at::ScalarType> dtype);
TORCH_API std::vector<Shape> compute_shape__to_copy(const at::Tensor & self, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory, bool non_blocking, c10::optional<at::MemoryFormat> memory_format);
TORCH_API std::vector<Shape> compute_shape_trace(const at::Tensor & self);
TORCH_API std::vector<Shape> compute_shape_zero_(at::Tensor & self);

} // namespace lazy
} // namespace torch
