#pragma once

// @generated from tools/autograd/templates/python_nn_functions_dispatch.h

#include "torch/csrc/utils/auto_gil.h"

#include <ATen/ATen.h>

// Contains inline wrappers around ATen functions that release the GIL and
// switch to the correct CUDA device.

namespace torch { namespace autograd {

using namespace at;
using at::Generator;

inline Tensor dispatch_adaptive_avg_pool2d(const Tensor & self, IntList output_size, Tensor output) {

  AutoNoGIL no_gil;
  return at::adaptive_avg_pool2d_out(output, self, output_size);
}
inline Tensor dispatch_adaptive_avg_pool2d(const Tensor & self, IntList output_size) {

  AutoNoGIL no_gil;
  return at::adaptive_avg_pool2d(self, output_size);
}
inline Tensor dispatch_adaptive_avg_pool3d(const Tensor & self, IntList output_size, Tensor output) {

  AutoNoGIL no_gil;
  return at::adaptive_avg_pool3d_out(output, self, output_size);
}
inline Tensor dispatch_adaptive_avg_pool3d(const Tensor & self, IntList output_size) {

  AutoNoGIL no_gil;
  return at::adaptive_avg_pool3d(self, output_size);
}
inline std::tuple<Tensor,Tensor> dispatch_adaptive_max_pool2d(const Tensor & self, IntList output_size, Tensor & output, Tensor & indices) {

  AutoNoGIL no_gil;
  return at::adaptive_max_pool2d_out(output, indices, self, output_size);
}
inline std::tuple<Tensor,Tensor> dispatch_adaptive_max_pool2d(const Tensor & self, IntList output_size) {

  AutoNoGIL no_gil;
  return at::adaptive_max_pool2d(self, output_size);
}
inline std::tuple<Tensor,Tensor> dispatch_adaptive_max_pool3d(const Tensor & self, IntList output_size, Tensor & output, Tensor & indices) {

  AutoNoGIL no_gil;
  return at::adaptive_max_pool3d_out(output, indices, self, output_size);
}
inline std::tuple<Tensor,Tensor> dispatch_adaptive_max_pool3d(const Tensor & self, IntList output_size) {

  AutoNoGIL no_gil;
  return at::adaptive_max_pool3d(self, output_size);
}
inline Tensor dispatch_avg_pool2d(const Tensor & self, IntList kernel_size, IntList stride, IntList padding, bool ceil_mode, bool count_include_pad, Tensor output) {

  AutoNoGIL no_gil;
  return at::avg_pool2d_out(output, self, kernel_size, stride, padding, ceil_mode, count_include_pad);
}
inline Tensor dispatch_avg_pool2d(const Tensor & self, IntList kernel_size, IntList stride, IntList padding, bool ceil_mode, bool count_include_pad) {

  AutoNoGIL no_gil;
  return at::avg_pool2d(self, kernel_size, stride, padding, ceil_mode, count_include_pad);
}
inline Tensor dispatch_avg_pool3d(const Tensor & self, IntList kernel_size, IntList stride, IntList padding, bool ceil_mode, bool count_include_pad, Tensor output) {

  AutoNoGIL no_gil;
  return at::avg_pool3d_out(output, self, kernel_size, stride, padding, ceil_mode, count_include_pad);
}
inline Tensor dispatch_avg_pool3d(const Tensor & self, IntList kernel_size, IntList stride, IntList padding, bool ceil_mode, bool count_include_pad) {

  AutoNoGIL no_gil;
  return at::avg_pool3d(self, kernel_size, stride, padding, ceil_mode, count_include_pad);
}
inline Tensor dispatch_binary_cross_entropy(const Tensor & self, const Tensor & target, const Tensor & weight, int64_t reduction, Tensor output) {

  AutoNoGIL no_gil;
  return at::binary_cross_entropy_out(output, self, target, weight, reduction);
}
inline Tensor dispatch_binary_cross_entropy(const Tensor & self, const Tensor & target, const Tensor & weight, int64_t reduction) {

  AutoNoGIL no_gil;
  return at::binary_cross_entropy(self, target, weight, reduction);
}
inline Tensor dispatch_elu(const Tensor & self, Scalar alpha, Scalar scale, Scalar input_scale, Tensor output) {

  AutoNoGIL no_gil;
  return at::elu_out(output, self, alpha, scale, input_scale);
}
inline Tensor dispatch_elu(const Tensor & self, Scalar alpha, Scalar scale, Scalar input_scale) {

  AutoNoGIL no_gil;
  return at::elu(self, alpha, scale, input_scale);
}
inline Tensor dispatch_elu_(Tensor self, Scalar alpha, Scalar scale, Scalar input_scale) {

  AutoNoGIL no_gil;
  return at::elu_(self, alpha, scale, input_scale);
}
inline std::tuple<Tensor,Tensor> dispatch_fractional_max_pool2d(const Tensor & self, IntList kernel_size, IntList output_size, const Tensor & random_samples, Tensor & output, Tensor & indices) {

  AutoNoGIL no_gil;
  return at::fractional_max_pool2d_out(output, indices, self, kernel_size, output_size, random_samples);
}
inline std::tuple<Tensor,Tensor> dispatch_fractional_max_pool2d(const Tensor & self, IntList kernel_size, IntList output_size, const Tensor & random_samples) {

  AutoNoGIL no_gil;
  return at::fractional_max_pool2d(self, kernel_size, output_size, random_samples);
}
inline Tensor dispatch_glu(const Tensor & self, int64_t dim, Tensor output) {

  AutoNoGIL no_gil;
  return at::glu_out(output, self, dim);
}
inline Tensor dispatch_glu(const Tensor & self, int64_t dim) {

  AutoNoGIL no_gil;
  return at::glu(self, dim);
}
inline Tensor dispatch_hardtanh(const Tensor & self, Scalar min_val, Scalar max_val, Tensor output) {

  AutoNoGIL no_gil;
  return at::hardtanh_out(output, self, min_val, max_val);
}
inline Tensor dispatch_hardtanh(const Tensor & self, Scalar min_val, Scalar max_val) {

  AutoNoGIL no_gil;
  return at::hardtanh(self, min_val, max_val);
}
inline Tensor dispatch_hardtanh_(Tensor self, Scalar min_val, Scalar max_val) {

  AutoNoGIL no_gil;
  return at::hardtanh_(self, min_val, max_val);
}
inline Tensor dispatch_l1_loss(const Tensor & self, const Tensor & target, int64_t reduction, Tensor output) {

  AutoNoGIL no_gil;
  return at::l1_loss_out(output, self, target, reduction);
}
inline Tensor dispatch_l1_loss(const Tensor & self, const Tensor & target, int64_t reduction) {

  AutoNoGIL no_gil;
  return at::l1_loss(self, target, reduction);
}
inline Tensor dispatch_leaky_relu(const Tensor & self, Scalar negative_slope, Tensor output) {

  AutoNoGIL no_gil;
  return at::leaky_relu_out(output, self, negative_slope);
}
inline Tensor dispatch_leaky_relu(const Tensor & self, Scalar negative_slope) {

  AutoNoGIL no_gil;
  return at::leaky_relu(self, negative_slope);
}
inline Tensor dispatch_leaky_relu_(Tensor self, Scalar negative_slope) {

  AutoNoGIL no_gil;
  return at::leaky_relu_(self, negative_slope);
}
inline Tensor dispatch_log_sigmoid(const Tensor & self, Tensor output) {

  AutoNoGIL no_gil;
  return at::log_sigmoid_out(output, self);
}
inline Tensor dispatch_log_sigmoid(const Tensor & self) {

  AutoNoGIL no_gil;
  return at::log_sigmoid(self);
}
inline std::tuple<Tensor,Tensor> dispatch_max_pool2d_with_indices(const Tensor & self, IntList kernel_size, IntList stride, IntList padding, IntList dilation, bool ceil_mode, Tensor & output, Tensor & indices) {

  AutoNoGIL no_gil;
  return at::max_pool2d_with_indices_out(output, indices, self, kernel_size, stride, padding, dilation, ceil_mode);
}
inline std::tuple<Tensor,Tensor> dispatch_max_pool2d_with_indices(const Tensor & self, IntList kernel_size, IntList stride, IntList padding, IntList dilation, bool ceil_mode) {

  AutoNoGIL no_gil;
  return at::max_pool2d_with_indices(self, kernel_size, stride, padding, dilation, ceil_mode);
}
inline std::tuple<Tensor,Tensor> dispatch_max_pool3d_with_indices(const Tensor & self, IntList kernel_size, IntList stride, IntList padding, IntList dilation, bool ceil_mode, Tensor & output, Tensor & indices) {

  AutoNoGIL no_gil;
  return at::max_pool3d_with_indices_out(output, indices, self, kernel_size, stride, padding, dilation, ceil_mode);
}
inline std::tuple<Tensor,Tensor> dispatch_max_pool3d_with_indices(const Tensor & self, IntList kernel_size, IntList stride, IntList padding, IntList dilation, bool ceil_mode) {

  AutoNoGIL no_gil;
  return at::max_pool3d_with_indices(self, kernel_size, stride, padding, dilation, ceil_mode);
}
inline Tensor dispatch_max_unpool2d(const Tensor & self, const Tensor & indices, IntList output_size, Tensor output) {

  AutoNoGIL no_gil;
  return at::max_unpool2d_out(output, self, indices, output_size);
}
inline Tensor dispatch_max_unpool2d(const Tensor & self, const Tensor & indices, IntList output_size) {

  AutoNoGIL no_gil;
  return at::max_unpool2d(self, indices, output_size);
}
inline Tensor dispatch_max_unpool3d(const Tensor & self, const Tensor & indices, IntList output_size, IntList stride, IntList padding, Tensor output) {

  AutoNoGIL no_gil;
  return at::max_unpool3d_out(output, self, indices, output_size, stride, padding);
}
inline Tensor dispatch_max_unpool3d(const Tensor & self, const Tensor & indices, IntList output_size, IntList stride, IntList padding) {

  AutoNoGIL no_gil;
  return at::max_unpool3d(self, indices, output_size, stride, padding);
}
inline Tensor dispatch_mse_loss(const Tensor & self, const Tensor & target, int64_t reduction, Tensor output) {

  AutoNoGIL no_gil;
  return at::mse_loss_out(output, self, target, reduction);
}
inline Tensor dispatch_mse_loss(const Tensor & self, const Tensor & target, int64_t reduction) {

  AutoNoGIL no_gil;
  return at::mse_loss(self, target, reduction);
}
inline Tensor dispatch_multi_margin_loss(const Tensor & self, const Tensor & target, Scalar p, Scalar margin, const Tensor & weight, int64_t reduction, Tensor output) {

  AutoNoGIL no_gil;
  return at::multi_margin_loss_out(output, self, target, p, margin, weight, reduction);
}
inline Tensor dispatch_multi_margin_loss(const Tensor & self, const Tensor & target, Scalar p, Scalar margin, const Tensor & weight, int64_t reduction) {

  AutoNoGIL no_gil;
  return at::multi_margin_loss(self, target, p, margin, weight, reduction);
}
inline Tensor dispatch_multilabel_margin_loss(const Tensor & self, const Tensor & target, int64_t reduction, Tensor output) {

  AutoNoGIL no_gil;
  return at::multilabel_margin_loss_out(output, self, target, reduction);
}
inline Tensor dispatch_multilabel_margin_loss(const Tensor & self, const Tensor & target, int64_t reduction) {

  AutoNoGIL no_gil;
  return at::multilabel_margin_loss(self, target, reduction);
}
inline Tensor dispatch_nll_loss(const Tensor & self, const Tensor & target, const Tensor & weight, int64_t reduction, int64_t ignore_index, Tensor output) {

  AutoNoGIL no_gil;
  return at::nll_loss_out(output, self, target, weight, reduction, ignore_index);
}
inline Tensor dispatch_nll_loss(const Tensor & self, const Tensor & target, const Tensor & weight, int64_t reduction, int64_t ignore_index) {

  AutoNoGIL no_gil;
  return at::nll_loss(self, target, weight, reduction, ignore_index);
}
inline Tensor dispatch_nll_loss2d(const Tensor & self, const Tensor & target, const Tensor & weight, int64_t reduction, int64_t ignore_index, Tensor output) {

  AutoNoGIL no_gil;
  return at::nll_loss2d_out(output, self, target, weight, reduction, ignore_index);
}
inline Tensor dispatch_nll_loss2d(const Tensor & self, const Tensor & target, const Tensor & weight, int64_t reduction, int64_t ignore_index) {

  AutoNoGIL no_gil;
  return at::nll_loss2d(self, target, weight, reduction, ignore_index);
}
inline Tensor dispatch_reflection_pad1d(const Tensor & self, IntList padding, Tensor output) {

  AutoNoGIL no_gil;
  return at::reflection_pad1d_out(output, self, padding);
}
inline Tensor dispatch_reflection_pad1d(const Tensor & self, IntList padding) {

  AutoNoGIL no_gil;
  return at::reflection_pad1d(self, padding);
}
inline Tensor dispatch_reflection_pad2d(const Tensor & self, IntList padding, Tensor output) {

  AutoNoGIL no_gil;
  return at::reflection_pad2d_out(output, self, padding);
}
inline Tensor dispatch_reflection_pad2d(const Tensor & self, IntList padding) {

  AutoNoGIL no_gil;
  return at::reflection_pad2d(self, padding);
}
inline Tensor dispatch_replication_pad1d(const Tensor & self, IntList padding, Tensor output) {

  AutoNoGIL no_gil;
  return at::replication_pad1d_out(output, self, padding);
}
inline Tensor dispatch_replication_pad1d(const Tensor & self, IntList padding) {

  AutoNoGIL no_gil;
  return at::replication_pad1d(self, padding);
}
inline Tensor dispatch_replication_pad2d(const Tensor & self, IntList padding, Tensor output) {

  AutoNoGIL no_gil;
  return at::replication_pad2d_out(output, self, padding);
}
inline Tensor dispatch_replication_pad2d(const Tensor & self, IntList padding) {

  AutoNoGIL no_gil;
  return at::replication_pad2d(self, padding);
}
inline Tensor dispatch_replication_pad3d(const Tensor & self, IntList padding, Tensor output) {

  AutoNoGIL no_gil;
  return at::replication_pad3d_out(output, self, padding);
}
inline Tensor dispatch_replication_pad3d(const Tensor & self, IntList padding) {

  AutoNoGIL no_gil;
  return at::replication_pad3d(self, padding);
}
inline Tensor dispatch_rrelu_with_noise(const Tensor & self, const Tensor & noise, Scalar lower, Scalar upper, bool training, Generator * generator, Tensor output) {

  AutoNoGIL no_gil;
  return at::rrelu_with_noise_out(output, self, noise, lower, upper, training, generator);
}
inline Tensor dispatch_rrelu_with_noise(const Tensor & self, const Tensor & noise, Scalar lower, Scalar upper, bool training, Generator * generator) {

  AutoNoGIL no_gil;
  return at::rrelu_with_noise(self, noise, lower, upper, training, generator);
}
inline Tensor dispatch_rrelu_with_noise_(Tensor self, const Tensor & noise, Scalar lower, Scalar upper, bool training, Generator * generator) {

  AutoNoGIL no_gil;
  return at::rrelu_with_noise_(self, noise, lower, upper, training, generator);
}
inline Tensor dispatch_smooth_l1_loss(const Tensor & self, const Tensor & target, int64_t reduction, Tensor output) {

  AutoNoGIL no_gil;
  return at::smooth_l1_loss_out(output, self, target, reduction);
}
inline Tensor dispatch_smooth_l1_loss(const Tensor & self, const Tensor & target, int64_t reduction) {

  AutoNoGIL no_gil;
  return at::smooth_l1_loss(self, target, reduction);
}
inline Tensor dispatch_soft_margin_loss(const Tensor & self, const Tensor & target, int64_t reduction, Tensor output) {

  AutoNoGIL no_gil;
  return at::soft_margin_loss_out(output, self, target, reduction);
}
inline Tensor dispatch_soft_margin_loss(const Tensor & self, const Tensor & target, int64_t reduction) {

  AutoNoGIL no_gil;
  return at::soft_margin_loss(self, target, reduction);
}
inline Tensor dispatch_softplus(const Tensor & self, Scalar beta, Scalar threshold, Tensor output) {

  AutoNoGIL no_gil;
  return at::softplus_out(output, self, beta, threshold);
}
inline Tensor dispatch_softplus(const Tensor & self, Scalar beta, Scalar threshold) {

  AutoNoGIL no_gil;
  return at::softplus(self, beta, threshold);
}
inline Tensor dispatch_softshrink(const Tensor & self, Scalar lambd, Tensor output) {

  AutoNoGIL no_gil;
  return at::softshrink_out(output, self, lambd);
}
inline Tensor dispatch_softshrink(const Tensor & self, Scalar lambd) {

  AutoNoGIL no_gil;
  return at::softshrink(self, lambd);
}
inline Tensor dispatch_thnn_col2im(const Tensor & self, IntList output_size, IntList kernel_size, IntList dilation, IntList padding, IntList stride) {

  AutoNoGIL no_gil;
  return at::thnn_col2im(self, output_size, kernel_size, dilation, padding, stride);
}
inline Tensor dispatch_thnn_conv2d(const Tensor & self, const Tensor & weight, IntList kernel_size, const Tensor & bias, IntList stride, IntList padding, Tensor output) {

  AutoNoGIL no_gil;
  return at::thnn_conv2d_out(output, self, weight, kernel_size, bias, stride, padding);
}
inline Tensor dispatch_thnn_conv2d(const Tensor & self, const Tensor & weight, IntList kernel_size, const Tensor & bias, IntList stride, IntList padding) {

  AutoNoGIL no_gil;
  return at::thnn_conv2d(self, weight, kernel_size, bias, stride, padding);
}
inline Tensor dispatch_thnn_conv3d(const Tensor & self, const Tensor & weight, IntList kernel_size, const Tensor & bias, IntList stride, IntList padding, Tensor output) {

  AutoNoGIL no_gil;
  return at::thnn_conv3d_out(output, self, weight, kernel_size, bias, stride, padding);
}
inline Tensor dispatch_thnn_conv3d(const Tensor & self, const Tensor & weight, IntList kernel_size, const Tensor & bias, IntList stride, IntList padding) {

  AutoNoGIL no_gil;
  return at::thnn_conv3d(self, weight, kernel_size, bias, stride, padding);
}
inline Tensor dispatch_thnn_conv_depthwise2d(const Tensor & self, const Tensor & weight, IntList kernel_size, const Tensor & bias, IntList stride, IntList padding, IntList dilation, Tensor output) {

  AutoNoGIL no_gil;
  return at::thnn_conv_depthwise2d_out(output, self, weight, kernel_size, bias, stride, padding, dilation);
}
inline Tensor dispatch_thnn_conv_depthwise2d(const Tensor & self, const Tensor & weight, IntList kernel_size, const Tensor & bias, IntList stride, IntList padding, IntList dilation) {

  AutoNoGIL no_gil;
  return at::thnn_conv_depthwise2d(self, weight, kernel_size, bias, stride, padding, dilation);
}
inline Tensor dispatch_thnn_conv_dilated2d(const Tensor & self, const Tensor & weight, IntList kernel_size, const Tensor & bias, IntList stride, IntList padding, IntList dilation, Tensor output) {

  AutoNoGIL no_gil;
  return at::thnn_conv_dilated2d_out(output, self, weight, kernel_size, bias, stride, padding, dilation);
}
inline Tensor dispatch_thnn_conv_dilated2d(const Tensor & self, const Tensor & weight, IntList kernel_size, const Tensor & bias, IntList stride, IntList padding, IntList dilation) {

  AutoNoGIL no_gil;
  return at::thnn_conv_dilated2d(self, weight, kernel_size, bias, stride, padding, dilation);
}
inline Tensor dispatch_thnn_conv_dilated3d(const Tensor & self, const Tensor & weight, IntList kernel_size, const Tensor & bias, IntList stride, IntList padding, IntList dilation, Tensor output) {

  AutoNoGIL no_gil;
  return at::thnn_conv_dilated3d_out(output, self, weight, kernel_size, bias, stride, padding, dilation);
}
inline Tensor dispatch_thnn_conv_dilated3d(const Tensor & self, const Tensor & weight, IntList kernel_size, const Tensor & bias, IntList stride, IntList padding, IntList dilation) {

  AutoNoGIL no_gil;
  return at::thnn_conv_dilated3d(self, weight, kernel_size, bias, stride, padding, dilation);
}
inline Tensor dispatch_thnn_conv_transpose2d(const Tensor & self, const Tensor & weight, IntList kernel_size, const Tensor & bias, IntList stride, IntList padding, IntList output_padding, IntList dilation, Tensor output) {

  AutoNoGIL no_gil;
  return at::thnn_conv_transpose2d_out(output, self, weight, kernel_size, bias, stride, padding, output_padding, dilation);
}
inline Tensor dispatch_thnn_conv_transpose2d(const Tensor & self, const Tensor & weight, IntList kernel_size, const Tensor & bias, IntList stride, IntList padding, IntList output_padding, IntList dilation) {

  AutoNoGIL no_gil;
  return at::thnn_conv_transpose2d(self, weight, kernel_size, bias, stride, padding, output_padding, dilation);
}
inline Tensor dispatch_thnn_conv_transpose3d(const Tensor & self, const Tensor & weight, IntList kernel_size, const Tensor & bias, IntList stride, IntList padding, IntList output_padding, IntList dilation, Tensor output) {

  AutoNoGIL no_gil;
  return at::thnn_conv_transpose3d_out(output, self, weight, kernel_size, bias, stride, padding, output_padding, dilation);
}
inline Tensor dispatch_thnn_conv_transpose3d(const Tensor & self, const Tensor & weight, IntList kernel_size, const Tensor & bias, IntList stride, IntList padding, IntList output_padding, IntList dilation) {

  AutoNoGIL no_gil;
  return at::thnn_conv_transpose3d(self, weight, kernel_size, bias, stride, padding, output_padding, dilation);
}
inline Tensor dispatch_thnn_im2col(const Tensor & self, IntList kernel_size, IntList dilation, IntList padding, IntList stride) {

  AutoNoGIL no_gil;
  return at::thnn_im2col(self, kernel_size, dilation, padding, stride);
}
inline Tensor dispatch_upsample_bicubic2d(const Tensor & self, IntList output_size, bool align_corners, Tensor output) {

  AutoNoGIL no_gil;
  return at::upsample_bicubic2d_out(output, self, output_size, align_corners);
}
inline Tensor dispatch_upsample_bicubic2d(const Tensor & self, IntList output_size, bool align_corners) {

  AutoNoGIL no_gil;
  return at::upsample_bicubic2d(self, output_size, align_corners);
}
inline Tensor dispatch_upsample_bilinear2d(const Tensor & self, IntList output_size, bool align_corners, Tensor output) {

  AutoNoGIL no_gil;
  return at::upsample_bilinear2d_out(output, self, output_size, align_corners);
}
inline Tensor dispatch_upsample_bilinear2d(const Tensor & self, IntList output_size, bool align_corners) {

  AutoNoGIL no_gil;
  return at::upsample_bilinear2d(self, output_size, align_corners);
}
inline Tensor dispatch_upsample_linear1d(const Tensor & self, IntList output_size, bool align_corners, Tensor output) {

  AutoNoGIL no_gil;
  return at::upsample_linear1d_out(output, self, output_size, align_corners);
}
inline Tensor dispatch_upsample_linear1d(const Tensor & self, IntList output_size, bool align_corners) {

  AutoNoGIL no_gil;
  return at::upsample_linear1d(self, output_size, align_corners);
}
inline Tensor dispatch_upsample_nearest1d(const Tensor & self, IntList output_size, Tensor output) {

  AutoNoGIL no_gil;
  return at::upsample_nearest1d_out(output, self, output_size);
}
inline Tensor dispatch_upsample_nearest1d(const Tensor & self, IntList output_size) {

  AutoNoGIL no_gil;
  return at::upsample_nearest1d(self, output_size);
}
inline Tensor dispatch_upsample_nearest2d(const Tensor & self, IntList output_size, Tensor output) {

  AutoNoGIL no_gil;
  return at::upsample_nearest2d_out(output, self, output_size);
}
inline Tensor dispatch_upsample_nearest2d(const Tensor & self, IntList output_size) {

  AutoNoGIL no_gil;
  return at::upsample_nearest2d(self, output_size);
}
inline Tensor dispatch_upsample_nearest3d(const Tensor & self, IntList output_size, Tensor output) {

  AutoNoGIL no_gil;
  return at::upsample_nearest3d_out(output, self, output_size);
}
inline Tensor dispatch_upsample_nearest3d(const Tensor & self, IntList output_size) {

  AutoNoGIL no_gil;
  return at::upsample_nearest3d(self, output_size);
}
inline Tensor dispatch_upsample_trilinear3d(const Tensor & self, IntList output_size, bool align_corners, Tensor output) {

  AutoNoGIL no_gil;
  return at::upsample_trilinear3d_out(output, self, output_size, align_corners);
}
inline Tensor dispatch_upsample_trilinear3d(const Tensor & self, IntList output_size, bool align_corners) {

  AutoNoGIL no_gil;
  return at::upsample_trilinear3d(self, output_size, align_corners);
}

}} // namespace torch::autograd
