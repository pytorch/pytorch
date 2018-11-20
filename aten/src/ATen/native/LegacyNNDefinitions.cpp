#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>

namespace at { namespace native {

Tensor & binary_cross_entropy_out(Tensor & output, const Tensor & self, const Tensor & target, const Tensor & weight, int64_t reduction) {
  return at::_thnn_binary_cross_entropy_out(output, self, target, weight, reduction);
}
Tensor binary_cross_entropy(const Tensor & self, const Tensor & target, const Tensor & weight, int64_t reduction) {
  return at::_thnn_binary_cross_entropy(self, target, weight, reduction);
}
Tensor & binary_cross_entropy_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, const Tensor & target, const Tensor & weight, int64_t reduction) {
  return at::_thnn_binary_cross_entropy_backward_out(grad_input, grad_output, self, target, weight, reduction);
}

Tensor binary_cross_entropy_backward(const Tensor & grad_output, const Tensor & self, const Tensor & target, const Tensor & weight, int64_t reduction) {
  return at::_thnn_binary_cross_entropy_backward(grad_output, self, target, weight, reduction);
}

Tensor & mse_loss_out(Tensor & output, const Tensor & self, const Tensor & target, int64_t reduction) {
  return at::_thnn_mse_loss_out(output, self, target, reduction);
}

Tensor mse_loss(const Tensor & self, const Tensor & target, int64_t reduction) {
  return at::_thnn_mse_loss(self, target, reduction);
}

Tensor & mse_loss_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, const Tensor & target, int64_t reduction) {
  return at::_thnn_mse_loss_backward_out(grad_input, grad_output, self, target, reduction);
}

Tensor mse_loss_backward(const Tensor & grad_output, const Tensor & self, const Tensor & target, int64_t reduction) {
  return at::_thnn_mse_loss_backward(grad_output, self, target, reduction);
}

Tensor & l1_loss_out(Tensor & output, const Tensor & self, const Tensor & target, int64_t reduction) {
  return at::_thnn_l1_loss_out(output, self, target, reduction);
}

Tensor l1_loss(const Tensor & self, const Tensor & target, int64_t reduction) {
  return at::_thnn_l1_loss(self, target, reduction);
}

Tensor & l1_loss_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, const Tensor & target, int64_t reduction) {
  return at::_thnn_l1_loss_backward_out(grad_input, grad_output, self, target, reduction);
}

Tensor l1_loss_backward(const Tensor & grad_output, const Tensor & self, const Tensor & target, int64_t reduction) {
  return at::_thnn_l1_loss_backward(grad_output, self, target, reduction);
}

Tensor & multi_margin_loss_out(Tensor & output, const Tensor & self, const Tensor & target,
                                     Scalar p, Scalar margin, const Tensor & weight, int64_t reduction) {
  return at::_thnn_multi_margin_loss_out(output, self, target, p, margin, weight, reduction);
}

Tensor multi_margin_loss(const Tensor & self, const Tensor & target,
                               Scalar p, Scalar margin, const Tensor & weight, int64_t reduction) {
  return at::_thnn_multi_margin_loss(self, target, p, margin, weight, reduction);
}

Tensor & multi_margin_loss_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, const Tensor & target,
                                        Scalar p, Scalar margin, const Tensor & weight, int64_t reduction) {
  return at::_thnn_multi_margin_loss_backward_out(grad_input, grad_output, self, target, p, margin, weight, reduction);
}

Tensor multi_margin_loss_backward(const Tensor & grad_output, const Tensor & self, const Tensor & target,
                                  Scalar p, Scalar margin, const Tensor & weight, int64_t reduction) {
  return at::_thnn_multi_margin_loss_backward(grad_output, self, target, p, margin, weight, reduction);
}

Tensor & smooth_l1_loss_out(Tensor & output, const Tensor & self, const Tensor & target, int64_t reduction) {
  return at::_thnn_smooth_l1_loss_out(output, self, target, reduction);
}

Tensor smooth_l1_loss(const Tensor & self, const Tensor & target, int64_t reduction) {
  return at::_thnn_smooth_l1_loss(self, target, reduction);
}

Tensor & smooth_l1_loss_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self,
                                     const Tensor & target, int64_t reduction) {
  return at::_thnn_smooth_l1_loss_backward_out(grad_input, grad_output, self, target, reduction);
}

Tensor smooth_l1_loss_backward(const Tensor & grad_output, const Tensor & self, const Tensor & target, int64_t reduction) {
  return at::_thnn_smooth_l1_loss_backward(grad_output, self, target, reduction);
}

Tensor & soft_margin_loss_out(Tensor & output, const Tensor & self, const Tensor & target, int64_t reduction) {
  return at::_thnn_soft_margin_loss_out(output, self, target, reduction);
}

Tensor soft_margin_loss(const Tensor & self, const Tensor & target, int64_t reduction) {
  return at::_thnn_soft_margin_loss(self, target, reduction);
}

Tensor & soft_margin_loss_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self,
                                       const Tensor & target, int64_t reduction) {
  return at::_thnn_soft_margin_loss_backward_out(grad_input, grad_output, self, target, reduction);
}

Tensor soft_margin_loss_backward(const Tensor & grad_output, const Tensor & self, const Tensor & target, int64_t reduction) {
  return at::_thnn_soft_margin_loss_backward(grad_output, self, target, reduction);
}

Tensor & elu_out(Tensor & output, const Tensor & self, Scalar alpha, Scalar scale, Scalar input_scale) {
  return at::_thnn_elu_out(output, self, alpha, scale, input_scale);
}

Tensor elu(const Tensor & self, Scalar alpha, Scalar scale, Scalar input_scale) {
  return at::_thnn_elu(self, alpha, scale, input_scale);
}

Tensor & elu_backward_out(Tensor & grad_input, const Tensor & grad_output, Scalar alpha, Scalar scale, Scalar input_scale, const Tensor & output) {
  return at::_thnn_elu_backward_out(grad_input, grad_output, alpha, scale, input_scale, output);
}

Tensor elu_backward(const Tensor & grad_output, Scalar alpha, Scalar scale, Scalar input_scale, const Tensor & output) {
  return at::_thnn_elu_backward(grad_output, alpha, scale, input_scale, output);
}

Tensor & elu_(Tensor & self, Scalar alpha, Scalar scale, Scalar input_scale) {
  return at::_thnn_elu_(self, alpha, scale, input_scale);
}

Tensor & glu_out(Tensor & output, const Tensor & self, int64_t dim) {
  return at::_thnn_glu_out(output, self, dim);
}

Tensor glu(const Tensor & self, int64_t dim) {
  return at::_thnn_glu(self, dim);
}

Tensor & glu_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, int64_t dim) {
  return at::_thnn_glu_backward_out(grad_input, grad_output, self, dim);
}

Tensor glu_backward(const Tensor & grad_output, const Tensor & self, int64_t dim) {
  return at::_thnn_glu_backward(grad_output, self, dim);
}

Tensor & hardtanh_out(Tensor & output, const Tensor & self, Scalar min_val, Scalar max_val) {
  return at::_thnn_hardtanh_out(output, self, min_val, max_val);
}

Tensor hardtanh(const Tensor & self, Scalar min_val, Scalar max_val) {
  return at::_thnn_hardtanh(self, min_val, max_val);
}

Tensor & hardtanh_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, Scalar min_val, Scalar max_val) {
  return at::_thnn_hardtanh_backward_out(grad_input, grad_output, self, min_val, max_val);
}

Tensor hardtanh_backward(const Tensor & grad_output, const Tensor & self, Scalar min_val, Scalar max_val) {
  return at::_thnn_hardtanh_backward(grad_output, self, min_val, max_val);
}

Tensor & hardtanh_(Tensor & self, Scalar min_val, Scalar max_val) {
  return at::_thnn_hardtanh_(self, min_val, max_val);
}

Tensor & leaky_relu_out(Tensor & output, const Tensor & self, Scalar negative_slope) {
  return at::_thnn_leaky_relu_out(output, self, negative_slope);
}

Tensor leaky_relu(const Tensor & self, Scalar negative_slope) {
  return at::_thnn_leaky_relu(self, negative_slope);
}

Tensor & leaky_relu_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, Scalar negative_slope) {
  return at::_thnn_leaky_relu_backward_out(grad_input, grad_output, self, negative_slope);
}

Tensor leaky_relu_backward(const Tensor & grad_output, const Tensor & self, Scalar negative_slope) {
  return at::_thnn_leaky_relu_backward(grad_output, self, negative_slope);
}

Tensor & leaky_relu_(Tensor & self, Scalar negative_slope) {
  return at::_thnn_leaky_relu_(self, negative_slope);
}

Tensor & rrelu_with_noise_out(Tensor & output, const Tensor & self, const Tensor & noise, Scalar lower, Scalar upper, bool training, Generator * generator) {
  return at::_thnn_rrelu_with_noise_out(output, self, noise, lower, upper, training, generator);
}

Tensor rrelu_with_noise(const Tensor & self, const Tensor & noise, Scalar lower, Scalar upper, bool training, Generator * generator) {
  return at::_thnn_rrelu_with_noise(self, noise, lower, upper, training, generator);
}

Tensor & rrelu_with_noise_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, const Tensor & noise, Scalar lower, Scalar upper, bool training) {
  return at::_thnn_rrelu_with_noise_backward_out(grad_input, grad_output, self, noise, lower, upper, training);
}

Tensor rrelu_with_noise_backward(const Tensor & grad_output, const Tensor & self, const Tensor & noise, Scalar lower, Scalar upper, bool training) {
  return at::_thnn_rrelu_with_noise_backward(grad_output, self, noise, lower, upper, training);
}

Tensor & rrelu_with_noise_(Tensor & self, const Tensor & noise, Scalar lower, Scalar upper, bool training, Generator * generator) {
  return at::_thnn_rrelu_with_noise_(self, noise, lower, upper, training, generator);
}

Tensor & softplus_out(Tensor & output, const Tensor & self, Scalar beta, Scalar threshold) {
  return at::_thnn_softplus_out(output, self, beta, threshold);
}

Tensor softplus(const Tensor & self, Scalar beta, Scalar threshold) {
  return at::_thnn_softplus(self, beta, threshold);
}

Tensor & softplus_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, Scalar beta, Scalar threshold, const Tensor & output) {
  return at::_thnn_softplus_backward_out(grad_input, grad_output, self, beta, threshold, output);
}

Tensor softplus_backward(const Tensor & grad_output, const Tensor & self, Scalar beta, Scalar threshold, const Tensor & output) {
  return at::_thnn_softplus_backward(grad_output, self, beta, threshold, output);
}

Tensor & softshrink_out(Tensor & output, const Tensor & self, Scalar lambd) {
  return at::_thnn_softshrink_out(output, self, lambd);
}

Tensor softshrink(const Tensor & self, Scalar lambd) {
  return at::_thnn_softshrink(self, lambd);
}

Tensor & softshrink_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, Scalar lambd) {
  return at::_thnn_softshrink_backward_out(grad_input, grad_output, self, lambd);
}

Tensor softshrink_backward(const Tensor & grad_output, const Tensor & self, Scalar lambd) {
  return at::_thnn_softshrink_backward(grad_output, self, lambd);
}

Tensor & adaptive_avg_pool2d_out(Tensor & output, const Tensor & self, IntList output_size) {
  return at::_thnn_adaptive_avg_pool2d_out(output, self, output_size);
}

Tensor adaptive_avg_pool2d(const Tensor & self, IntList output_size) {
  return at::_thnn_adaptive_avg_pool2d(self, output_size);
}

Tensor & adaptive_avg_pool2d_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self) {
  return at::_thnn_adaptive_avg_pool2d_backward_out(grad_input, grad_output, self);
}

Tensor adaptive_avg_pool2d_backward(const Tensor & grad_output, const Tensor & self) {
  return at::_thnn_adaptive_avg_pool2d_backward(grad_output, self);
}

Tensor & adaptive_avg_pool3d_out(Tensor & output, const Tensor & self, IntList output_size) {
  return at::_thnn_adaptive_avg_pool3d_out(output, self, output_size);
}

Tensor adaptive_avg_pool3d(const Tensor & self, IntList output_size) {
  return at::_thnn_adaptive_avg_pool3d(self, output_size);
}

Tensor & adaptive_avg_pool3d_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self) {
  return at::_thnn_adaptive_avg_pool3d_backward_out(grad_input, grad_output, self);
}

Tensor adaptive_avg_pool3d_backward(const Tensor & grad_output, const Tensor & self) {
  return at::_thnn_adaptive_avg_pool3d_backward(grad_output, self);
}

std::tuple<Tensor &,Tensor &> adaptive_max_pool2d_out(Tensor & output, Tensor & indices, const Tensor & self, IntList output_size) {
  return at::_thnn_adaptive_max_pool2d_out(output, indices, self, output_size);
}

std::tuple<Tensor,Tensor> adaptive_max_pool2d(const Tensor & self, IntList output_size) {
  return at::_thnn_adaptive_max_pool2d(self, output_size);
}

Tensor & adaptive_max_pool2d_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, const Tensor & indices) {
  return at::_thnn_adaptive_max_pool2d_backward_out(grad_input, grad_output, self, indices);
}

Tensor adaptive_max_pool2d_backward(const Tensor & grad_output, const Tensor & self, const Tensor & indices) {
  return at::_thnn_adaptive_max_pool2d_backward(grad_output, self, indices);
}

std::tuple<Tensor &,Tensor &> adaptive_max_pool3d_out(Tensor & output, Tensor & indices, const Tensor & self, IntList output_size) {
  return at::_thnn_adaptive_max_pool3d_out(output, indices, self, output_size);
}

std::tuple<Tensor,Tensor> adaptive_max_pool3d(const Tensor & self, IntList output_size) {
  return at::_thnn_adaptive_max_pool3d(self, output_size);
}

Tensor & adaptive_max_pool3d_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, const Tensor & indices) {
  return at::_thnn_adaptive_max_pool3d_backward_out(grad_input, grad_output, self, indices);
}

Tensor adaptive_max_pool3d_backward(const Tensor & grad_output, const Tensor & self, const Tensor & indices) {
  return at::_thnn_adaptive_max_pool3d_backward(grad_output, self, indices);
}

Tensor & avg_pool2d_out(Tensor & output, const Tensor & self, IntList kernel_size, IntList stride, IntList padding, bool ceil_mode, bool count_include_pad) {
  return at::_thnn_avg_pool2d_out(output, self, kernel_size, stride, padding, ceil_mode, count_include_pad);
}

Tensor avg_pool2d(const Tensor & self, IntList kernel_size, IntList stride, IntList padding, bool ceil_mode, bool count_include_pad) {
  return at::_thnn_avg_pool2d(self, kernel_size, stride, padding, ceil_mode, count_include_pad);
}

Tensor & avg_pool2d_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, IntList kernel_size, IntList stride, IntList padding, bool ceil_mode, bool count_include_pad) {
  return at::_thnn_avg_pool2d_backward_out(grad_input, grad_output, self, kernel_size, stride, padding, ceil_mode, count_include_pad);
}

Tensor avg_pool2d_backward(const Tensor & grad_output, const Tensor & self, IntList kernel_size, IntList stride, IntList padding, bool ceil_mode, bool count_include_pad) {
  return at::_thnn_avg_pool2d_backward(grad_output, self, kernel_size, stride, padding, ceil_mode, count_include_pad);
}

Tensor & avg_pool3d_out(Tensor & output, const Tensor & self, IntList kernel_size, IntList stride, IntList padding, bool ceil_mode, bool count_include_pad) {
  return at::_thnn_avg_pool3d_out(output, self, kernel_size, stride, padding, ceil_mode, count_include_pad);
}

Tensor avg_pool3d(const Tensor & self, IntList kernel_size, IntList stride, IntList padding, bool ceil_mode, bool count_include_pad) {
  return at::_thnn_avg_pool3d(self, kernel_size, stride, padding, ceil_mode, count_include_pad);
}

Tensor & avg_pool3d_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, IntList kernel_size, IntList stride, IntList padding, bool ceil_mode, bool count_include_pad) {
  return at::_thnn_avg_pool3d_backward_out(grad_input, grad_output, self, kernel_size, stride, padding, ceil_mode, count_include_pad);
}

Tensor avg_pool3d_backward(const Tensor & grad_output, const Tensor & self, IntList kernel_size, IntList stride, IntList padding, bool ceil_mode, bool count_include_pad) {
  return at::_thnn_avg_pool3d_backward(grad_output, self, kernel_size, stride, padding, ceil_mode, count_include_pad);
}

std::tuple<Tensor &,Tensor &> fractional_max_pool2d_out(Tensor & output, Tensor & indices, const Tensor & self, IntList kernel_size, IntList output_size, const Tensor & random_samples) {
  return at::_thnn_fractional_max_pool2d_out(output, indices, self, kernel_size, output_size, random_samples);
}

std::tuple<Tensor,Tensor> fractional_max_pool2d(const Tensor & self, IntList kernel_size, IntList output_size, const Tensor & random_samples) {
  return at::_thnn_fractional_max_pool2d(self, kernel_size, output_size, random_samples);
}

Tensor & fractional_max_pool2d_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, IntList kernel_size, IntList output_size, const Tensor & indices) {
  return at::_thnn_fractional_max_pool2d_backward_out(grad_input, grad_output, self, kernel_size, output_size, indices);
}

Tensor fractional_max_pool2d_backward(const Tensor & grad_output, const Tensor & self, IntList kernel_size, IntList output_size, const Tensor & indices) {
  return at::_thnn_fractional_max_pool2d_backward(grad_output, self, kernel_size, output_size, indices);
}

std::tuple<Tensor &,Tensor &> max_pool2d_with_indices_out(Tensor & output, Tensor & indices, const Tensor & self, IntList kernel_size, IntList stride, IntList padding, IntList dilation, bool ceil_mode) {
  return at::_thnn_max_pool2d_with_indices_out(output, indices, self, kernel_size, stride, padding, dilation, ceil_mode);
}

std::tuple<Tensor,Tensor> max_pool2d_with_indices(const Tensor & self, IntList kernel_size, IntList stride, IntList padding, IntList dilation, bool ceil_mode) {
  return at::_thnn_max_pool2d_with_indices(self, kernel_size, stride, padding, dilation, ceil_mode);
}

Tensor & max_pool2d_with_indices_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, IntList kernel_size, IntList stride, IntList padding, IntList dilation, bool ceil_mode, const Tensor & indices) {
  return at::_thnn_max_pool2d_with_indices_backward_out(grad_input, grad_output, self, kernel_size, stride, padding, dilation, ceil_mode, indices);
}

Tensor max_pool2d_with_indices_backward(const Tensor & grad_output, const Tensor & self, IntList kernel_size, IntList stride, IntList padding, IntList dilation, bool ceil_mode, const Tensor & indices) {
  return at::_thnn_max_pool2d_with_indices_backward(grad_output, self, kernel_size, stride, padding, dilation, ceil_mode, indices);
}

std::tuple<Tensor &,Tensor &> max_pool3d_with_indices_out(Tensor & output, Tensor & indices, const Tensor & self, IntList kernel_size, IntList stride, IntList padding, IntList dilation, bool ceil_mode) {
  return at::_thnn_max_pool3d_with_indices_out(output, indices, self, kernel_size, stride, padding, dilation, ceil_mode);
}

std::tuple<Tensor,Tensor> max_pool3d_with_indices(const Tensor & self, IntList kernel_size, IntList stride, IntList padding, IntList dilation, bool ceil_mode) {
  return at::_thnn_max_pool3d_with_indices(self, kernel_size, stride, padding, dilation, ceil_mode);
}

Tensor & max_pool3d_with_indices_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, IntList kernel_size, IntList stride, IntList padding, IntList dilation, bool ceil_mode, const Tensor & indices) {
  return at::_thnn_max_pool3d_with_indices_backward_out(grad_input, grad_output, self, kernel_size, stride, padding, dilation, ceil_mode, indices);
}

Tensor max_pool3d_with_indices_backward(const Tensor & grad_output, const Tensor & self, IntList kernel_size, IntList stride, IntList padding, IntList dilation, bool ceil_mode, const Tensor & indices) {
  return at::_thnn_max_pool3d_with_indices_backward(grad_output, self, kernel_size, stride, padding, dilation, ceil_mode, indices);
}

Tensor & max_unpool2d_out(Tensor & output, const Tensor & self, const Tensor & indices, IntList output_size) {
  return at::_thnn_max_unpool2d_out(output, self, indices, output_size);
}

Tensor max_unpool2d(const Tensor & self, const Tensor & indices, IntList output_size) {
  return at::_thnn_max_unpool2d(self, indices, output_size);
}

Tensor & max_unpool2d_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, const Tensor & indices, IntList output_size) {
  return at::_thnn_max_unpool2d_backward_out(grad_input, grad_output, self, indices, output_size);
}

Tensor max_unpool2d_backward(const Tensor & grad_output, const Tensor & self, const Tensor & indices, IntList output_size) {
  return at::_thnn_max_unpool2d_backward(grad_output, self, indices, output_size);
}

Tensor & max_unpool3d_out(Tensor & output, const Tensor & self, const Tensor & indices, IntList output_size, IntList stride, IntList padding) {
  return at::_thnn_max_unpool3d_out(output, self, indices, output_size, stride, padding);
}

Tensor max_unpool3d(const Tensor & self, const Tensor & indices, IntList output_size, IntList stride, IntList padding) {
  return at::_thnn_max_unpool3d(self, indices, output_size, stride, padding);
}

Tensor & max_unpool3d_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, const Tensor & indices, IntList output_size, IntList stride, IntList padding) {
  return at::_thnn_max_unpool3d_backward_out(grad_input, grad_output, self, indices, output_size, stride, padding);
}

Tensor max_unpool3d_backward(const Tensor & grad_output, const Tensor & self, const Tensor & indices, IntList output_size, IntList stride, IntList padding) {
  return at::_thnn_max_unpool3d_backward(grad_output, self, indices, output_size, stride, padding);
}

Tensor & reflection_pad1d_out(Tensor & output, const Tensor & self, IntList padding) {
  return at::_thnn_reflection_pad1d_out(output, self, padding);
}

Tensor reflection_pad1d(const Tensor & self, IntList padding) {
  return at::_thnn_reflection_pad1d(self, padding);
}

Tensor & reflection_pad1d_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, IntList padding) {
  return at::_thnn_reflection_pad1d_backward_out(grad_input, grad_output, self, padding);
}

Tensor reflection_pad1d_backward(const Tensor & grad_output, const Tensor & self, IntList padding) {
  return at::_thnn_reflection_pad1d_backward(grad_output, self, padding);
}

Tensor & reflection_pad2d_out(Tensor & output, const Tensor & self, IntList padding) {
  return at::_thnn_reflection_pad2d_out(output, self, padding);
}

Tensor reflection_pad2d(const Tensor & self, IntList padding) {
  return at::_thnn_reflection_pad2d(self, padding);
}

Tensor & reflection_pad2d_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, IntList padding) {
  return at::_thnn_reflection_pad2d_backward_out(grad_input, grad_output, self, padding);
}

Tensor reflection_pad2d_backward(const Tensor & grad_output, const Tensor & self, IntList padding) {
  return at::_thnn_reflection_pad2d_backward(grad_output, self, padding);
}

Tensor & replication_pad1d_out(Tensor & output, const Tensor & self, IntList padding) {
  return at::_thnn_replication_pad1d_out(output, self, padding);
}

Tensor replication_pad1d(const Tensor & self, IntList padding) {
  return at::_thnn_replication_pad1d(self, padding);
}

Tensor & replication_pad1d_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, IntList padding) {
  return at::_thnn_replication_pad1d_backward_out(grad_input, grad_output, self, padding);
}

Tensor replication_pad1d_backward(const Tensor & grad_output, const Tensor & self, IntList padding) {
  return at::_thnn_replication_pad1d_backward(grad_output, self, padding);
}

Tensor & replication_pad2d_out(Tensor & output, const Tensor & self, IntList padding) {
  return at::_thnn_replication_pad2d_out(output, self, padding);
}

Tensor replication_pad2d(const Tensor & self, IntList padding) {
  return at::_thnn_replication_pad2d(self, padding);
}

Tensor & replication_pad2d_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, IntList padding) {
  return at::_thnn_replication_pad2d_backward_out(grad_input, grad_output, self, padding);
}

Tensor replication_pad2d_backward(const Tensor & grad_output, const Tensor & self, IntList padding) {
  return at::_thnn_replication_pad2d_backward(grad_output, self, padding);
}

Tensor & replication_pad3d_out(Tensor & output, const Tensor & self, IntList padding) {
  return at::_thnn_replication_pad3d_out(output, self, padding);
}

Tensor replication_pad3d(const Tensor & self, IntList padding) {
  return at::_thnn_replication_pad3d(self, padding);
}

Tensor & replication_pad3d_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, IntList padding) {
  return at::_thnn_replication_pad3d_backward_out(grad_input, grad_output, self, padding);
}

Tensor replication_pad3d_backward(const Tensor & grad_output, const Tensor & self, IntList padding) {
  return at::_thnn_replication_pad3d_backward(grad_output, self, padding);
}

Tensor & upsample_linear1d_out(Tensor & output, const Tensor & self, IntList output_size, bool align_corners) {
  return at::_thnn_upsample_linear1d_out(output, self, output_size, align_corners);
}

Tensor upsample_linear1d(const Tensor & self, IntList output_size, bool align_corners) {
  return at::_thnn_upsample_linear1d(self, output_size, align_corners);
}

Tensor & upsample_linear1d_backward_out(Tensor & grad_input, const Tensor & grad_output, IntList output_size, IntList input_size, bool align_corners) {
  return at::_thnn_upsample_linear1d_backward_out(grad_input, grad_output, output_size, input_size, align_corners);
}

Tensor upsample_linear1d_backward(const Tensor & grad_output, IntList output_size, IntList input_size, bool align_corners) {
  return at::_thnn_upsample_linear1d_backward(grad_output, output_size, input_size, align_corners);
}

Tensor & upsample_bilinear2d_out(Tensor & output, const Tensor & self, IntList output_size, bool align_corners) {
  return at::_thnn_upsample_bilinear2d_out(output, self, output_size, align_corners);
}

Tensor upsample_bilinear2d(const Tensor & self, IntList output_size, bool align_corners) {
  return at::_thnn_upsample_bilinear2d(self, output_size, align_corners);
}

Tensor & upsample_bilinear2d_backward_out(Tensor & grad_input, const Tensor & grad_output, IntList output_size, IntList input_size, bool align_corners) {
  return at::_thnn_upsample_bilinear2d_backward_out(grad_input, grad_output, output_size, input_size, align_corners);
}

Tensor upsample_bilinear2d_backward(const Tensor & grad_output, IntList output_size, IntList input_size, bool align_corners) {
  return at::_thnn_upsample_bilinear2d_backward(grad_output, output_size, input_size, align_corners);
}

Tensor & upsample_trilinear3d_out(Tensor & output, const Tensor & self, IntList output_size, bool align_corners) {
  return at::_thnn_upsample_trilinear3d_out(output, self, output_size, align_corners);
}

Tensor upsample_trilinear3d(const Tensor & self, IntList output_size, bool align_corners) {
  return at::_thnn_upsample_trilinear3d(self, output_size, align_corners);
}

Tensor & upsample_trilinear3d_backward_out(Tensor & grad_input, const Tensor & grad_output, IntList output_size, IntList input_size, bool align_corners) {
  return at::_thnn_upsample_trilinear3d_backward_out(grad_input, grad_output, output_size, input_size, align_corners);
}

Tensor upsample_trilinear3d_backward(const Tensor & grad_output, IntList output_size, IntList input_size, bool align_corners) {
  return at::_thnn_upsample_trilinear3d_backward(grad_output, output_size, input_size, align_corners);
}

Tensor & upsample_nearest1d_out(Tensor & output, const Tensor & self, IntList output_size) {
  return at::_thnn_upsample_nearest1d_out(output, self, output_size);
}

Tensor upsample_nearest1d(const Tensor & self, IntList output_size) {
  return at::_thnn_upsample_nearest1d(self, output_size);
}

Tensor & upsample_nearest1d_backward_out(Tensor & grad_input, const Tensor & grad_output, IntList output_size, IntList input_size) {
  return at::_thnn_upsample_nearest1d_backward_out(grad_input, grad_output, output_size, input_size);
}

Tensor upsample_nearest1d_backward(const Tensor & grad_output, IntList output_size, IntList input_size) {
  return at::_thnn_upsample_nearest1d_backward(grad_output, output_size, input_size);
}

Tensor & upsample_nearest2d_out(Tensor & output, const Tensor & self, IntList output_size) {
  return at::_thnn_upsample_nearest2d_out(output, self, output_size);
}

Tensor upsample_nearest2d(const Tensor & self, IntList output_size) {
  return at::_thnn_upsample_nearest2d(self, output_size);
}

Tensor & upsample_nearest2d_backward_out(Tensor & grad_input, const Tensor & grad_output, IntList output_size, IntList input_size) {
  return at::_thnn_upsample_nearest2d_backward_out(grad_input, grad_output, output_size, input_size);
}

Tensor upsample_nearest2d_backward(const Tensor & grad_output, IntList output_size, IntList input_size) {
  return at::_thnn_upsample_nearest2d_backward(grad_output, output_size, input_size);
}

Tensor & upsample_nearest3d_out(Tensor & output, const Tensor & self, IntList output_size) {
  return at::_thnn_upsample_nearest3d_out(output, self, output_size);
}

Tensor upsample_nearest3d(const Tensor & self, IntList output_size) {
  return at::_thnn_upsample_nearest3d(self, output_size);
}

Tensor & upsample_nearest3d_backward_out(Tensor & grad_input, const Tensor & grad_output, IntList output_size, IntList input_size) {
  return at::_thnn_upsample_nearest3d_backward_out(grad_input, grad_output, output_size, input_size);
}

Tensor upsample_nearest3d_backward(const Tensor & grad_output, IntList output_size, IntList input_size) {
  return at::_thnn_upsample_nearest3d_backward(grad_output, output_size, input_size);
}

}} // namespace at::native
