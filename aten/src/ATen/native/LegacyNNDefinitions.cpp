#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <ATen/LegacyTHFunctions.h>

namespace at { namespace native {

Tensor & binary_cross_entropy_out(Tensor & output, const Tensor & self, const Tensor & target, const Tensor & weight, int64_t reduction) {
  return at::legacy::th::_thnn_binary_cross_entropy_forward_out(output, self, target, weight, reduction);
}
Tensor binary_cross_entropy(const Tensor & self, const Tensor & target, const Tensor & weight, int64_t reduction) {
  return at::legacy::th::_thnn_binary_cross_entropy_forward(self, target, weight, reduction);
}
Tensor & binary_cross_entropy_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, const Tensor & target, const Tensor & weight, int64_t reduction) {
  return at::legacy::th::_thnn_binary_cross_entropy_backward_out(grad_input, grad_output, self, target, weight, reduction);
}

Tensor binary_cross_entropy_backward(const Tensor & grad_output, const Tensor & self, const Tensor & target, const Tensor & weight, int64_t reduction) {
  return at::legacy::th::_thnn_binary_cross_entropy_backward(grad_output, self, target, weight, reduction);
}

Tensor & mse_loss_out(Tensor & output, const Tensor & self, const Tensor & target, int64_t reduction) {
  return at::legacy::th::_thnn_mse_loss_forward_out(output, self, target, reduction);
}

Tensor mse_loss(const Tensor & self, const Tensor & target, int64_t reduction) {
  return at::legacy::th::_thnn_mse_loss_forward(self, target, reduction);
}

Tensor & mse_loss_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, const Tensor & target, int64_t reduction) {
  return at::legacy::th::_thnn_mse_loss_backward_out(grad_input, grad_output, self, target, reduction);
}

Tensor mse_loss_backward(const Tensor & grad_output, const Tensor & self, const Tensor & target, int64_t reduction) {
  return at::legacy::th::_thnn_mse_loss_backward(grad_output, self, target, reduction);
}

Tensor & l1_loss_out(Tensor & output, const Tensor & self, const Tensor & target, int64_t reduction) {
  return at::legacy::th::_thnn_l1_loss_forward_out(output, self, target, reduction);
}

Tensor l1_loss(const Tensor & self, const Tensor & target, int64_t reduction) {
  return at::legacy::th::_thnn_l1_loss_forward(self, target, reduction);
}

Tensor & l1_loss_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, const Tensor & target, int64_t reduction) {
  return at::legacy::th::_thnn_l1_loss_backward_out(grad_input, grad_output, self, target, reduction);
}

Tensor l1_loss_backward(const Tensor & grad_output, const Tensor & self, const Tensor & target, int64_t reduction) {
  return at::legacy::th::_thnn_l1_loss_backward(grad_output, self, target, reduction);
}

Tensor & multi_margin_loss_out(Tensor & output, const Tensor & self, const Tensor & target,
                                     Scalar p, Scalar margin, const Tensor & weight, int64_t reduction) {
  return at::legacy::th::_thnn_multi_margin_loss_forward_out(output, self, target, p, margin, weight, reduction);
}

Tensor multi_margin_loss(const Tensor & self, const Tensor & target,
                               Scalar p, Scalar margin, const Tensor & weight, int64_t reduction) {
  return at::legacy::th::_thnn_multi_margin_loss_forward(self, target, p, margin, weight, reduction);
}

Tensor & multi_margin_loss_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, const Tensor & target,
                                        Scalar p, Scalar margin, const Tensor & weight, int64_t reduction) {
  return at::legacy::th::_thnn_multi_margin_loss_backward_out(grad_input, grad_output, self, target, p, margin, weight, reduction);
}

Tensor multi_margin_loss_backward(const Tensor & grad_output, const Tensor & self, const Tensor & target,
                                  Scalar p, Scalar margin, const Tensor & weight, int64_t reduction) {
  return at::legacy::th::_thnn_multi_margin_loss_backward(grad_output, self, target, p, margin, weight, reduction);
}

Tensor & multilabel_margin_loss_out(Tensor & output, const Tensor & self, const Tensor & target, int64_t reduction) {
  Tensor is_target = at::empty({0}, self.options());
  return std::get<0>(at::multilabel_margin_loss_forward_out(output, is_target, self, target, reduction));
}

Tensor multilabel_margin_loss(const Tensor & self, const Tensor & target, int64_t reduction) {
  return std::get<0>(at::multilabel_margin_loss_forward(self, target, reduction));
}

std::tuple<Tensor &,Tensor &> multilabel_margin_loss_forward_out(Tensor & output, Tensor & is_target, const Tensor & self, const Tensor & target, int64_t reduction) {
  return at::legacy::th::_thnn_multilabel_margin_loss_forward_out(output, is_target, self, target, reduction);
}

std::tuple<Tensor,Tensor> multilabel_margin_loss_forward(const Tensor & self, const Tensor & target, int64_t reduction) {
  return at::legacy::th::_thnn_multilabel_margin_loss_forward(self, target, reduction);
}

Tensor & multilabel_margin_loss_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, const Tensor & target, int64_t reduction, const Tensor & is_target) {
  return at::legacy::th::_thnn_multilabel_margin_loss_backward_out(grad_input, grad_output, self, target, reduction, is_target);
}

Tensor multilabel_margin_loss_backward(const Tensor & grad_output, const Tensor & self, const Tensor & target, int64_t reduction, const Tensor & is_target) {
  return at::legacy::th::_thnn_multilabel_margin_loss_backward(grad_output, self, target, reduction, is_target);
}

Tensor & nll_loss_out(Tensor & output, const Tensor & self, const Tensor & target, const Tensor & weight, int64_t reduction, int64_t ignore_index) {
  Tensor total_weight = at::empty({0}, self.options());
  return std::get<0>(at::nll_loss_forward_out(output, total_weight, self, target, weight, reduction, ignore_index));
}

Tensor nll_loss(const Tensor & self, const Tensor & target, const Tensor & weight, int64_t reduction, int64_t ignore_index) {
  return std::get<0>(at::nll_loss_forward(self, target, weight, reduction, ignore_index));
}

std::tuple<Tensor &,Tensor &> nll_loss_forward_out(Tensor & output, Tensor & total_weight, const Tensor & self, const Tensor & target, const Tensor & weight, int64_t reduction, int64_t ignore_index) {
  return at::legacy::th::_thnn_nll_loss_forward_out(output, total_weight, self, target, weight, reduction, ignore_index);
}

std::tuple<Tensor,Tensor> nll_loss_forward(const Tensor & self, const Tensor & target, const Tensor & weight, int64_t reduction, int64_t ignore_index) {
  return at::legacy::th::_thnn_nll_loss_forward(self, target, weight, reduction, ignore_index);
}

Tensor & nll_loss_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, const Tensor & target, const Tensor & weight, int64_t reduction, int64_t ignore_index, const Tensor & total_weight) {
  return at::legacy::th::_thnn_nll_loss_backward_out(grad_input, grad_output, self, target, weight, reduction, ignore_index, total_weight);
}

Tensor nll_loss_backward(const Tensor & grad_output, const Tensor & self, const Tensor & target, const Tensor & weight, int64_t reduction, int64_t ignore_index, const Tensor & total_weight) {
  return at::legacy::th::_thnn_nll_loss_backward(grad_output, self, target, weight, reduction, ignore_index, total_weight);
}

Tensor & nll_loss2d_out(Tensor & output, const Tensor & self, const Tensor & target, const Tensor & weight, int64_t reduction, int64_t ignore_index) {
  Tensor total_weight = at::empty({0}, self.options());
  return std::get<0>(at::nll_loss2d_forward_out(output, total_weight, self, target, weight, reduction, ignore_index));
}

Tensor nll_loss2d(const Tensor & self, const Tensor & target, const Tensor & weight, int64_t reduction, int64_t ignore_index) {
  return std::get<0>(at::nll_loss2d_forward(self, target, weight, reduction, ignore_index));
}

std::tuple<Tensor &,Tensor &> nll_loss2d_forward_out(Tensor & output, Tensor & total_weight, const Tensor & self, const Tensor & target, const Tensor & weight, int64_t reduction, int64_t ignore_index) {
  return at::legacy::th::_thnn_nll_loss2d_forward_out(output, total_weight, self, target, weight, reduction, ignore_index);
}

std::tuple<Tensor,Tensor> nll_loss2d_forward(const Tensor & self, const Tensor & target, const Tensor & weight, int64_t reduction, int64_t ignore_index) {
  return at::legacy::th::_thnn_nll_loss2d_forward(self, target, weight, reduction, ignore_index);
}

Tensor & nll_loss2d_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, const Tensor & target, const Tensor & weight, int64_t reduction, int64_t ignore_index, const Tensor & total_weight) {
  return at::legacy::th::_thnn_nll_loss2d_backward_out(grad_input, grad_output, self, target, weight, reduction, ignore_index, total_weight);
}

Tensor nll_loss2d_backward(const Tensor & grad_output, const Tensor & self, const Tensor & target, const Tensor & weight, int64_t reduction, int64_t ignore_index, const Tensor & total_weight) {
  return at::legacy::th::_thnn_nll_loss2d_backward(grad_output, self, target, weight, reduction, ignore_index, total_weight);
}

Tensor & smooth_l1_loss_out(Tensor & output, const Tensor & self, const Tensor & target, int64_t reduction) {
  return at::legacy::th::_thnn_smooth_l1_loss_forward_out(output, self, target, reduction);
}

Tensor smooth_l1_loss(const Tensor & self, const Tensor & target, int64_t reduction) {
  return at::legacy::th::_thnn_smooth_l1_loss_forward(self, target, reduction);
}

Tensor & smooth_l1_loss_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self,
                                     const Tensor & target, int64_t reduction) {
  return at::legacy::th::_thnn_smooth_l1_loss_backward_out(grad_input, grad_output, self, target, reduction);
}

Tensor smooth_l1_loss_backward(const Tensor & grad_output, const Tensor & self, const Tensor & target, int64_t reduction) {
  return at::legacy::th::_thnn_smooth_l1_loss_backward(grad_output, self, target, reduction);
}

Tensor & soft_margin_loss_out(Tensor & output, const Tensor & self, const Tensor & target, int64_t reduction) {
  return at::legacy::th::_thnn_soft_margin_loss_forward_out(output, self, target, reduction);
}

Tensor soft_margin_loss(const Tensor & self, const Tensor & target, int64_t reduction) {
  return at::legacy::th::_thnn_soft_margin_loss_forward(self, target, reduction);
}

Tensor & soft_margin_loss_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self,
                                       const Tensor & target, int64_t reduction) {
  return at::legacy::th::_thnn_soft_margin_loss_backward_out(grad_input, grad_output, self, target, reduction);
}

Tensor soft_margin_loss_backward(const Tensor & grad_output, const Tensor & self, const Tensor & target, int64_t reduction) {
  return at::legacy::th::_thnn_soft_margin_loss_backward(grad_output, self, target, reduction);
}

Tensor & elu_out(Tensor & output, const Tensor & self, Scalar alpha, Scalar scale, Scalar input_scale) {
  return at::legacy::th::_thnn_elu_forward_out(output, self, alpha, scale, input_scale);
}

Tensor elu(const Tensor & self, Scalar alpha, Scalar scale, Scalar input_scale) {
  return at::legacy::th::_thnn_elu_forward(self, alpha, scale, input_scale);
}

Tensor & elu_backward_out(Tensor & grad_input, const Tensor & grad_output, Scalar alpha, Scalar scale, Scalar input_scale, const Tensor & output) {
  return at::legacy::th::_thnn_elu_backward_out(grad_input, grad_output, alpha, scale, input_scale, output);
}

Tensor elu_backward(const Tensor & grad_output, Scalar alpha, Scalar scale, Scalar input_scale, const Tensor & output) {
  return at::legacy::th::_thnn_elu_backward(grad_output, alpha, scale, input_scale, output);
}

Tensor & elu_(Tensor & self, Scalar alpha, Scalar scale, Scalar input_scale) {
  return at::legacy::th::_thnn_elu_forward_(self, alpha, scale, input_scale);
}

Tensor & glu_out(Tensor & output, const Tensor & self, int64_t dim) {
  return at::legacy::th::_thnn_glu_forward_out(output, self, dim);
}

Tensor glu(const Tensor & self, int64_t dim) {
  return at::legacy::th::_thnn_glu_forward(self, dim);
}

Tensor & glu_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, int64_t dim) {
  return at::legacy::th::_thnn_glu_backward_out(grad_input, grad_output, self, dim);
}

Tensor glu_backward(const Tensor & grad_output, const Tensor & self, int64_t dim) {
  return at::legacy::th::_thnn_glu_backward(grad_output, self, dim);
}

Tensor & hardtanh_out(Tensor & output, const Tensor & self, Scalar min_val, Scalar max_val) {
  return at::legacy::th::_thnn_hardtanh_forward_out(output, self, min_val, max_val);
}

Tensor hardtanh(const Tensor & self, Scalar min_val, Scalar max_val) {
  return at::legacy::th::_thnn_hardtanh_forward(self, min_val, max_val);
}

Tensor & hardtanh_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, Scalar min_val, Scalar max_val) {
  return at::legacy::th::_thnn_hardtanh_backward_out(grad_input, grad_output, self, min_val, max_val);
}

Tensor hardtanh_backward(const Tensor & grad_output, const Tensor & self, Scalar min_val, Scalar max_val) {
  return at::legacy::th::_thnn_hardtanh_backward(grad_output, self, min_val, max_val);
}

Tensor & hardtanh_(Tensor & self, Scalar min_val, Scalar max_val) {
  return at::legacy::th::_thnn_hardtanh_forward_(self, min_val, max_val);
}

Tensor & leaky_relu_out(Tensor & output, const Tensor & self, Scalar negative_slope) {
  return at::legacy::th::_thnn_leaky_relu_forward_out(output, self, negative_slope);
}

Tensor leaky_relu(const Tensor & self, Scalar negative_slope) {
  return at::legacy::th::_thnn_leaky_relu_forward(self, negative_slope);
}

Tensor & leaky_relu_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, Scalar negative_slope) {
  return at::legacy::th::_thnn_leaky_relu_backward_out(grad_input, grad_output, self, negative_slope);
}

Tensor leaky_relu_backward(const Tensor & grad_output, const Tensor & self, Scalar negative_slope) {
  return at::legacy::th::_thnn_leaky_relu_backward(grad_output, self, negative_slope);
}

Tensor & leaky_relu_(Tensor & self, Scalar negative_slope) {
  return at::legacy::th::_thnn_leaky_relu_forward_(self, negative_slope);
}

Tensor & log_sigmoid_out(Tensor & output, const Tensor & self) {
  Tensor buffer = at::empty({0}, self.options());
  return std::get<0>(at::legacy::th::_thnn_log_sigmoid_forward_out(output, buffer, self));
}

Tensor log_sigmoid(const Tensor & self) {
  return std::get<0>(at::log_sigmoid_forward(self));
}

std::tuple<Tensor &,Tensor &> log_sigmoid_forward_out(Tensor & output, Tensor & buffer, const Tensor & self) {
  return at::legacy::th::_thnn_log_sigmoid_forward_out(output, buffer, self);
}

std::tuple<Tensor,Tensor> log_sigmoid_forward(const Tensor & self) {
  return at::legacy::th::_thnn_log_sigmoid_forward(self);
}

Tensor & log_sigmoid_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, const Tensor & buffer) {
  return at::legacy::th::_thnn_log_sigmoid_backward_out(grad_input, grad_output, self, buffer);
}

Tensor log_sigmoid_backward(const Tensor & grad_output, const Tensor & self, const Tensor & buffer) {
  return at::legacy::th::_thnn_log_sigmoid_backward(grad_output, self, buffer);
}

Tensor & rrelu_with_noise_out(Tensor & output, const Tensor & self, const Tensor & noise, Scalar lower, Scalar upper, bool training, Generator * generator) {
  return at::legacy::th::_thnn_rrelu_with_noise_forward_out(output, self, noise, lower, upper, training, generator);
}

Tensor rrelu_with_noise(const Tensor & self, const Tensor & noise, Scalar lower, Scalar upper, bool training, Generator * generator) {
  return at::legacy::th::_thnn_rrelu_with_noise_forward(self, noise, lower, upper, training, generator);
}

Tensor & rrelu_with_noise_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, const Tensor & noise, Scalar lower, Scalar upper, bool training) {
  return at::legacy::th::_thnn_rrelu_with_noise_backward_out(grad_input, grad_output, self, noise, lower, upper, training);
}

Tensor rrelu_with_noise_backward(const Tensor & grad_output, const Tensor & self, const Tensor & noise, Scalar lower, Scalar upper, bool training) {
  return at::legacy::th::_thnn_rrelu_with_noise_backward(grad_output, self, noise, lower, upper, training);
}

Tensor & rrelu_with_noise_(Tensor & self, const Tensor & noise, Scalar lower, Scalar upper, bool training, Generator * generator) {
  return at::legacy::th::_thnn_rrelu_with_noise_forward_(self, noise, lower, upper, training, generator);
}

Tensor & softplus_out(Tensor & output, const Tensor & self, Scalar beta, Scalar threshold) {
  return at::legacy::th::_thnn_softplus_forward_out(output, self, beta, threshold);
}

Tensor softplus(const Tensor & self, Scalar beta, Scalar threshold) {
  return at::legacy::th::_thnn_softplus_forward(self, beta, threshold);
}

Tensor & softplus_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, Scalar beta, Scalar threshold, const Tensor & output) {
  return at::legacy::th::_thnn_softplus_backward_out(grad_input, grad_output, self, beta, threshold, output);
}

Tensor softplus_backward(const Tensor & grad_output, const Tensor & self, Scalar beta, Scalar threshold, const Tensor & output) {
  return at::legacy::th::_thnn_softplus_backward(grad_output, self, beta, threshold, output);
}

Tensor & softshrink_out(Tensor & output, const Tensor & self, Scalar lambd) {
  return at::legacy::th::_thnn_softshrink_forward_out(output, self, lambd);
}

Tensor softshrink(const Tensor & self, Scalar lambd) {
  return at::legacy::th::_thnn_softshrink_forward(self, lambd);
}

Tensor & softshrink_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, Scalar lambd) {
  return at::legacy::th::_thnn_softshrink_backward_out(grad_input, grad_output, self, lambd);
}

Tensor softshrink_backward(const Tensor & grad_output, const Tensor & self, Scalar lambd) {
  return at::legacy::th::_thnn_softshrink_backward(grad_output, self, lambd);
}

Tensor & adaptive_avg_pool3d_out(Tensor & output, const Tensor & self, IntArrayRef output_size) {
  return at::legacy::th::_thnn_adaptive_avg_pool3d_forward_out(output, self, output_size);
}

Tensor adaptive_avg_pool3d(const Tensor & self, IntArrayRef output_size) {
  return at::legacy::th::_thnn_adaptive_avg_pool3d_forward(self, output_size);
}

Tensor & adaptive_avg_pool3d_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self) {
  return at::legacy::th::_thnn_adaptive_avg_pool3d_backward_out(grad_input, grad_output, self);
}

Tensor adaptive_avg_pool3d_backward(const Tensor & grad_output, const Tensor & self) {
  return at::legacy::th::_thnn_adaptive_avg_pool3d_backward(grad_output, self);
}

std::tuple<Tensor &,Tensor &> adaptive_max_pool2d_out(Tensor & output, Tensor & indices, const Tensor & self, IntArrayRef output_size) {
  return at::legacy::th::_thnn_adaptive_max_pool2d_forward_out(output, indices, self, output_size);
}

std::tuple<Tensor,Tensor> adaptive_max_pool2d(const Tensor & self, IntArrayRef output_size) {
  return at::legacy::th::_thnn_adaptive_max_pool2d_forward(self, output_size);
}

Tensor & adaptive_max_pool2d_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, const Tensor & indices) {
  return at::legacy::th::_thnn_adaptive_max_pool2d_backward_out(grad_input, grad_output, self, indices);
}

Tensor adaptive_max_pool2d_backward(const Tensor & grad_output, const Tensor & self, const Tensor & indices) {
  return at::legacy::th::_thnn_adaptive_max_pool2d_backward(grad_output, self, indices);
}

std::tuple<Tensor &,Tensor &> adaptive_max_pool3d_out(Tensor & output, Tensor & indices, const Tensor & self, IntArrayRef output_size) {
  return at::legacy::th::_thnn_adaptive_max_pool3d_forward_out(output, indices, self, output_size);
}

std::tuple<Tensor,Tensor> adaptive_max_pool3d(const Tensor & self, IntArrayRef output_size) {
  return at::legacy::th::_thnn_adaptive_max_pool3d_forward(self, output_size);
}

Tensor & adaptive_max_pool3d_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, const Tensor & indices) {
  return at::legacy::th::_thnn_adaptive_max_pool3d_backward_out(grad_input, grad_output, self, indices);
}

Tensor adaptive_max_pool3d_backward(const Tensor & grad_output, const Tensor & self, const Tensor & indices) {
  return at::legacy::th::_thnn_adaptive_max_pool3d_backward(grad_output, self, indices);
}

Tensor & avg_pool2d_out(Tensor & output, const Tensor & self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, bool ceil_mode, bool count_include_pad) {
  return at::legacy::th::_thnn_avg_pool2d_forward_out(output, self, kernel_size, stride, padding, ceil_mode, count_include_pad);
}

Tensor avg_pool2d(const Tensor & self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, bool ceil_mode, bool count_include_pad) {
  return at::legacy::th::_thnn_avg_pool2d_forward(self, kernel_size, stride, padding, ceil_mode, count_include_pad);
}

Tensor & avg_pool2d_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, bool ceil_mode, bool count_include_pad) {
  return at::legacy::th::_thnn_avg_pool2d_backward_out(grad_input, grad_output, self, kernel_size, stride, padding, ceil_mode, count_include_pad);
}

Tensor avg_pool2d_backward(const Tensor & grad_output, const Tensor & self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, bool ceil_mode, bool count_include_pad) {
  return at::legacy::th::_thnn_avg_pool2d_backward(grad_output, self, kernel_size, stride, padding, ceil_mode, count_include_pad);
}

Tensor & avg_pool3d_out(Tensor & output, const Tensor & self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, bool ceil_mode, bool count_include_pad) {
  return at::legacy::th::_thnn_avg_pool3d_forward_out(output, self, kernel_size, stride, padding, ceil_mode, count_include_pad);
}

Tensor avg_pool3d(const Tensor & self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, bool ceil_mode, bool count_include_pad) {
  return at::legacy::th::_thnn_avg_pool3d_forward(self, kernel_size, stride, padding, ceil_mode, count_include_pad);
}

Tensor & avg_pool3d_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, bool ceil_mode, bool count_include_pad) {
  return at::legacy::th::_thnn_avg_pool3d_backward_out(grad_input, grad_output, self, kernel_size, stride, padding, ceil_mode, count_include_pad);
}

Tensor avg_pool3d_backward(const Tensor & grad_output, const Tensor & self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, bool ceil_mode, bool count_include_pad) {
  return at::legacy::th::_thnn_avg_pool3d_backward(grad_output, self, kernel_size, stride, padding, ceil_mode, count_include_pad);
}

std::tuple<Tensor &,Tensor &> max_pool2d_with_indices_out(Tensor & output, Tensor & indices, const Tensor & self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, bool ceil_mode) {
  return at::legacy::th::_thnn_max_pool2d_with_indices_forward_out(output, indices, self, kernel_size, stride, padding, dilation, ceil_mode);
}

std::tuple<Tensor,Tensor> max_pool2d_with_indices(const Tensor & self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, bool ceil_mode) {
  return at::legacy::th::_thnn_max_pool2d_with_indices_forward(self, kernel_size, stride, padding, dilation, ceil_mode);
}

Tensor & max_pool2d_with_indices_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, bool ceil_mode, const Tensor & indices) {
  return at::legacy::th::_thnn_max_pool2d_with_indices_backward_out(grad_input, grad_output, self, kernel_size, stride, padding, dilation, ceil_mode, indices);
}

Tensor max_pool2d_with_indices_backward(const Tensor & grad_output, const Tensor & self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, bool ceil_mode, const Tensor & indices) {
  return at::legacy::th::_thnn_max_pool2d_with_indices_backward(grad_output, self, kernel_size, stride, padding, dilation, ceil_mode, indices);
}

std::tuple<Tensor &,Tensor &> max_pool3d_with_indices_out(Tensor & output, Tensor & indices, const Tensor & self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, bool ceil_mode) {
  return at::legacy::th::_thnn_max_pool3d_with_indices_forward_out(output, indices, self, kernel_size, stride, padding, dilation, ceil_mode);
}

std::tuple<Tensor,Tensor> max_pool3d_with_indices(const Tensor & self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, bool ceil_mode) {
  return at::legacy::th::_thnn_max_pool3d_with_indices_forward(self, kernel_size, stride, padding, dilation, ceil_mode);
}

Tensor & max_pool3d_with_indices_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, bool ceil_mode, const Tensor & indices) {
  return at::legacy::th::_thnn_max_pool3d_with_indices_backward_out(grad_input, grad_output, self, kernel_size, stride, padding, dilation, ceil_mode, indices);
}

Tensor max_pool3d_with_indices_backward(const Tensor & grad_output, const Tensor & self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, bool ceil_mode, const Tensor & indices) {
  return at::legacy::th::_thnn_max_pool3d_with_indices_backward(grad_output, self, kernel_size, stride, padding, dilation, ceil_mode, indices);
}

Tensor & max_unpool2d_out(Tensor & output, const Tensor & self, const Tensor & indices, IntArrayRef output_size) {
  return at::legacy::th::_thnn_max_unpool2d_forward_out(output, self, indices, output_size);
}

Tensor max_unpool2d(const Tensor & self, const Tensor & indices, IntArrayRef output_size) {
  return at::legacy::th::_thnn_max_unpool2d_forward(self, indices, output_size);
}

Tensor & max_unpool2d_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, const Tensor & indices, IntArrayRef output_size) {
  return at::legacy::th::_thnn_max_unpool2d_backward_out(grad_input, grad_output, self, indices, output_size);
}

Tensor max_unpool2d_backward(const Tensor & grad_output, const Tensor & self, const Tensor & indices, IntArrayRef output_size) {
  return at::legacy::th::_thnn_max_unpool2d_backward(grad_output, self, indices, output_size);
}

Tensor & max_unpool3d_out(Tensor & output, const Tensor & self, const Tensor & indices, IntArrayRef output_size, IntArrayRef stride, IntArrayRef padding) {
  return at::legacy::th::_thnn_max_unpool3d_forward_out(output, self, indices, output_size, stride, padding);
}

Tensor max_unpool3d(const Tensor & self, const Tensor & indices, IntArrayRef output_size, IntArrayRef stride, IntArrayRef padding) {
  return at::legacy::th::_thnn_max_unpool3d_forward(self, indices, output_size, stride, padding);
}

Tensor & max_unpool3d_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, const Tensor & indices, IntArrayRef output_size, IntArrayRef stride, IntArrayRef padding) {
  return at::legacy::th::_thnn_max_unpool3d_backward_out(grad_input, grad_output, self, indices, output_size, stride, padding);
}

Tensor max_unpool3d_backward(const Tensor & grad_output, const Tensor & self, const Tensor & indices, IntArrayRef output_size, IntArrayRef stride, IntArrayRef padding) {
  return at::legacy::th::_thnn_max_unpool3d_backward(grad_output, self, indices, output_size, stride, padding);
}

Tensor & upsample_linear1d_out(Tensor & output, const Tensor & self, IntArrayRef output_size, bool align_corners) {
  return at::legacy::th::_thnn_upsample_linear1d_forward_out(output, self, output_size, align_corners);
}

Tensor upsample_linear1d(const Tensor & self, IntArrayRef output_size, bool align_corners) {
  return at::legacy::th::_thnn_upsample_linear1d_forward(self, output_size, align_corners);
}

Tensor & upsample_linear1d_backward_out(Tensor & grad_input, const Tensor & grad_output, IntArrayRef output_size, IntArrayRef input_size, bool align_corners) {
  return at::legacy::th::_thnn_upsample_linear1d_backward_out(grad_input, grad_output, output_size, input_size, align_corners);
}

Tensor upsample_linear1d_backward(const Tensor & grad_output, IntArrayRef output_size, IntArrayRef input_size, bool align_corners) {
  return at::legacy::th::_thnn_upsample_linear1d_backward(grad_output, output_size, input_size, align_corners);
}

Tensor & upsample_bilinear2d_out(Tensor & output, const Tensor & self, IntArrayRef output_size, bool align_corners) {
  return at::legacy::th::_thnn_upsample_bilinear2d_forward_out(output, self, output_size, align_corners);
}

Tensor upsample_bilinear2d(const Tensor & self, IntArrayRef output_size, bool align_corners) {
  return at::legacy::th::_thnn_upsample_bilinear2d_forward(self, output_size, align_corners);
}

Tensor & upsample_bilinear2d_backward_out(Tensor & grad_input, const Tensor & grad_output, IntArrayRef output_size, IntArrayRef input_size, bool align_corners) {
  return at::legacy::th::_thnn_upsample_bilinear2d_backward_out(grad_input, grad_output, output_size, input_size, align_corners);
}

Tensor upsample_bilinear2d_backward(const Tensor & grad_output, IntArrayRef output_size, IntArrayRef input_size, bool align_corners) {
  return at::legacy::th::_thnn_upsample_bilinear2d_backward(grad_output, output_size, input_size, align_corners);
}

Tensor & upsample_bicubic2d_out(Tensor & output, const Tensor & self, IntArrayRef output_size, bool align_corners) {
  return at::legacy::th::_thnn_upsample_bicubic2d_forward_out(output, self, output_size, align_corners);
}

Tensor upsample_bicubic2d(const Tensor & self, IntArrayRef output_size, bool align_corners) {
  return at::legacy::th::_thnn_upsample_bicubic2d_forward(self, output_size, align_corners);
}

Tensor & upsample_bicubic2d_backward_out(Tensor & grad_input, const Tensor & grad_output, IntArrayRef output_size, IntArrayRef input_size, bool align_corners) {
  return at::legacy::th::_thnn_upsample_bicubic2d_backward_out(grad_input, grad_output, output_size, input_size, align_corners);
}

Tensor upsample_bicubic2d_backward(const Tensor & grad_output, IntArrayRef output_size, IntArrayRef input_size, bool align_corners) {
  return at::legacy::th::_thnn_upsample_bicubic2d_backward(grad_output, output_size, input_size, align_corners);
}

Tensor & upsample_trilinear3d_out(Tensor & output, const Tensor & self, IntArrayRef output_size, bool align_corners) {
  return at::legacy::th::_thnn_upsample_trilinear3d_forward_out(output, self, output_size, align_corners);
}

Tensor upsample_trilinear3d(const Tensor & self, IntArrayRef output_size, bool align_corners) {
  return at::legacy::th::_thnn_upsample_trilinear3d_forward(self, output_size, align_corners);
}

Tensor & upsample_trilinear3d_backward_out(Tensor & grad_input, const Tensor & grad_output, IntArrayRef output_size, IntArrayRef input_size, bool align_corners) {
  return at::legacy::th::_thnn_upsample_trilinear3d_backward_out(grad_input, grad_output, output_size, input_size, align_corners);
}

Tensor upsample_trilinear3d_backward(const Tensor & grad_output, IntArrayRef output_size, IntArrayRef input_size, bool align_corners) {
  return at::legacy::th::_thnn_upsample_trilinear3d_backward(grad_output, output_size, input_size, align_corners);
}

Tensor & upsample_nearest1d_out(Tensor & output, const Tensor & self, IntArrayRef output_size) {
  return at::legacy::th::_thnn_upsample_nearest1d_forward_out(output, self, output_size);
}

Tensor upsample_nearest1d(const Tensor & self, IntArrayRef output_size) {
  return at::legacy::th::_thnn_upsample_nearest1d_forward(self, output_size);
}

Tensor & upsample_nearest1d_backward_out(Tensor & grad_input, const Tensor & grad_output, IntArrayRef output_size, IntArrayRef input_size) {
  return at::legacy::th::_thnn_upsample_nearest1d_backward_out(grad_input, grad_output, output_size, input_size);
}

Tensor upsample_nearest1d_backward(const Tensor & grad_output, IntArrayRef output_size, IntArrayRef input_size) {
  return at::legacy::th::_thnn_upsample_nearest1d_backward(grad_output, output_size, input_size);
}

Tensor & upsample_nearest2d_out(Tensor & output, const Tensor & self, IntArrayRef output_size) {
  return at::legacy::th::_thnn_upsample_nearest2d_forward_out(output, self, output_size);
}

Tensor upsample_nearest2d(const Tensor & self, IntArrayRef output_size) {
  return at::legacy::th::_thnn_upsample_nearest2d_forward(self, output_size);
}

Tensor & upsample_nearest2d_backward_out(Tensor & grad_input, const Tensor & grad_output, IntArrayRef output_size, IntArrayRef input_size) {
  return at::legacy::th::_thnn_upsample_nearest2d_backward_out(grad_input, grad_output, output_size, input_size);
}

Tensor upsample_nearest2d_backward(const Tensor & grad_output, IntArrayRef output_size, IntArrayRef input_size) {
  return at::legacy::th::_thnn_upsample_nearest2d_backward(grad_output, output_size, input_size);
}

Tensor & upsample_nearest3d_out(Tensor & output, const Tensor & self, IntArrayRef output_size) {
  return at::legacy::th::_thnn_upsample_nearest3d_forward_out(output, self, output_size);
}

Tensor upsample_nearest3d(const Tensor & self, IntArrayRef output_size) {
  return at::legacy::th::_thnn_upsample_nearest3d_forward(self, output_size);
}

Tensor & upsample_nearest3d_backward_out(Tensor & grad_input, const Tensor & grad_output, IntArrayRef output_size, IntArrayRef input_size) {
  return at::legacy::th::_thnn_upsample_nearest3d_backward_out(grad_input, grad_output, output_size, input_size);
}

Tensor upsample_nearest3d_backward(const Tensor & grad_output, IntArrayRef output_size, IntArrayRef input_size) {
  return at::legacy::th::_thnn_upsample_nearest3d_backward(grad_output, output_size, input_size);
}

Tensor & sigmoid_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & output) {
  return at::legacy::th::_thnn_sigmoid_backward_out(grad_input, grad_output, output);
}

Tensor sigmoid_backward(const Tensor & grad_output, const Tensor & output) {
  return at::legacy::th::_thnn_sigmoid_backward(grad_output, output);
}

Tensor & tanh_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & output) {
  return at::legacy::th::_thnn_tanh_backward_out(grad_input, grad_output, output);
}

Tensor tanh_backward(const Tensor & grad_output, const Tensor & output) {
  return at::legacy::th::_thnn_tanh_backward(grad_output, output);
}

Tensor & thnn_conv_transpose2d_out(Tensor & output, const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, const Tensor & bias, IntArrayRef stride, IntArrayRef padding, IntArrayRef output_padding, IntArrayRef dilation) {
  Tensor columns = at::empty({0}, self.options());
  Tensor ones = at::empty({0}, self.options());
  return std::get<0>(at::thnn_conv_transpose2d_forward_out(output, columns, ones, self, weight, kernel_size, bias, stride, padding, output_padding, dilation));
}

Tensor thnn_conv_transpose2d(const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, const Tensor & bias, IntArrayRef stride, IntArrayRef padding, IntArrayRef output_padding, IntArrayRef dilation) {
  return std::get<0>(at::thnn_conv_transpose2d_forward(self, weight, kernel_size, bias, stride, padding, output_padding, dilation));
}

std::tuple<Tensor &,Tensor &,Tensor &> thnn_conv_transpose2d_forward_out(Tensor & output, Tensor & columns, Tensor & ones, const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, const Tensor & bias, IntArrayRef stride, IntArrayRef padding, IntArrayRef output_padding, IntArrayRef dilation) {
  return at::legacy::th::_thnn_conv_transpose2d_forward_out(output, columns, ones, self, weight, kernel_size, bias, stride, padding, output_padding, dilation);
}

std::tuple<Tensor,Tensor,Tensor> thnn_conv_transpose2d_forward(const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, const Tensor & bias, IntArrayRef stride, IntArrayRef padding, IntArrayRef output_padding, IntArrayRef dilation) {
  return at::legacy::th::_thnn_conv_transpose2d_forward(self, weight, kernel_size, bias, stride, padding, output_padding, dilation);
}

std::tuple<Tensor &,Tensor &,Tensor &> thnn_conv_transpose2d_backward_out(Tensor & grad_input, Tensor & grad_weight, Tensor & grad_bias, const Tensor & grad_output, const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, IntArrayRef output_padding, IntArrayRef dilation, const Tensor & columns, const Tensor & ones) {
  return at::legacy::th::_thnn_conv_transpose2d_backward_out(grad_input, grad_weight, grad_bias, grad_output, self, weight, kernel_size, stride, padding, output_padding, dilation, columns, ones);
}

std::tuple<Tensor,Tensor,Tensor> thnn_conv_transpose2d_backward(const Tensor & grad_output, const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, IntArrayRef output_padding, IntArrayRef dilation, const Tensor & columns, const Tensor & ones, std::array<bool,3> output_mask) {
  return at::legacy::th::_thnn_conv_transpose2d_backward(grad_output, self, weight, kernel_size, stride, padding, output_padding, dilation, columns, ones, output_mask);
}

Tensor & thnn_conv_transpose3d_out(Tensor & output, const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, const Tensor & bias, IntArrayRef stride, IntArrayRef padding, IntArrayRef output_padding, IntArrayRef dilation) {
  Tensor finput = at::empty({0}, self.options());
  Tensor fgrad_input = at::empty({0}, self.options());
  return std::get<0>(at::thnn_conv_transpose3d_forward_out(output, finput, fgrad_input, self, weight, kernel_size, bias, stride, padding, output_padding, dilation));
}

Tensor thnn_conv_transpose3d(const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, const Tensor & bias, IntArrayRef stride, IntArrayRef padding, IntArrayRef output_padding, IntArrayRef dilation) {
  return std::get<0>(at::thnn_conv_transpose3d_forward(self, weight, kernel_size, bias, stride, padding, output_padding, dilation));
}

std::tuple<Tensor &,Tensor &,Tensor &> thnn_conv_transpose3d_forward_out(Tensor & output, Tensor & finput, Tensor & fgrad_input, const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, const Tensor & bias, IntArrayRef stride, IntArrayRef padding, IntArrayRef output_padding, IntArrayRef dilation) {
  return at::legacy::th::_thnn_conv_transpose3d_forward_out(output, finput, fgrad_input, self, weight, kernel_size, bias, stride, padding, output_padding, dilation);
}

std::tuple<Tensor,Tensor,Tensor> thnn_conv_transpose3d_forward(const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, const Tensor & bias, IntArrayRef stride, IntArrayRef padding, IntArrayRef output_padding, IntArrayRef dilation) {
  return at::legacy::th::_thnn_conv_transpose3d_forward(self, weight, kernel_size, bias, stride, padding, output_padding, dilation);
}

std::tuple<Tensor &,Tensor &,Tensor &> thnn_conv_transpose3d_backward_out(Tensor & grad_input, Tensor & grad_weight, Tensor & grad_bias, const Tensor & grad_output, const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, IntArrayRef output_padding, IntArrayRef dilation, const Tensor & finput, const Tensor & fgrad_input) {
  return at::legacy::th::_thnn_conv_transpose3d_backward_out(grad_input, grad_weight, grad_bias, grad_output, self, weight, kernel_size, stride, padding, output_padding, dilation, finput, fgrad_input);
}

std::tuple<Tensor,Tensor,Tensor> thnn_conv_transpose3d_backward(const Tensor & grad_output, const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, IntArrayRef output_padding, IntArrayRef dilation, const Tensor & finput, const Tensor & fgrad_input, std::array<bool,3> output_mask) {
  return at::legacy::th::_thnn_conv_transpose3d_backward(grad_output, self, weight, kernel_size, stride, padding, output_padding, dilation, finput, fgrad_input, output_mask);
}

Tensor & thnn_conv2d_out(Tensor & output, const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, const Tensor & bias, IntArrayRef stride, IntArrayRef padding) {
  Tensor finput = at::empty({0}, self.options());
  Tensor fgrad_input = at::empty({0}, self.options());
  return std::get<0>(at::thnn_conv2d_forward_out(output, finput, fgrad_input, self, weight, kernel_size, bias, stride, padding));
}

Tensor thnn_conv2d(const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, const Tensor & bias, IntArrayRef stride, IntArrayRef padding) {
  return std::get<0>(at::thnn_conv2d_forward(self, weight, kernel_size, bias, stride, padding));
}

std::tuple<Tensor &,Tensor &,Tensor &> thnn_conv2d_forward_out(Tensor & output, Tensor & finput, Tensor & fgrad_input, const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, const Tensor & bias, IntArrayRef stride, IntArrayRef padding) {
  return at::legacy::th::_thnn_conv2d_forward_out(output, finput, fgrad_input, self, weight, kernel_size, bias, stride, padding);
}

std::tuple<Tensor,Tensor,Tensor> thnn_conv2d_forward(const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, const Tensor & bias, IntArrayRef stride, IntArrayRef padding) {
  return at::legacy::th::_thnn_conv2d_forward(self, weight, kernel_size, bias, stride, padding);
}

std::tuple<Tensor &,Tensor &,Tensor &> thnn_conv2d_backward_out(Tensor & grad_input, Tensor & grad_weight, Tensor & grad_bias, const Tensor & grad_output, const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, const Tensor & finput, const Tensor & fgrad_input) {
  return at::legacy::th::_thnn_conv2d_backward_out(grad_input, grad_weight, grad_bias, grad_output, self, weight, kernel_size, stride, padding, finput, fgrad_input);
}

std::tuple<Tensor,Tensor,Tensor> thnn_conv2d_backward(const Tensor & grad_output, const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, const Tensor & finput, const Tensor & fgrad_input, std::array<bool,3> output_mask) {
  return at::legacy::th::_thnn_conv2d_backward(grad_output, self, weight, kernel_size, stride, padding, finput, fgrad_input, output_mask);
}

Tensor & thnn_conv3d_out(Tensor & output, const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, const Tensor & bias, IntArrayRef stride, IntArrayRef padding) {
  Tensor finput = at::empty({0}, self.options());
  Tensor fgrad_input = at::empty({0}, self.options());
  return std::get<0>(at::thnn_conv3d_forward_out(output, finput, fgrad_input, self, weight, kernel_size, bias, stride, padding));
}

Tensor & thnn_conv_depthwise2d_out(Tensor & output, const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, const Tensor & bias, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation) {
  return at::thnn_conv_depthwise2d_forward_out(output, self, weight, kernel_size, bias, stride, padding, dilation);
}

Tensor thnn_conv_depthwise2d(const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, const Tensor & bias, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation) {
  return at::thnn_conv_depthwise2d_forward(self, weight, kernel_size, bias, stride, padding, dilation);
}

Tensor & thnn_conv_depthwise2d_forward_out(Tensor & output, const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, const Tensor & bias, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation) {
  return at::legacy::th::_thnn_conv_depthwise2d_forward_out(output, self, weight, kernel_size, bias, stride, padding, dilation);
}

Tensor thnn_conv_depthwise2d_forward(const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, const Tensor & bias, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation) {
  return at::legacy::th::_thnn_conv_depthwise2d_forward(self, weight, kernel_size, bias, stride, padding, dilation);
}

std::tuple<Tensor &,Tensor &> thnn_conv_depthwise2d_backward_out(Tensor & grad_input, Tensor & grad_weight, const Tensor & grad_output, const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation) {
  return at::legacy::th::_thnn_conv_depthwise2d_backward_out(grad_input, grad_weight, grad_output, self, weight, kernel_size, stride, padding, dilation);
}

std::tuple<Tensor,Tensor> thnn_conv_depthwise2d_backward(const Tensor & grad_output, const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, std::array<bool,2> output_mask) {
  return at::legacy::th::_thnn_conv_depthwise2d_backward(grad_output, self, weight, kernel_size, stride, padding, dilation, output_mask);
}

Tensor thnn_conv3d(const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, const Tensor & bias, IntArrayRef stride, IntArrayRef padding) {
  return std::get<0>(at::thnn_conv3d_forward(self, weight, kernel_size, bias, stride, padding));
}

std::tuple<Tensor &,Tensor &,Tensor &> thnn_conv3d_forward_out(Tensor & output, Tensor & finput, Tensor & fgrad_input, const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, const Tensor & bias, IntArrayRef stride, IntArrayRef padding) {
  return at::legacy::th::_thnn_conv3d_forward_out(output, finput, fgrad_input, self, weight, kernel_size, bias, stride, padding);
}

std::tuple<Tensor,Tensor,Tensor> thnn_conv3d_forward(const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, const Tensor & bias, IntArrayRef stride, IntArrayRef padding) {
  return at::legacy::th::_thnn_conv3d_forward(self, weight, kernel_size, bias, stride, padding);
}

std::tuple<Tensor &,Tensor &,Tensor &> thnn_conv3d_backward_out(Tensor & grad_input, Tensor & grad_weight, Tensor & grad_bias, const Tensor & grad_output, const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, const Tensor & finput, const Tensor & fgrad_input) {
  return at::legacy::th::_thnn_conv3d_backward_out(grad_input, grad_weight, grad_bias, grad_output, self, weight, kernel_size, stride, padding, finput, fgrad_input);
}

std::tuple<Tensor,Tensor,Tensor> thnn_conv3d_backward(const Tensor & grad_output, const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, const Tensor & finput, const Tensor & fgrad_input, std::array<bool,3> output_mask) {
  return at::legacy::th::_thnn_conv3d_backward(grad_output, self, weight, kernel_size, stride, padding, finput, fgrad_input, output_mask);
}

Tensor & thnn_conv_dilated2d_out(Tensor & output, const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, const Tensor & bias, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation) {
  Tensor columns = at::empty({0}, self.options());
  Tensor ones = at::empty({0}, self.options());
  return std::get<0>(at::thnn_conv_dilated2d_forward_out(output, columns, ones, self, weight, kernel_size, bias, stride, padding, dilation));
}

Tensor thnn_conv_dilated2d(const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, const Tensor & bias, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation) {
  return std::get<0>(at::thnn_conv_dilated2d_forward(self, weight, kernel_size, bias, stride, padding, dilation));
}

std::tuple<Tensor &,Tensor &,Tensor &> thnn_conv_dilated2d_forward_out(Tensor & output, Tensor & columns, Tensor & ones, const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, const Tensor & bias, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation) {
  return at::legacy::th::_thnn_conv_dilated2d_forward_out(output, columns, ones, self, weight, kernel_size, bias, stride, padding, dilation);
}

std::tuple<Tensor,Tensor,Tensor> thnn_conv_dilated2d_forward(const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, const Tensor & bias, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation) {
  return at::legacy::th::_thnn_conv_dilated2d_forward(self, weight, kernel_size, bias, stride, padding, dilation);
}

std::tuple<Tensor &,Tensor &,Tensor &> thnn_conv_dilated2d_backward_out(Tensor & grad_input, Tensor & grad_weight, Tensor & grad_bias, const Tensor & grad_output, const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, const Tensor & columns, const Tensor & ones) {
  return at::legacy::th::_thnn_conv_dilated2d_backward_out(grad_input, grad_weight, grad_bias, grad_output, self, weight, kernel_size, stride, padding, dilation, columns, ones);
}

std::tuple<Tensor,Tensor,Tensor> thnn_conv_dilated2d_backward(const Tensor & grad_output, const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, const Tensor & columns, const Tensor & ones, std::array<bool,3> output_mask) {
  return at::legacy::th::_thnn_conv_dilated2d_backward(grad_output, self, weight, kernel_size, stride, padding, dilation, columns, ones, output_mask);
}

Tensor & thnn_conv_dilated3d_out(Tensor & output, const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, const Tensor & bias, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation) {
  Tensor columns = at::empty({0}, self.options());
  Tensor ones = at::empty({0}, self.options());
  return std::get<0>(at::thnn_conv_dilated3d_forward_out(output, columns, ones, self, weight, kernel_size, bias, stride, padding, dilation));
}

Tensor thnn_conv_dilated3d(const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, const Tensor & bias, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation) {
  return std::get<0>(at::thnn_conv_dilated3d_forward(self, weight, kernel_size, bias, stride, padding, dilation));
}

std::tuple<Tensor &,Tensor &,Tensor &> thnn_conv_dilated3d_forward_out(Tensor & output, Tensor & columns, Tensor & ones, const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, const Tensor & bias, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation) {
  return at::legacy::th::_thnn_conv_dilated3d_forward_out(output, columns, ones, self, weight, kernel_size, bias, stride, padding, dilation);
}

std::tuple<Tensor,Tensor,Tensor> thnn_conv_dilated3d_forward(const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, const Tensor & bias, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation) {
  return at::legacy::th::_thnn_conv_dilated3d_forward(self, weight, kernel_size, bias, stride, padding, dilation);
}

std::tuple<Tensor &,Tensor &,Tensor &> thnn_conv_dilated3d_backward_out(Tensor & grad_input, Tensor & grad_weight, Tensor & grad_bias, const Tensor & grad_output, const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, const Tensor & columns, const Tensor & ones) {
  return at::legacy::th::_thnn_conv_dilated3d_backward_out(grad_input, grad_weight, grad_bias, grad_output, self, weight, kernel_size, stride, padding, dilation, columns, ones);
}

std::tuple<Tensor,Tensor,Tensor> thnn_conv_dilated3d_backward(const Tensor & grad_output, const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, const Tensor & columns, const Tensor & ones, std::array<bool,3> output_mask) {
  return at::legacy::th::_thnn_conv_dilated3d_backward(grad_output, self, weight, kernel_size, stride, padding, dilation, columns, ones, output_mask);
}

Tensor thnn_col2im(const Tensor & self, IntArrayRef output_size, IntArrayRef kernel_size, IntArrayRef dilation, IntArrayRef padding, IntArrayRef stride) {
  return at::legacy::th::_thnn_col2im_forward(self, output_size, kernel_size, dilation, padding, stride);
}

Tensor & thnn_col2im_out(Tensor & output, const Tensor & self, IntArrayRef output_size, IntArrayRef kernel_size, IntArrayRef dilation, IntArrayRef padding, IntArrayRef stride) {
  return at::legacy::th::_thnn_col2im_forward_out(output, self, output_size, kernel_size, dilation, padding, stride);
}

Tensor thnn_col2im_backward(const Tensor & grad_output, IntArrayRef kernel_size, IntArrayRef dilation, IntArrayRef padding, IntArrayRef stride) {
  return at::legacy::th::_thnn_col2im_backward(grad_output, kernel_size, dilation, padding, stride);
}

Tensor & thnn_col2im_backward_out(Tensor & grad_input, const Tensor & grad_output, IntArrayRef kernel_size, IntArrayRef dilation, IntArrayRef padding, IntArrayRef stride) {
  return at::legacy::th::_thnn_col2im_backward_out(grad_input, grad_output, kernel_size, dilation, padding, stride);
}

Tensor thnn_im2col(const Tensor & self, IntArrayRef kernel_size, IntArrayRef dilation, IntArrayRef padding, IntArrayRef stride) {
  return at::legacy::th::_thnn_im2col_forward(self, kernel_size, dilation, padding, stride);
}

Tensor & thnn_im2col_out(Tensor & output, const Tensor & self, IntArrayRef kernel_size, IntArrayRef dilation, IntArrayRef padding, IntArrayRef stride) {
  return at::legacy::th::_thnn_im2col_forward_out(output, self, kernel_size, dilation, padding, stride);
}

Tensor thnn_im2col_backward(const Tensor & grad_output, IntArrayRef input_size, IntArrayRef kernel_size, IntArrayRef dilation, IntArrayRef padding, IntArrayRef stride) {
  return at::legacy::th::_thnn_im2col_backward(grad_output, input_size, kernel_size, dilation, padding, stride);
}

Tensor & thnn_im2col_backward_out(Tensor & grad_input, const Tensor & grad_output, IntArrayRef input_size, IntArrayRef kernel_size, IntArrayRef dilation, IntArrayRef padding, IntArrayRef stride) {
  return at::legacy::th::_thnn_im2col_backward_out(grad_input, grad_output, input_size, kernel_size, dilation, padding, stride);
}

}} // namespace at::native
