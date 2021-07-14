#pragma once

#include <ATen/Context.h>
#include <c10/core/ScalarType.h>
#include <c10/core/TensorOptions.h>

namespace c10 {
class Scalar;
}
namespace at {
struct Generator;
class Tensor;
struct Type;
} // namespace at

namespace at {
namespace native {
namespace legacy {
namespace cuda {

std::tuple<Tensor &,Tensor &> _th_gels_out(const Tensor & self, const Tensor & A, Tensor & res1, Tensor & res2);
std::tuple<Tensor,Tensor> _th_gels(const Tensor & self, const Tensor & A);
Tensor & _th_potri_out(Tensor & output, const Tensor & self, bool upper);
Tensor _th_potri(const Tensor & self, bool upper);
Tensor & _th_copy_ignoring_overlaps_(Tensor & self, const Tensor & src);
std::tuple<Tensor &,Tensor &> _thnn_multilabel_margin_loss_forward_out(const Tensor & self, const Tensor & target, int64_t reduction, Tensor & output, Tensor & is_target);
std::tuple<Tensor,Tensor> _thnn_multilabel_margin_loss_forward(const Tensor & self, const Tensor & target, int64_t reduction);
Tensor & _thnn_multilabel_margin_loss_backward_out(const Tensor & grad_output, const Tensor & self, const Tensor & target, int64_t reduction, const Tensor & is_target, Tensor & grad_input);
Tensor _thnn_multilabel_margin_loss_backward(const Tensor & grad_output, const Tensor & self, const Tensor & target, int64_t reduction, const Tensor & is_target);
std::tuple<Tensor &,Tensor &> _thnn_nll_loss2d_forward_out(const Tensor & self, const Tensor & target, const c10::optional<Tensor>& weight_opt, int64_t reduction, int64_t ignore_index, Tensor & output, Tensor & total_weight);
std::tuple<Tensor,Tensor> _thnn_nll_loss2d_forward(const Tensor & self, const Tensor & target, const optional<Tensor> & weight, int64_t reduction, int64_t ignore_index);
Tensor & _thnn_nll_loss2d_backward_out(const Tensor & grad_output, const Tensor & self, const Tensor & target, const c10::optional<Tensor>& weight_opt, int64_t reduction, int64_t ignore_index, const Tensor & total_weight, Tensor & grad_input);
Tensor _thnn_nll_loss2d_backward(const Tensor & grad_output, const Tensor & self, const Tensor & target, const optional<Tensor> & weight, int64_t reduction, int64_t ignore_index, const Tensor & total_weight);
Tensor _thnn_rrelu_with_noise_backward(const Tensor & grad_output, const Tensor & self, const Tensor & noise, const Scalar& lower, const Scalar& upper, bool training);
std::tuple<Tensor &,Tensor &,Tensor &> _thnn_conv2d_forward_out(const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, const c10::optional<Tensor>& bias_opt, IntArrayRef stride, IntArrayRef padding, Tensor & output, Tensor & columns, Tensor & ones);
std::tuple<Tensor,Tensor,Tensor> _thnn_conv2d_forward(const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, const optional<Tensor> & bias, IntArrayRef stride, IntArrayRef padding);
std::tuple<Tensor &,Tensor &,Tensor &> _thnn_conv2d_backward_out(Tensor & grad_input, Tensor & grad_weight, Tensor & grad_bias, const Tensor & grad_output, const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, const Tensor & columns, const Tensor & ones);
std::tuple<Tensor,Tensor,Tensor> _thnn_conv2d_backward(const Tensor & grad_output, const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, const Tensor & columns, const Tensor & ones, std::array<bool,3> output_mask);
Tensor & _thnn_conv_depthwise2d_forward_out(const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, const c10::optional<Tensor>& bias_opt, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, Tensor & output);
Tensor _thnn_conv_depthwise2d_forward(const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, const optional<Tensor> & bias, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation);
std::tuple<Tensor &,Tensor &> _thnn_conv_depthwise2d_backward_out(Tensor & grad_input, Tensor & grad_weight, const Tensor & grad_output, const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation);
std::tuple<Tensor,Tensor> _thnn_conv_depthwise2d_backward(const Tensor & grad_output, const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, std::array<bool,2> output_mask);

} // namespace th
} // namespace legacy
} // namespace native
} // namespace at
