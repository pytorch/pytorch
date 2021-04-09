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

Tensor & _th_masked_fill_(Tensor & self, const Tensor & mask, const Scalar& value);
Tensor & _th_masked_fill_bool_(Tensor & self, const Tensor & mask, const Scalar& value);
std::tuple<Tensor &,Tensor &> _th_sort_out(const Tensor & self, int64_t dim, bool descending, Tensor & values, Tensor & indices);
std::tuple<Tensor,Tensor> _th_sort(const Tensor & self, int64_t dim, bool descending);
std::tuple<Tensor &,Tensor &> _th_sort_out_stable(const Tensor & self, c10::optional<bool> stable, int64_t dim, bool descending, Tensor & values, Tensor & indices);
std::tuple<Tensor,Tensor> _th_sort_stable(const Tensor & self, c10::optional<bool> stable, int64_t dim, bool descending);
std::tuple<Tensor &,Tensor &> _th_topk_out(const Tensor & self, int64_t k, int64_t dim, bool largest, bool sorted, Tensor & values, Tensor & indices);
std::tuple<Tensor,Tensor> _th_topk(const Tensor & self, int64_t k, int64_t dim, bool largest, bool sorted);
Tensor & _th_renorm_out(const Tensor & self, const Scalar& p, int64_t dim, const Scalar& maxnorm, Tensor & result);
Tensor _th_renorm(const Tensor & self, const Scalar& p, int64_t dim, const Scalar& maxnorm);
Tensor & _th_renorm_(Tensor & self, const Scalar& p, int64_t dim, const Scalar& maxnorm);
Tensor & _th_cross_kernel_out(Tensor & result, const Tensor & self, const Tensor & other, int64_t dim);
Tensor _th_cross_kernel(const Tensor & self, const Tensor & other, int64_t dim);
std::tuple<Tensor &,Tensor &> _th_gels_out(const Tensor & self, const Tensor & A, Tensor & res1, Tensor & res2);
std::tuple<Tensor,Tensor> _th_gels(const Tensor & self, const Tensor & A);
Tensor & _th_potri_out(Tensor & output, const Tensor & self, bool upper);
Tensor _th_potri(const Tensor & self, bool upper);
std::tuple<Tensor &,Tensor &> _th_geqrf_out(const Tensor & self, Tensor & res1, Tensor & res2);
std::tuple<Tensor,Tensor> _th_geqrf(const Tensor & self);
Tensor & _th_copy_ignoring_overlaps_(Tensor & self, const Tensor & src);
Tensor & _thnn_multi_margin_loss_forward_out(const Tensor & self, const Tensor & target, const Scalar& p, const Scalar& margin, const c10::optional<Tensor>& weight_opt, int64_t reduction, Tensor & output);
Tensor _thnn_multi_margin_loss_forward(const Tensor & self, const Tensor & target, const Scalar& p, const Scalar& margin, const optional<Tensor> & weight, int64_t reduction);
Tensor & _thnn_multi_margin_loss_backward_out(const Tensor & grad_output, const Tensor & self, const Tensor & target, const Scalar& p, const Scalar& margin, const c10::optional<Tensor>& weight_opt, int64_t reduction, Tensor & grad_input);
Tensor _thnn_multi_margin_loss_backward(const Tensor & grad_output, const Tensor & self, const Tensor & target, const Scalar& p, const Scalar& margin, const optional<Tensor> & weight, int64_t reduction);
std::tuple<Tensor &,Tensor &> _thnn_multilabel_margin_loss_forward_out(const Tensor & self, const Tensor & target, int64_t reduction, Tensor & output, Tensor & is_target);
std::tuple<Tensor,Tensor> _thnn_multilabel_margin_loss_forward(const Tensor & self, const Tensor & target, int64_t reduction);
Tensor & _thnn_multilabel_margin_loss_backward_out(const Tensor & grad_output, const Tensor & self, const Tensor & target, int64_t reduction, const Tensor & is_target, Tensor & grad_input);
Tensor _thnn_multilabel_margin_loss_backward(const Tensor & grad_output, const Tensor & self, const Tensor & target, int64_t reduction, const Tensor & is_target);
std::tuple<Tensor &,Tensor &> _thnn_nll_loss_forward_out(const Tensor & self, const Tensor & target, const c10::optional<Tensor>& weight_opt, int64_t reduction, int64_t ignore_index, Tensor & output, Tensor & total_weight);
std::tuple<Tensor,Tensor> _thnn_nll_loss_forward(const Tensor & self, const Tensor & target, const optional<Tensor> & weight, int64_t reduction, int64_t ignore_index);
Tensor & _thnn_nll_loss_backward_out(const Tensor & grad_output, const Tensor & self, const Tensor & target, const c10::optional<Tensor>& weight_opt, int64_t reduction, int64_t ignore_index, const Tensor & total_weight, Tensor & grad_input);
Tensor _thnn_nll_loss_backward(const Tensor & grad_output, const Tensor & self, const Tensor & target, const optional<Tensor> & weight, int64_t reduction, int64_t ignore_index, const Tensor & total_weight);
std::tuple<Tensor &,Tensor &> _thnn_nll_loss2d_forward_out(const Tensor & self, const Tensor & target, const c10::optional<Tensor>& weight_opt, int64_t reduction, int64_t ignore_index, Tensor & output, Tensor & total_weight);
std::tuple<Tensor,Tensor> _thnn_nll_loss2d_forward(const Tensor & self, const Tensor & target, const optional<Tensor> & weight, int64_t reduction, int64_t ignore_index);
Tensor & _thnn_nll_loss2d_backward_out(const Tensor & grad_output, const Tensor & self, const Tensor & target, const c10::optional<Tensor>& weight_opt, int64_t reduction, int64_t ignore_index, const Tensor & total_weight, Tensor & grad_input);
Tensor _thnn_nll_loss2d_backward(const Tensor & grad_output, const Tensor & self, const Tensor & target, const optional<Tensor> & weight, int64_t reduction, int64_t ignore_index, const Tensor & total_weight);
Tensor & _thnn_glu_forward_out(const Tensor & self, int64_t dim, Tensor & output);
Tensor _thnn_glu_forward(const Tensor & self, int64_t dim);
Tensor & _thnn_glu_backward_out(const Tensor & grad_output, const Tensor & self, int64_t dim, Tensor & grad_input);
Tensor _thnn_glu_backward(const Tensor & grad_output, const Tensor & self, int64_t dim);
std::tuple<Tensor &,Tensor &> _thnn_log_sigmoid_forward_out(const Tensor & self, Tensor & output, Tensor & buffer);
std::tuple<Tensor,Tensor> _thnn_log_sigmoid_forward(const Tensor & self);
Tensor & _thnn_log_sigmoid_backward_out(const Tensor & grad_output, const Tensor & self, const Tensor & buffer, Tensor & grad_input);
Tensor _thnn_log_sigmoid_backward(const Tensor & grad_output, const Tensor & self, const Tensor & buffer);
Tensor & _thnn_rrelu_with_noise_forward_out(const Tensor & self, const Tensor & noise, const Scalar& lower, const Scalar& upper, bool training, c10::optional<at::Generator> generator, Tensor & output);
Tensor _thnn_rrelu_with_noise_forward(const Tensor & self, const Tensor & noise, const Scalar& lower, const Scalar& upper, bool training, c10::optional<at::Generator> generator);
Tensor _thnn_rrelu_with_noise_backward(const Tensor & grad_output, const Tensor & self, const Tensor & noise, const Scalar& lower, const Scalar& upper, bool training);
Tensor & _thnn_rrelu_with_noise_forward_(Tensor & self, const Tensor & noise, const Scalar& lower, const Scalar& upper, bool training, c10::optional<at::Generator> generator);
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
