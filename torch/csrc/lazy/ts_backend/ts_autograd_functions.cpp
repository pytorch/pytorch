#include <ATen/Operators.h>
#include <ATen/native/CPUFallback.h>
#include <torch/csrc/lazy/ts_backend/ts_autograd_functions.h>
#include <torch/csrc/lazy/ts_backend/ts_eager_fallback.h>

namespace torch::lazy {

at::Tensor MaxPool3dAutogradFunctionTS::forward(
    torch::autograd::AutogradContext* ctx,
    const at::Tensor& self,
    at::IntArrayRef kernel_size,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef dilation,
    bool ceil_mode) {
  ctx->saved_data["kernel_size"] = kernel_size;
  ctx->saved_data["stride"] = stride;
  ctx->saved_data["padding"] = padding;
  ctx->saved_data["dilation"] = dilation;
  ctx->saved_data["ceil_mode"] = ceil_mode;
  auto results = at::native::
      call_fallback_fn<&ltc_eager_fallback, ATEN_OP(max_pool3d_with_indices)>::
          call(self, kernel_size, stride, padding, dilation, ceil_mode);
  ctx->save_for_backward({self, std::get<1>(results)});
  return std::get<0>(results);
}

torch::autograd::variable_list MaxPool3dAutogradFunctionTS::backward(
    torch::autograd::AutogradContext* ctx,
    torch::autograd::variable_list grad_output) {
  auto kernel_size = ctx->saved_data["kernel_size"].toIntList().vec();
  auto stride = ctx->saved_data["stride"].toIntList().vec();
  auto padding = ctx->saved_data["padding"].toIntList().vec();
  auto dilation = ctx->saved_data["dilation"].toIntList().vec();
  auto ceil_mode = ctx->saved_data["ceil_mode"].toBool();
  auto saved = ctx->get_saved_variables();
  const auto& self = saved[0];
  at::Tensor grad;
  const auto& indices = saved[1];
  grad = at::native::call_fallback_fn<
      &ltc_eager_fallback,
      ATEN_OP(max_pool3d_with_indices_backward)>::
      call(
          grad_output[0],
          self,
          kernel_size,
          stride,
          padding,
          dilation,
          ceil_mode,
          indices);

  at::Tensor undef;
  torch::autograd::variable_list grad_inputs = {
      grad, undef, undef, undef, undef, undef};
  return grad_inputs;
}

} // namespace torch::lazy
