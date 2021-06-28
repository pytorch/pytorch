#include "lazy_tensor_core/csrc/ts_backend/aten_autograd_ops_ts.h"

#include <ATen/Operators.h>
#include <ATen/native/CPUFallback.h>

#include "lazy_tensor_core/csrc/aten_ltc_bridge.h"
#include "lazy_tensor_core/csrc/ts_backend/aten_eager_fallback.h"

namespace torch_lazy_tensors {
namespace aten_autograd_ops_ts {

torch::Tensor MaxPool2dAutogradFunctionTS::forward(
    torch::autograd::AutogradContext* ctx, torch::Tensor self,
    torch::IntArrayRef kernel_size, torch::IntArrayRef stride,
    torch::IntArrayRef padding, torch::IntArrayRef dilation, bool ceil_mode) {
  ctx->saved_data["kernel_size"] = kernel_size;
  ctx->saved_data["stride"] = stride;
  ctx->saved_data["padding"] = padding;
  ctx->saved_data["dilation"] = dilation;
  ctx->saved_data["ceil_mode"] = ceil_mode;
  auto results = at::native::call_fallback_fn<
      &ltc_eager_fallback, ATEN_OP(max_pool2d_with_indices)>::call(self,
                                                                   kernel_size,
                                                                   stride,
                                                                   padding,
                                                                   dilation,
                                                                   ceil_mode);
  ctx->save_for_backward({self, std::get<1>(results)});
  return std::get<0>(results);
}

torch::autograd::variable_list MaxPool2dAutogradFunctionTS::backward(
    torch::autograd::AutogradContext* ctx,
    torch::autograd::variable_list grad_output) {
  auto kernel_size = ctx->saved_data["kernel_size"].toIntList().vec();
  auto stride = ctx->saved_data["stride"].toIntList().vec();
  auto padding = ctx->saved_data["padding"].toIntList().vec();
  auto dilation = ctx->saved_data["dilation"].toIntList().vec();
  auto ceil_mode = ctx->saved_data["ceil_mode"].toBool();
  auto saved = ctx->get_saved_variables();
  auto self = saved[0];
  auto indices = saved[1];
  torch::Tensor grad = at::native::call_fallback_fn<
      &ltc_eager_fallback,
      ATEN_OP(max_pool2d_with_indices_backward)>::call(grad_output[0], self,
                                                       kernel_size, stride,
                                                       padding, dilation,
                                                       ceil_mode, indices);

  torch::Tensor undef;
  torch::autograd::variable_list grad_inputs = {grad,  undef, undef,
                                                undef, undef, undef};
  return grad_inputs;
}

torch::Tensor MaxPool3dAutogradFunctionTS::forward(
    torch::autograd::AutogradContext* ctx, torch::Tensor self,
    torch::IntArrayRef kernel_size, torch::IntArrayRef stride,
    torch::IntArrayRef padding, torch::IntArrayRef dilation, bool ceil_mode) {
  ctx->saved_data["kernel_size"] = kernel_size;
  ctx->saved_data["stride"] = stride;
  ctx->saved_data["padding"] = padding;
  ctx->saved_data["dilation"] = dilation;
  ctx->saved_data["ceil_mode"] = ceil_mode;
  auto results = at::native::call_fallback_fn<
      &ltc_eager_fallback, ATEN_OP(max_pool3d_with_indices)>::call(self,
                                                                   kernel_size,
                                                                   stride,
                                                                   padding,
                                                                   dilation,
                                                                   ceil_mode);
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
  auto self = saved[0];
  torch::Tensor grad;
  auto indices = saved[1];
  grad = at::native::call_fallback_fn<
      &ltc_eager_fallback,
      ATEN_OP(max_pool3d_with_indices_backward)>::call(grad_output[0], self,
                                                       kernel_size, stride,
                                                       padding, dilation,
                                                       ceil_mode, indices);

  torch::Tensor undef;
  torch::autograd::variable_list grad_inputs = {grad,  undef, undef,
                                                undef, undef, undef};
  return grad_inputs;
}

}  // namespace aten_autograd_ops_ts
}  // namespace torch_lazy_tensors
