#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/DeviceGuard.h>
#include <ATen/TypeDefault.h>
#include <ATen/native/ForeachUtils.h>
#include <ATen/native/cuda/fused_adam_amsgrad_impl.cuh>
#include <ATen/native/cuda/fused_adam_impl.cuh>
#include <ATen/native/cuda/fused_adam_utils.cuh>
#include <c10/util/Exception.h>
#include <c10/util/Optional.h>


namespace at::native {

// note(crcrpar): To observe the CI rules, i.e. 20 minutes per file to compile, defensively split instantiations into _impl files.
// this is only for CUDA 11.3 for which it took about 20 minutes and 28 minutes in my workstation and CI, respectively.
// As a data point, it took about 20 seconds for CUDA 11.7 installed in my environment.
// See https://github.com/pytorch/pytorch/pull/81705 for details.
void _fused_adam_kernel_cuda_(
    at::TensorList params,
    at::TensorList grads,
    at::TensorList exp_avgs,
    at::TensorList exp_avg_sqs,
    at::TensorList max_exp_avg_sqs,
    at::TensorList state_steps,
    const double lr,
    const double beta1,
    const double beta2,
    const double weight_decay,
    const double eps,
    const bool amsgrad,
    const bool maximize,
    const c10::optional<at::Tensor>& grad_scale,
    const c10::optional<at::Tensor>& found_inf
) {
  auto device_grad_scale_map = init_map(grad_scale);
  auto device_found_inf_map = init_map(found_inf);
  if (amsgrad) {
    TORCH_CHECK(
        at::native::check_fast_path_restrictions({params, grads, exp_avgs, exp_avg_sqs, max_exp_avg_sqs}),
        "params, grads, exp_avgs, exp_avg_sqs, and max_exp_avg_sqs must have same dtype, device, and layout");
    const auto device = device_of(params);
    TORCH_CHECK(device.has_value());
    OptionalDeviceGuard device_guard(device);
    _fused_adam_amsgrad_cuda_impl_(
        params, grads, exp_avgs, exp_avg_sqs, max_exp_avg_sqs, state_steps, lr, beta1, beta2, weight_decay, eps, maximize,
        get_device_tensor(device_grad_scale_map, grad_scale, device.value()), get_device_tensor(device_found_inf_map, found_inf, device.value()));
  } else {
    TORCH_CHECK(
        at::native::check_fast_path_restrictions({params, grads, exp_avgs, exp_avg_sqs}),
        "params, grads, exp_avgs, and exp_avg_sqs must have same dtype, device, and layout");
    const auto device = device_of(params);
    TORCH_CHECK(device.has_value());
    OptionalDeviceGuard device_guard(device);
    _fused_adam_cuda_impl_(
        params, grads, exp_avgs, exp_avg_sqs, state_steps, lr, beta1, beta2, weight_decay, eps, maximize,
        get_device_tensor(device_grad_scale_map, grad_scale, device.value()), get_device_tensor(device_found_inf_map, found_inf, device.value()));
  }
}

} // namespace at::native
