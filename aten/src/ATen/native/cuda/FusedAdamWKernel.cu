#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/DeviceGuard.h>
#include <ATen/TypeDefault.h>
#include <ATen/native/ForeachUtils.h>
#include <ATen/native/cuda/fused_adamw_amsgrad_impl.cuh>
#include <ATen/native/cuda/fused_adamw_impl.cuh>
#include <ATen/native/cuda/fused_adam_utils.cuh>
#include <c10/util/Exception.h>
#include <c10/util/Optional.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_foreach_add_native.h>
#include <ATen/ops/_foreach_sub_native.h>
#endif


namespace at { namespace native {

// note(crcrpar): To observe the CI rules, i.e. 20 minutes per file to compile, defensively split instantiations into _impl files.
// this is only for CUDA 11.3 for which it took about 20 minutes and 28 minutes in my workstation and CI, respectively.
// As a data point, it took about 20 seconds for CUDA 11.7 installed in my environment.
// See https://github.com/pytorch/pytorch/pull/81705 for details.
void _fused_adamw_kernel_cuda_(
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
  const auto nested_tensorlists = [&]() -> std::vector<TensorList> {
    if (amsgrad) {
      return {params, grads, exp_avgs, exp_avg_sqs, max_exp_avg_sqs, state_steps};
    } else {
      return {params, grads, exp_avgs, exp_avg_sqs, state_steps};
    }
  }();
  OptionalDeviceGuard guard;
  for (const auto & device_map : group_tensors_by_device_and_scalartype(nested_tensorlists, /* has_state_steps */ true)) {
    const auto & device = device_map.first;
    const auto cur_scale = get_device_tensor(device_grad_scale_map, grad_scale, device);
    const auto cur_found_inf = get_device_tensor(device_found_inf_map, found_inf, device);

    guard.reset_device(device);
    for (const auto & scalar_type_nested_tensors : device_map.second) {
      const auto& nested_tensors = scalar_type_nested_tensors.second;
      const TensorList cur_params{nested_tensors[0]}, cur_grads{nested_tensors[1]}, cur_exp_avgs{nested_tensors[2]}, cur_exp_avg_sqs{nested_tensors[3]}, cur_state_steps{nested_tensors[4 + static_cast<int>(amsgrad)]};
      at::native::foreach_tensor_add_scalar_kernel_cuda_(cur_state_steps, 1);
      if (amsgrad) {
        const TensorList cur_max_exp_avg_sqs{nested_tensors[4]};
        _fused_adamw_amsgrad_cuda_impl_(
            cur_params, cur_grads, cur_exp_avgs, cur_exp_avg_sqs, cur_max_exp_avg_sqs, cur_state_steps, lr, beta1, beta2, weight_decay, eps, maximize,
            cur_scale, cur_found_inf);
      } else {
        _fused_adamw_cuda_impl_(
            cur_params, cur_grads, cur_exp_avgs, cur_exp_avg_sqs, cur_state_steps, lr, beta1, beta2, weight_decay, eps, maximize,
            cur_scale, cur_found_inf);
          }
      if (cur_found_inf.has_value()) {
        at::native::foreach_tensor_sub_list_kernel_cuda_(cur_state_steps, std::vector<at::Tensor>(cur_state_steps.size(), cur_found_inf.value()), 1);
      }
    }
  }
}

}} // namespace at::native
