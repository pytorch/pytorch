#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/native/ForeachUtils.h>
#include <ATen/native/mps/operations/MultiTensorApply.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_fused_sgd.h>
#include <ATen/ops/_fused_sgd_native.h>
#endif

namespace at::native {

namespace mps {

static void _fused_sgd_with_momentum_kernel_mps_(at::TensorList params,
                                                 at::TensorList grads,
                                                 at::TensorList momentum_buffer_list,
                                                 const double weight_decay,
                                                 const double momentum,
                                                 const double lr,
                                                 const double dampening,
                                                 const bool nesterov,
                                                 const bool maximize,
                                                 const bool is_first_step,
                                                 const std::optional<at::Tensor>& grad_scale,
                                                 const std::optional<at::Tensor>& found_inf) {
  TORCH_CHECK_GT(momentum, 0);
  TORCH_CHECK(at::native::check_fast_path_restrictions({params, grads, momentum_buffer_list}));

  std::vector<std::vector<at::Tensor>> tensor_lists{params.vec(), grads.vec(), momentum_buffer_list.vec()};

  const std::string kernel_name = "fused_sgd_momentum_" + scalarToMetalTypeString(params[0].scalar_type());

  at::TensorList state_steps;

  multi_tensor_apply_for_fused_optimizer<3, 512>(kernel_name,
                                                 tensor_lists,
                                                 state_steps,
                                                 mps::FusedSgdEncodingFunctor(),
                                                 weight_decay,
                                                 momentum,
                                                 lr,
                                                 dampening,
                                                 nesterov,
                                                 maximize,
                                                 is_first_step);
}

static void _fused_sgd_with_momentum_kernel_mps_(at::TensorList params,
                                                 at::TensorList grads,
                                                 at::TensorList momentum_buffer_list,
                                                 const double weight_decay,
                                                 const double momentum,
                                                 const at::Tensor& lr_tensor,
                                                 const double dampening,
                                                 const bool nesterov,
                                                 const bool maximize,
                                                 const bool is_first_step,
                                                 const std::optional<at::Tensor>& grad_scale,
                                                 const std::optional<at::Tensor>& found_inf) {
  TORCH_CHECK_GT(momentum, 0);
  TORCH_CHECK(at::native::check_fast_path_restrictions({params, grads, momentum_buffer_list}));

  std::vector<std::vector<at::Tensor>> tensor_lists{params.vec(), grads.vec(), momentum_buffer_list.vec()};

  const std::string kernel_name = "fused_sgd_momentum_" + scalarToMetalTypeString(params[0].scalar_type());

  at::TensorList state_steps;

  multi_tensor_apply_for_fused_optimizer<3, 512>(kernel_name,
                                                 tensor_lists,
                                                 state_steps,
                                                 mps::FusedSgdEncodingFunctor(),
                                                 weight_decay,
                                                 momentum,
                                                 lr_tensor,
                                                 dampening,
                                                 nesterov,
                                                 maximize,
                                                 is_first_step);
}

} // namespace mps

void _fused_sgd_kernel_mps_(at::TensorList params,
                            at::TensorList grads,
                            at::TensorList momentum_buffer_list,
                            const double weight_decay,
                            const double momentum,
                            const double lr,
                            const double dampening,
                            const bool nesterov,
                            const bool maximize,
                            const bool is_first_step,
                            const std::optional<at::Tensor>& grad_scale,
                            const std::optional<at::Tensor>& found_inf) {
  TORCH_CHECK(!grad_scale.has_value() && !found_inf.has_value(), "grad_scale and found_inf are not supported on MPS");

  if (!momentum_buffer_list.empty()) {
    mps::_fused_sgd_with_momentum_kernel_mps_(params,
                                              grads,
                                              momentum_buffer_list,
                                              weight_decay,
                                              momentum,
                                              lr,
                                              dampening,
                                              nesterov,
                                              maximize,
                                              is_first_step,
                                              grad_scale,
                                              found_inf);
    return;
  }
  TORCH_CHECK_EQ(momentum, 0);
  TORCH_CHECK(at::native::check_fast_path_restrictions({params, grads}));
  if (is_first_step) {
    TORCH_WARN_ONCE("`is_first_step` argument has no effect when `momentum_buffer_list` is empty");
  }

  std::vector<std::vector<at::Tensor>> tensor_lists{params.vec(), grads.vec()};

  const std::string kernel_name = "fused_sgd_" + mps::scalarToMetalTypeString(params[0].scalar_type());

  at::TensorList state_steps;

  multi_tensor_apply_for_fused_optimizer<2, 512>(kernel_name,
                                                 tensor_lists,
                                                 state_steps,
                                                 mps::FusedSgdEncodingFunctor(),
                                                 weight_decay,
                                                 momentum,
                                                 lr,
                                                 dampening,
                                                 nesterov,
                                                 maximize,
                                                 is_first_step);
}

void _fused_sgd_kernel_mps_(at::TensorList params,
                            at::TensorList grads,
                            at::TensorList momentum_buffer_list,
                            const double weight_decay,
                            const double momentum,
                            const at::Tensor& lr_tensor,
                            const double dampening,
                            const bool nesterov,
                            const bool maximize,
                            const bool is_first_step,
                            const std::optional<at::Tensor>& grad_scale,
                            const std::optional<at::Tensor>& found_inf) {
  TORCH_CHECK(!grad_scale.has_value() && !found_inf.has_value(), "grad_scale and found_inf are not supported on MPS");

  if (!momentum_buffer_list.empty()) {
    mps::_fused_sgd_with_momentum_kernel_mps_(params,
                                              grads,
                                              momentum_buffer_list,
                                              weight_decay,
                                              momentum,
                                              lr_tensor,
                                              dampening,
                                              nesterov,
                                              maximize,
                                              is_first_step,
                                              grad_scale,
                                              found_inf);
    return;
  }
  if (lr_tensor.is_cpu()) {
    _fused_sgd_kernel_mps_(params,
                           grads,
                           momentum_buffer_list,
                           weight_decay,
                           momentum,
                           lr_tensor.item<double>(),
                           dampening,
                           nesterov,
                           maximize,
                           is_first_step,
                           grad_scale,
                           found_inf);
    return;
  }
  TORCH_CHECK_EQ(momentum, 0);
  TORCH_CHECK(at::native::check_fast_path_restrictions({params, grads}));
  if (is_first_step) {
    TORCH_WARN_ONCE("`is_first_step` argument has no effect when `momentum_buffer_list` is empty");
  }

  TORCH_CHECK(lr_tensor.device() == params[0].device(), "lr must be on the same GPU device as the params");

  std::vector<std::vector<at::Tensor>> tensor_lists{params.vec(), grads.vec()};

  const std::string kernel_name = "fused_sgd_" + mps::scalarToMetalTypeString(params[0].scalar_type());

  at::TensorList state_steps;

  multi_tensor_apply_for_fused_optimizer<2, 512>(kernel_name,
                                                 tensor_lists,
                                                 state_steps,
                                                 mps::FusedSgdEncodingFunctor(),
                                                 weight_decay,
                                                 momentum,
                                                 lr_tensor,
                                                 dampening,
                                                 nesterov,
                                                 maximize,
                                                 is_first_step);
}

} // namespace at::native
