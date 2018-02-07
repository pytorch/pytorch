#include "tensor.h"

#include "torch/csrc/autograd/variable.h"
#include "torch/csrc/autograd/functions/basic_ops.h"
#include "torch/csrc/autograd/functions/utils.h"
#include "torch/csrc/autograd/generated/Functions.h"
#include "torch/csrc/utils/auto_gpu.h"

namespace torch { namespace autograd {

auto CopyBackwards::apply(const variable_list& grads) -> variable_list {
  check_input_variables("CopyBackwards", grads, 1);
  auto& grad = grads[0];
  variable_list grad_inputs(2);
  if (should_compute_output(0)) {
    grad_inputs[0] = at::zeros_like(grad);
  }
  if (should_compute_output(1)) {
    AutoGPU autoGPU(src_device);
    if (grad.is_cuda() && grad.get_device() != src_device) {
      grad_inputs[1] = src_type->copy(grad);
    } else {
      grad_inputs[1] = grad.toType(*src_type);
    }
  }
  return grad_inputs;
};

CopySlices::CopySlices(const Variable& base_var, at::TensorGeometry view_, std::shared_ptr<Function> fn_)
  : base(base_var)
  , view(std::move(view_))
  , fn(std::move(fn_))
{
  num_inputs = 1;

  // Take the next_functions of fn as our own, except for index 0 which goes
  // to base instead of the view.
  const auto num_connections = fn->next_functions.size();
  next_functions.reserve(num_connections);
  next_functions.push_back(base_var.gradient_edge());
  for (size_t i = 1; i < num_connections; i++) {
    next_functions.push_back(fn->next_functions[i]);
  }
}

auto CopySlices::apply(const variable_list& inputs) -> variable_list {
  check_input_variables("CopySlices", inputs, 1);
  auto& grad = inputs[0];

  if (!fn) {
    throw std::runtime_error(ERR_BACKWARD_TWICE);
  }

  auto result = grad.type().tensor(base.sizes(), base.strides());
  result.copy_(grad);

  auto offset = view.storage_offset() - base.storage_offset();
  auto grad_slice = result.as_strided(view.sizes(), view.strides(), offset);

  // TODO: We clone grad_slice because we modify it below and "fn" might save
  // it for the backward of res. We might be able to avoid the clone() if
  // double-backprop is disabled.
  auto res = (*fn)({ grad_slice.clone() });

  variable_list grad_inputs(next_functions.size());
  for (size_t i = 0; i < res.size(); i++) {
    if (should_compute_output(i)) {
      TORCH_ASSERT(res[i].defined());
      if (i == 0) {
        grad_slice.copy_(res[i]);
        grad_inputs[i] = std::move(result);
      } else {
        grad_inputs[i] = std::move(res[i]);
      }
    }
  }

  return grad_inputs;
}

void CopySlices::releaseVariables() {
  fn = nullptr;
}

}} // namespace torch::autograd
