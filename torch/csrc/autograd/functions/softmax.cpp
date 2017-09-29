
#include "softmax.h"

#include "torch/csrc/autograd/variable.h"
#include "torch/csrc/autograd/functions/utils.h"
#include "torch/csrc/utils/auto_gpu.h"
#include "torch/csrc/DynamicTypes.h"
#include "torch/csrc/Exceptions.h"

namespace torch { namespace autograd {

template<bool is_log>
variable_list SoftmaxBase<is_log>::apply(const variable_list& inputs) {
  using BackwardBase = SoftmaxBackwardBase<is_log>;
  check_input_variables("SoftmaxBase", inputs, 1);
  AutoGPU gpu_guard(inputs[0].data());

  auto input = inputs[0].data().contiguous();
  auto output = input.type().tensor(input.sizes());

  if (is_log) {
    at::log_softmax_out(output, input, dim);
  } else {
    at::softmax_out(output, input, dim);
  }

  // This gets a bit weird because we need to save the output...
  std::shared_ptr<BackwardBase> backward;
  auto outputs = wrap_outputs(inputs, as_tensor_list(output), [this, &backward](FunctionFlags f) {
    backward = std::make_shared<BackwardBase>(std::move(f), this->dim);
    return backward;
  });
  if (backward && backward->is_executable) {
    backward->saved_output = SavedVariable(outputs[0], backward.get());
  }
  return outputs;
};

template<bool is_log>
variable_list SoftmaxBackwardBase<is_log>::apply(const variable_list& grad_outputs) {
  using BackwardBase = typename std::conditional<is_log, LogSoftmaxBackwardBackward, SoftmaxBackwardBackward>::type;
  check_input_variables("SoftmaxBackwardBase", grad_outputs, 1);
  AutoGPU gpu_guard(grad_outputs[0]);

  auto output_var = saved_output.unpack(shared_from_this());
  auto& grad_output_var = grad_outputs[0];
  auto& output = output_var.data();
  auto& grad_output = grad_output_var.data();
  auto grad_input = output.type().tensor(output.sizes());

  auto input = output.type().tensor(); // We don't save the input, because THNN doesn't use it anyway...
  if (is_log) {
    at::log_softmax_backward_out(grad_input, grad_output, input, dim, output);
  } else {
    at::softmax_backward_out(grad_input, grad_output, input, dim, output);
  }

  variable_list all_inputs {output_var, grad_output_var};
  return wrap_outputs(all_inputs, as_tensor_list(grad_input), [this, &output_var, &grad_output_var](FunctionFlags f) {
    auto fn = std::make_shared<BackwardBase>(std::move(f));
    if (fn->is_executable) {
      fn->saved_output = SavedVariable(output_var, fn.get());
      fn->saved_grad_output = SavedVariable(grad_output_var, fn.get());
      fn->dim = this->dim;
    }
    return fn;
  });
}

// These need to be explicitly instantiated, because they're not in the header.
template struct SoftmaxBase<true>;
template struct SoftmaxBase<false>;
template struct SoftmaxBackwardBase<true>;
template struct SoftmaxBackwardBase<false>;

variable_list SoftmaxBackwardBackward::apply(const variable_list& grad_grad_inputs) {
  check_input_variables("SoftmaxBackwardBackward", grad_grad_inputs, 1);
  auto output = saved_output.unpack(shared_from_this());
  auto gO = saved_grad_output.unpack();
  auto& ggI = grad_grad_inputs[0];

  // Terms for reuse
  auto ggI_out_sum = (ggI * output).sum(dim, true);
  auto gO_out_sum = (gO * output).sum(dim, true);

  // NOTE: this is 2nd order grad output
  auto gO2 = (gO - gO_out_sum) * ggI - gO * ggI_out_sum;
  auto ggO = output * (ggI - ggI_out_sum);

  return {Variable(std::move(gO2)), Variable(std::move(ggO))};
}

variable_list LogSoftmaxBackwardBackward::apply(const variable_list& grad_grad_inputs) {
  check_input_variables("LogSoftmaxBackwardBackward", grad_grad_inputs, 1);
  auto output = saved_output.unpack(shared_from_this());
  auto gO = saved_grad_output.unpack();
  auto& ggI = grad_grad_inputs[0];

  auto output_exp = output.exp();
  // NOTE: this is 2nd order grad output
  auto gO2 = (-output_exp) * ggI * gO.sum(dim, true);
  auto ggO = ggI - (ggI * output_exp).sum(dim, true);

  return {Variable(std::move(gO2)), Variable(std::move(ggO))};
}

}}
