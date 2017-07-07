#include "batch_normalization.h"

#include "torch/csrc/autograd/variable.h"
#include "torch/csrc/autograd/functions/utils.h"
#include "torch/csrc/autograd/functions/basic_ops.h"
#include "torch/csrc/nn/THNN_generic.h"
#include "torch/csrc/utils/auto_gpu.h"

#ifdef WITH_CUDNN
#include "torch/csrc/cudnn/BatchNorm.h"
#include "torch/csrc/cudnn/Handles.h"
#include "torch/csrc/cudnn/Types.h"
extern THCState* state;
#endif

namespace torch { namespace autograd {

using thpp::Tensor;

#ifndef CUDNN_BN_MIN_EPSILON
#define CUDNN_BN_MIN_EPSILON 0
#endif

auto BatchNormForward::apply(const variable_list& inputs) -> variable_list {
  check_input_variables("BatchNorm", inputs, 3, 1);

  auto& input = inputs[0];
  auto& weight = inputs[1];
  auto& bias = inputs[2];
  AutoGPU guard(input->data->getDevice());

  bool use_cudnn = false;
#ifdef WITH_CUDNN
  use_cudnn = (input->data->isCuda()
               && input->data->type() != thpp::Type::HALF
               && weight && bias
               && cudnn_enabled && CUDNN_VERSION >= 5110L);
#endif

  auto output = input->data->newTensor();
  output->resizeAs(*input->data);

  std::unique_ptr<Tensor> save_mean(output->newTensor());
  save_mean->resizeAs(*running_mean);
  std::unique_ptr<Tensor> save_std(output->newTensor());
  save_std->resizeAs(*running_var);

  if (use_cudnn && eps >= CUDNN_BN_MIN_EPSILON) {
#ifdef WITH_CUDNN
    torch::cudnn::cudnn_batch_norm_forward(
        state,
        torch::cudnn::getCudnnHandle(),
        torch::cudnn::getCudnnDataType(*input->data),
        (THVoidTensor*)input->data->cdata(),
        (THVoidTensor*)output->cdata(),
        (THVoidTensor*)weight->data->cdata(),
        (THVoidTensor*)bias->data->cdata(),
        (THVoidTensor*)running_mean->cdata(),
        (THVoidTensor*)running_var->cdata(),
        (THVoidTensor*)save_mean->cdata(),
        (THVoidTensor*)save_std->cdata(),
        training,
        momentum,
        eps);
#endif
  } else {
    torch::nn::BatchNormalization_updateOutput(
        input->data.get(),
        output.get(),
        weight ? weight->data.get() : nullptr,
        bias ? bias->data.get() : nullptr,
        running_mean.get(),
        running_var.get(),
        save_mean.get(),
        save_std.get(),
        training,
        momentum,
        eps);
  }

  auto outputs = as_tensor_list(std::move(output));
  return wrap_outputs(inputs, std::move(outputs), [&](FunctionFlags f) {
    return std::make_shared<BatchNormBackward>(
        f, *this, std::move(save_mean), std::move(save_std),
        input->save(this),
        Variable::save_opt(weight.get(), this),
        Variable::save_opt(bias.get(), this));
  });
};

auto BatchNormBackward::apply(const variable_list& grad_outputs) -> variable_list {
  check_input_variables("BatchNormBackward", grad_outputs, 1);
  auto input = this->input.unpack_data();
  auto weight = this->weight.unpack_data();
  auto bias = this->bias.unpack_data();
  AutoGPU guard(input->getDevice());

  bool use_cudnn = false;
#ifdef WITH_CUDNN
  use_cudnn = (input->isCuda()
               && input->type() != thpp::Type::HALF
               && weight && bias && training
               && cudnn_enabled && CUDNN_VERSION >= 5110L);
#endif

  std::unique_ptr<Tensor> grad_input;
  if (should_compute_output(0) || use_cudnn) {
    grad_input = input->newTensor();
    grad_input->resizeAs(*input);
  }

  std::unique_ptr<Tensor> grad_weight;
  if (should_compute_output(1) || use_cudnn) {
    grad_weight = weight->newTensor();
    grad_weight->resizeAs(*weight);
    if (!use_cudnn) {
      grad_weight->zero();
    }
  }

  std::unique_ptr<Tensor> grad_bias;
  if (should_compute_output(2) || use_cudnn) {
    grad_bias = bias->newTensor();
    grad_bias->resizeAs(*bias);
    if (!use_cudnn) {
      grad_bias->zero();
    }
  }

  auto grad_output = grad_outputs[0]->data->contiguous();

  if (use_cudnn && eps >= CUDNN_BN_MIN_EPSILON) {
#ifdef WITH_CUDNN
    torch::cudnn::cudnn_batch_norm_backward(
        state,
        torch::cudnn::getCudnnHandle(),
        torch::cudnn::getCudnnDataType(*input),
        (THVoidTensor*)input->cdata(),
        (THVoidTensor*)grad_output->cdata(),
        (THVoidTensor*)grad_input->cdata(),
        (THVoidTensor*)grad_weight->cdata(),
        (THVoidTensor*)grad_bias->cdata(),
        (THVoidTensor*)weight->cdata(),
        (THVoidTensor*)running_mean->cdata(),
        (THVoidTensor*)running_var->cdata(),
        (THVoidTensor*)save_mean->cdata(),
        (THVoidTensor*)save_std->cdata(),
        training,
        eps);
#endif
  } else {
    torch::nn::BatchNormalization_backward(
        input.get(),
        grad_output.get(),
        grad_input.get(),
        grad_weight.get(),
        grad_bias.get(),
        weight.get(),
        running_mean.get(),
        running_var.get(),
        save_mean.get(),
        save_std.get(),
        training,
        1.0,
        eps);
  }

  auto outputs =  as_tensor_list(std::move(grad_input),
                                 std::move(grad_weight),
                                 std::move(grad_bias));
  return wrap_outputs(grad_outputs, std::move(outputs), [&](FunctionFlags f) {
    return std::make_shared<Error>("BatchNormBackward is not differentiable", std::move(f));
  });
};

auto BatchNormBackward::releaseVariables() -> void {
  input.data.reset();
  weight.data.reset();
  bias.data.reset();
}

}} // namespace torch::autograd
