#include "batch_normalization.h"

#include "torch/csrc/autograd/python_function.h"
#include "torch/csrc/autograd/python_variable.h"
#include "torch/csrc/autograd/variable.h"
#include "torch/csrc/autograd/functions/utils.h"
#include "torch/csrc/autograd/functions/basic_ops.h"
#include "torch/csrc/nn/THNN_generic.h"
#include "torch/csrc/utils/auto_gil.h"
#include "torch/csrc/utils/auto_gpu.h"
#include "torch/csrc/Exceptions.h"
#include <sstream>

#ifdef WITH_CUDNN
#include "torch/csrc/cudnn/BatchNorm.h"
#include "torch/csrc/cudnn/Handles.h"
#include "torch/csrc/cudnn/Types.h"
extern THCState* state;
#endif

namespace {
    void check_dims_match_num_input_features(const std::string& arg_name, long expected, long actual){
      if (actual != expected){
        std::stringstream ss;
        ss << arg_name << " should contain " << expected << " elements not " << actual ;
        throw std::runtime_error(ss.str());
      }
    }
}

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
   
  auto num_features = input->data->rawSizes()[1];
  check_dims_match_num_input_features("running_mean", num_features, running_mean->numel());
  check_dims_match_num_input_features("running_var", num_features, running_var->numel());
  if (weight){
    check_dims_match_num_input_features("weight", num_features, weight->data->numel());
  }
  if (bias){
    check_dims_match_num_input_features("bias", num_features, bias->data->numel());
  }

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
  auto input_var = this->input.unpack();
  auto weight_var = this->weight.unpack();
  auto bias_var = this->bias.unpack();

  std::unique_ptr<thpp::Tensor> input {input_var->data->clone_shallow()};
  std::unique_ptr<thpp::Tensor> weight {weight_var ? weight_var->data->clone_shallow() : nullptr};
  std::unique_ptr<thpp::Tensor> bias {bias_var ? bias_var->data->clone_shallow() : nullptr};

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

  // Add saved variables used out of the pure autograd to inputs
  variable_list all_inputs(grad_outputs);
  all_inputs.push_back(input_var);
  if (weight.get()) {
    all_inputs.push_back(weight_var);
  }
  auto outputs =  as_tensor_list(std::move(grad_input),
                                 std::move(grad_weight),
                                 std::move(grad_bias));
  return wrap_outputs(all_inputs, std::move(outputs), [&](FunctionFlags f) {
    return std::make_shared<BatchNormBackwardBackward>(
      f, *this, std::move(save_mean), std::move(save_std),
      input_var->save(this), Variable::save_opt(weight_var.get(), this),
      grad_outputs[0]->save(this));
    });
};

auto BatchNormBackward::releaseVariables() -> void {
  input.data.reset();
  weight.data.reset();
  bias.data.reset();
}

std::shared_ptr<torch::autograd::Variable> getReturnTupleVar(PyObject *p, Py_ssize_t pos) {
  PyObject *item = PyTuple_GET_ITEM(p, pos);
  return item == Py_None ? nullptr : ((THPVariable*)item)->cdata;
}

auto BatchNormBackwardBackward::apply(const variable_list& grad_grad_inputs) -> variable_list {
  check_input_variables("BatchNormBackwardBackward", grad_grad_inputs, 3, 0);
  auto ggI = grad_grad_inputs[0];
  auto ggW = grad_grad_inputs[1];
  auto ggb = grad_grad_inputs[2];

  auto gO = grad_output.unpack();
  auto input_var = input.unpack();
  auto weight_var = weight.unpack();

  AutoGIL gil;
  THPObjectPtr input_pvar(THPVariable_Wrap(input_var));
  THPObjectPtr weight_pvar(weight_var ? THPVariable_Wrap(weight_var) : Py_None);
  THPObjectPtr ggi_pvar(ggI ? THPVariable_Wrap(ggI) : Py_None);
  THPObjectPtr ggW_pvar(ggW ? THPVariable_Wrap(ggW) : Py_None);
  THPObjectPtr ggb_pvar(ggb ? THPVariable_Wrap(ggb) : Py_None);
  THPObjectPtr gO_pvar(THPVariable_Wrap(gO));
  THPObjectPtr eps_py(PyFloat_FromDouble(eps));
  PyObject* args = PyTuple_Pack(7, input_pvar.get(), weight_pvar.get(),
                                ggi_pvar.get(), ggW_pvar.get(), ggb_pvar.get(),
                                gO_pvar.get(), eps_py.get());
  THPObjectPtr r(PyObject_CallObject(THPBatchNormBackwardBackwardFunction, args));
  if (!r) throw python_error();
  if (!PyTuple_Check(r.get())) {
    throw std::runtime_error("expected PyTuple return from BatchNormBackwardBackward");
  }

  auto gI_var = getReturnTupleVar(r, 0);
  auto gG_var = getReturnTupleVar(r, 1);
  auto ggO_var = getReturnTupleVar(r, 2);

  if (weight_var) {
    return {ggO_var, gI_var, gG_var};
  } else {
    return {ggO_var, gI_var};
  }
};

auto BatchNormBackwardBackward::releaseVariables() -> void {
  input.data.reset();
  weight.data.reset();
  grad_output.data.reset();
}


}} // namespace torch::autograd
