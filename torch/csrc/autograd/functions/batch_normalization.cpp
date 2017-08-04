#include "batch_normalization.h"

#include "torch/csrc/autograd/python_function.h"
#include "torch/csrc/autograd/python_variable.h"
#include "torch/csrc/autograd/variable.h"
#include "torch/csrc/autograd/functions/utils.h"
#include "torch/csrc/autograd/functions/basic_ops.h"
#include "torch/csrc/utils/auto_gil.h"
#include "torch/csrc/utils/auto_gpu.h"
#include "torch/csrc/DynamicTypes.h"
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

#ifndef CUDNN_BN_MIN_EPSILON
#define CUDNN_BN_MIN_EPSILON 0
#endif

auto BatchNormForward::apply(const variable_list& inputs) -> variable_list {
  check_input_variables("BatchNorm", inputs, 3, 1);

  auto& input = inputs[0];
  auto& weight = inputs[1];
  auto& bias = inputs[2];
  AutoGPU guard(input->data);

  auto num_features = input->data.sizes()[1];
  check_dims_match_num_input_features("running_mean", num_features, running_mean.numel());
  check_dims_match_num_input_features("running_var", num_features, running_var.numel());
  if (weight){
    check_dims_match_num_input_features("weight", num_features, weight->data.numel());
  }
  if (bias){
    check_dims_match_num_input_features("bias", num_features, bias->data.numel());
  }

  bool use_cudnn = false;
#ifdef WITH_CUDNN
  use_cudnn = (input->data.type().isCuda()
               && input->data.type().scalarType() != at::kHalf
               && weight && bias
               && cudnn_enabled && CUDNN_VERSION >= 5110L);
#endif

  auto output = input->data.type().tensor(input->data.sizes());
  auto save_mean = running_mean.type().tensor(running_mean.sizes());
  auto save_std = running_var.type().tensor(running_var.sizes());

  if (use_cudnn && eps >= CUDNN_BN_MIN_EPSILON) {
#ifdef WITH_CUDNN
    torch::cudnn::cudnn_batch_norm_forward(
        state,
        torch::cudnn::getCudnnHandle(),
        torch::cudnn::getCudnnDataType(input->data),
        (THVoidTensor*)input->data.unsafeGetTH(false),
        (THVoidTensor*)output.unsafeGetTH(false),
        (THVoidTensor*)weight->data.unsafeGetTH(false),
        (THVoidTensor*)bias->data.unsafeGetTH(false),
        (THVoidTensor*)running_mean.unsafeGetTH(false),
        (THVoidTensor*)running_var.unsafeGetTH(false),
        (THVoidTensor*)save_mean.unsafeGetTH(false),
        (THVoidTensor*)save_std.unsafeGetTH(false),
        training,
        momentum,
        eps);
#endif
  } else {
      at::Tensor nt;
      at::BatchNormalization_updateOutput(
        input->data,
        output,
        weight ? weight->data : nt,
        bias ? bias->data : nt,
        running_mean,
        running_var,
        save_mean,
        save_std,
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

  auto input = input_var->data;
  auto weight = weight_var ? weight_var->data : at::Tensor();
  auto bias = bias_var ? bias_var->data : at::Tensor();

  AutoGPU guard(input);

  bool use_cudnn = false;
#ifdef WITH_CUDNN
  use_cudnn = (input.type().backend() == at::kCUDA
               && input.type().scalarType() != at::kHalf
               && weight.defined() && bias.defined() && training
               && cudnn_enabled && CUDNN_VERSION >= 5110L);
#endif

  at::Tensor grad_input;
  if (should_compute_output(0) || use_cudnn) {
    grad_input = input.type().tensor(input.sizes());
  }

  at::Tensor grad_weight;
  if (should_compute_output(1) || use_cudnn) {
    grad_weight = weight.type().tensor(weight.sizes());
    if (!use_cudnn) {
      grad_weight.zero_();
    }
  }

  at::Tensor grad_bias;
  if (should_compute_output(2) || use_cudnn) {
    grad_bias = bias.type().tensor(bias.sizes());
    if (!use_cudnn) {
      grad_bias.zero_();
    }
  }

  auto grad_output = grad_outputs[0]->data.contiguous();

  if (use_cudnn && eps >= CUDNN_BN_MIN_EPSILON) {
#ifdef WITH_CUDNN
    torch::cudnn::cudnn_batch_norm_backward(
        state,
        torch::cudnn::getCudnnHandle(),
        torch::cudnn::getCudnnDataType(input),
        (THVoidTensor*)input.unsafeGetTH(false),
        (THVoidTensor*)grad_output.unsafeGetTH(false),
        (THVoidTensor*)grad_input.unsafeGetTH(false),
        (THVoidTensor*)grad_weight.unsafeGetTH(false),
        (THVoidTensor*)grad_bias.unsafeGetTH(false),
        (THVoidTensor*)weight.unsafeGetTH(false),
        (THVoidTensor*)running_mean.unsafeGetTH(false),
        (THVoidTensor*)running_var.unsafeGetTH(false),
        (THVoidTensor*)save_mean.unsafeGetTH(false),
        (THVoidTensor*)save_std.unsafeGetTH(false),
        training,
        eps);
#endif
  } else {
    at::BatchNormalization_backward(
        input,
        grad_output,
        grad_input,
        grad_weight,
        grad_bias,
        weight,
        running_mean,
        running_var,
        save_mean,
        save_std,
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
      f, *this, save_mean, save_std,
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
  THPObjectPtr save_mean_py(createPyObject(save_mean));
  THPObjectPtr save_std_py(createPyObject(save_std));
  THPObjectPtr running_mean_py(createPyObject(running_mean));
  THPObjectPtr running_var_py(createPyObject(running_var));
  THPObjectPtr training_py(training ? Py_True : Py_False);

  PyObject* args = PyTuple_Pack(12, input_pvar.get(), weight_pvar.get(),
                                ggi_pvar.get(), ggW_pvar.get(), ggb_pvar.get(),
                                gO_pvar.get(), eps_py.get(),
                                save_mean_py.get(), save_std_py.get(),
                                running_mean_py.get(), running_var_py.get(),
                                training_py.get());
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
