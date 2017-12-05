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
#include <ATen/cudnn/cudnn-wrapper.h>
#endif

namespace {
    void check_dims_match_num_input_features(const std::string& arg_name, int64_t expected, int64_t actual){
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

  AutoGPU guard(inputs[0]);
  auto& input_var = inputs[0];
  auto& weight_var = inputs[1];
  auto& bias_var = inputs[2];

  auto input = input_var.data();
  auto weight = weight_var.opt_data();
  auto bias = bias_var.opt_data();

  auto num_features = input.sizes()[1];
  check_dims_match_num_input_features("running_mean", num_features, running_mean.numel());
  check_dims_match_num_input_features("running_var", num_features, running_var.numel());
  if (weight.defined()) {
    check_dims_match_num_input_features("weight", num_features, weight.numel());
  }
  if (bias.defined()) {
    check_dims_match_num_input_features("bias", num_features, bias.numel());
  }

  bool use_cudnn = false;
#ifdef WITH_CUDNN
  use_cudnn = (input.type().is_cuda()
               && (input.type().scalarType() != at::kHalf
               || weight.type().scalarType() == at::kFloat)
               && weight.defined() && bias.defined()
               && input.size(0) <= 131070
               && cudnn_enabled && CUDNN_VERSION >= 5110L);
#endif

  Tensor output;
  auto save_mean = running_mean.type().tensor(running_mean.sizes());
  auto save_std = running_var.type().tensor(running_var.sizes());

  if (use_cudnn && eps >= CUDNN_BN_MIN_EPSILON) {
#ifdef WITH_CUDNN
    output = at::cudnn_batch_norm_forward(
                input, weight, bias,
                running_mean, running_var, save_mean, save_std,
                training, momentum, eps);
#endif
  } else {
    output = at::batch_norm_forward(
        input, weight, bias,
        running_mean, running_var, training, momentum, eps,
        save_mean, save_std);
  }

  auto outputs = as_tensor_list(std::move(output));
  return wrap_outputs(inputs, std::move(outputs), [&](FunctionFlags f) {
    return std::make_shared<BatchNormBackward>(
        f, *this, std::move(save_mean), std::move(save_std),
        input_var, weight_var, bias_var);
  });
};

auto BatchNormBackward::apply(const variable_list& grad_outputs) -> variable_list {
  check_input_variables("BatchNormBackward", grad_outputs, 1);
  auto input_var = this->input.unpack();
  auto weight_var = this->weight.unpack();
  auto bias_var = this->bias.unpack();

  auto input = input_var.data();
  auto weight = weight_var.opt_data();
  auto bias = bias_var.opt_data();

  AutoGPU guard(input);

  bool use_cudnn = false;
#ifdef WITH_CUDNN
  use_cudnn = (input.type().backend() == at::kCUDA
               && (input.type().scalarType() != at::kHalf
               || weight.type().scalarType() == at::kFloat)
               && weight.defined() && bias.defined() && training
               && input.size(0) <= 131070
               && cudnn_enabled && CUDNN_VERSION >= 5110L);
#endif

  at::Tensor grad_input;
  at::Tensor grad_weight;
  at::Tensor grad_bias;

  auto grad_output = grad_outputs[0].data().contiguous();

  if (use_cudnn && eps >= CUDNN_BN_MIN_EPSILON) {
#ifdef WITH_CUDNN
    std::tie(grad_input, grad_weight, grad_bias) =
      at::cudnn_batch_norm_backward(
          input, grad_output, weight,
          save_mean, save_std,
          training, eps
          );
#endif
  } else {
    std::array<bool, 3> mask = {
      should_compute_output(0),
      should_compute_output(1),
      should_compute_output(2),
    };
    std::tie(grad_input, grad_weight, grad_bias) = at::batch_norm_backward(
        grad_output, input, weight, running_mean, running_var,
        training, eps, save_mean, save_std,
        mask);
  }

  // Add saved variables used out of the pure autograd to inputs
  variable_list all_inputs(grad_outputs);
  all_inputs.push_back(input_var);
  if (weight.defined()) {
    all_inputs.push_back(weight_var);
  }
  auto outputs =  as_tensor_list(std::move(grad_input),
                                 std::move(grad_weight),
                                 std::move(grad_bias));
  return wrap_outputs(all_inputs, std::move(outputs), [&](FunctionFlags f) {
    return std::make_shared<BatchNormBackwardBackward>(
      f, *this, save_mean, save_std,
      input_var, weight_var,
      grad_outputs[0]);
    });
};

auto BatchNormBackward::releaseVariables() -> void {
  input.data.reset();
  weight.data.reset();
  bias.data.reset();
}

Variable getReturnTupleVar(PyObject *p, Py_ssize_t pos) {
  PyObject *item = PyTuple_GET_ITEM(p, pos);
  if (item != Py_None) {
    return ((THPVariable*)item)->cdata;
  }
  return Variable();
}

auto BatchNormBackwardBackward::apply(const variable_list& grad_grad_inputs) -> variable_list {
  check_input_variables("BatchNormBackwardBackward", grad_grad_inputs, 3, 0);
  auto ggI = grad_grad_inputs[0];
  auto ggW = grad_grad_inputs[1];
  auto ggb = grad_grad_inputs[2];

  auto input_var = input.unpack();
  AutoGPU guard(input_var);

  auto weight_var = weight.unpack();
  auto gO_var = grad_output.unpack();

  auto input = input_var.data();
  AutoGIL gil;

  THPObjectPtr input_pvar(THPVariable_Wrap(input_var));
  THPObjectPtr weight_pvar(THPVariable_Wrap(weight_var));

  THPObjectPtr ggi_pvar(THPVariable_Wrap(ggI));
  THPObjectPtr ggW_pvar(THPVariable_Wrap(ggW));
  THPObjectPtr ggb_pvar(THPVariable_Wrap(ggb));
  THPObjectPtr gO_pvar(THPVariable_Wrap(gO_var));
  THPObjectPtr eps_py(PyFloat_FromDouble(eps));
  THPObjectPtr save_mean_py(createPyObject(save_mean));
  THPObjectPtr save_std_py(createPyObject(save_std));
  THPObjectPtr running_mean_py(createPyObject(running_mean));
  THPObjectPtr running_var_py(createPyObject(running_var));
  PyObject *training_pyo = training ? Py_True : Py_False;

  THPObjectPtr args(PyTuple_Pack(12, input_pvar.get(), weight_pvar.get(),
                                 ggi_pvar.get(), ggW_pvar.get(), ggb_pvar.get(),
                                 gO_pvar.get(), eps_py.get(),
                                 save_mean_py.get(), save_std_py.get(),
                                 running_mean_py.get(), running_var_py.get(),
                                 training_pyo));
  THPObjectPtr r(PyObject_CallObject(THPBatchNormBackwardBackwardFunction, args.get()));
  if (!r) throw python_error();
  if (!PyTuple_Check(r.get())) {
    throw std::runtime_error("expected PyTuple return from BatchNormBackwardBackward");
  }

  auto gI_var = getReturnTupleVar(r, 0);
  auto gG_var = getReturnTupleVar(r, 1);
  auto ggO_var = getReturnTupleVar(r, 2);

  if (weight_var.defined()) {
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
