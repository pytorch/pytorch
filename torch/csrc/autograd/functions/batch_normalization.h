#pragma once

#include <Python.h>
#include <memory>
#include <ATen/ATen.h>

#include "torch/csrc/autograd/function.h"
#include "torch/csrc/autograd/variable.h"
#include "torch/csrc/autograd/symbolic.h"
#include "torch/csrc/autograd/saved_variable.h"

namespace torch { namespace autograd {

struct BatchNormParams {
  at::Tensor running_mean;
  at::Tensor running_var;
  bool training;
  double momentum;
  double eps;
  bool cudnn_enabled;
};

struct BatchNormForward : public ForwardFunction<>, public BatchNormParams, public HasSymbolic {
  BatchNormForward(BatchNormParams params)
    : BatchNormParams(std::move(params)) {}

  virtual variable_list apply(const variable_list& inputs) override;
  virtual jit::node_list symbolic(SymbolicContext* ctx, jit::node_list inputs) override;
};

struct BatchNormBackward : public Function, public BatchNormParams {
  BatchNormBackward(
      FunctionFlags flags,
      BatchNormParams params,
      at::Tensor save_mean,
      at::Tensor save_std,
      Variable input,
      Variable weight,
      Variable bias)
    : Function(std::move(flags))
    , BatchNormParams(std::move(params)) {
      if (is_executable) {
        this->save_mean = std::move(save_mean);
        this->save_std = std::move(save_std);
        this->input = SavedVariable(input, this);
        this->weight = SavedVariable(weight, this);
        this->bias = SavedVariable(bias, this);
      }
    }

  virtual variable_list apply(const variable_list& gradOutputs) override;

  virtual void releaseVariables() override;

  at::Tensor save_mean;
  at::Tensor save_std;
  SavedVariable input;
  SavedVariable weight;
  SavedVariable bias;
};

struct BatchNormBackwardBackward : public Function, public BatchNormParams {
  BatchNormBackwardBackward(
      FunctionFlags flags,
      BatchNormParams params,
      at::Tensor save_mean,
      at::Tensor save_std,
      Variable input,
      Variable weight,
      Variable grad_output)
    : Function(std::move(flags))
    , BatchNormParams(std::move(params)) {
      if (is_executable) {
        this->save_mean = std::move(save_mean);
        this->save_std = std::move(save_std);
        this->input = SavedVariable(input, this);
        this->weight = SavedVariable(weight, this);
        this->grad_output = SavedVariable(grad_output, this);
      }
    }

  virtual variable_list apply(const variable_list& grad_grad_inputs) override;

  virtual void releaseVariables() override;

  at::Tensor save_mean;
  at::Tensor save_std;
  SavedVariable input;
  SavedVariable weight;
  SavedVariable grad_output;
};

}}
