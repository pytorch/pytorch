#pragma once

#include <Python.h>
#include <memory>
#include <ATen/ATen.h>

#include "torch/csrc/autograd/function.h"
#include "torch/csrc/autograd/variable.h"

namespace torch { namespace autograd {

struct BatchNormParams {
  at::Tensor running_mean;
  at::Tensor running_var;
  bool training;
  double momentum;
  double eps;
  bool cudnn_enabled;
};

struct BatchNormForward : public Function, public BatchNormParams {
  BatchNormForward(BatchNormParams params)
    : BatchNormParams(std::move(params)) {}

  virtual variable_list apply(const variable_list& inputs) override;
};

struct BatchNormBackward : public Function, public BatchNormParams {
  BatchNormBackward(
      FunctionFlags flags,
      BatchNormParams params,
      at::Tensor save_mean,
      at::Tensor save_std,
      SavedVariable input,
      SavedVariable weight,
      SavedVariable bias)
    : Function(std::move(flags))
    , BatchNormParams(std::move(params)) {
      if (is_executable) {
        this->save_mean = std::move(save_mean);
        this->save_std = std::move(save_std);
        this->input = std::move(input);
        this->weight = std::move(weight);
        this->bias = std::move(bias);
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
      SavedVariable input,
      SavedVariable weight,
      SavedVariable grad_output)
    : Function(std::move(flags))
    , BatchNormParams(std::move(params)) {
      if (is_executable) {
        this->save_mean = std::move(save_mean);
        this->save_std = std::move(save_std);
        this->input = std::move(input);
        this->weight = std::move(weight);
        this->grad_output = std::move(grad_output);
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
