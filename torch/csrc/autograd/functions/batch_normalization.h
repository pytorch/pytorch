#pragma once

#include <Python.h>
#include <memory>
#include <ATen/ATen.h>

#include "torch/csrc/autograd/function.h"
#include "torch/csrc/autograd/variable.h"
#include "torch/csrc/autograd/primspec.h"

namespace torch { namespace autograd {

struct BatchNormParams {
  at::Tensor cached_running_mean;
  at::Tensor cached_running_var;
  bool training;
  double momentum;
  double eps;
  bool cudnn_enabled;
};

struct BatchNormForward : public ForwardFunction<>, public BatchNormParams, public HasPrimSpec {
  BatchNormForward(BatchNormParams params)
    : BatchNormParams(std::move(params)) {}

  virtual variable_list apply(const variable_list& inputs) override;
  HAS_PRIMSPEC;
};

struct BatchNormBackward : public Function, public BatchNormParams {
  BatchNormBackward(
      FunctionFlags flags,
      BatchNormParams params,
      at::Tensor save_mean,
      at::Tensor save_std,
      const std::shared_ptr<Variable> &input,
      const std::shared_ptr<Variable> &weight,
      const std::shared_ptr<Variable> &bias)
    : Function(std::move(flags))
    , BatchNormParams(std::move(params)) {
      if (is_executable) {
        this->save_mean = std::move(save_mean);
        this->save_std = std::move(save_std);
        this->input = input->save(this);
        this->weight = Variable::save_opt(weight.get(), this);
        this->bias = Variable::save_opt(bias.get(), this);
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
      const std::shared_ptr<Variable> &input,
      const std::shared_ptr<Variable> &weight,
      const std::shared_ptr<Variable> &grad_output)
    : Function(std::move(flags))
    , BatchNormParams(std::move(params)) {
      if (is_executable) {
        this->save_mean = std::move(save_mean);
        this->save_std = std::move(save_std);
        this->input = input->save(this);
        this->weight = Variable::save_opt(weight.get(), this);
        this->grad_output = grad_output->save(this);
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
