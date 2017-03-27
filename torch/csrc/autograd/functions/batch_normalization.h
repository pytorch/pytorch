#pragma once

#include <memory>
#include <THPP/THPP.h>

#include "torch/csrc/autograd/function.h"
#include "torch/csrc/autograd/variable.h"

namespace torch { namespace autograd {

struct BatchNormParams {
  std::shared_ptr<thpp::Tensor> running_mean;
  std::shared_ptr<thpp::Tensor> running_var;
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
      std::unique_ptr<thpp::Tensor> save_mean,
      std::unique_ptr<thpp::Tensor> save_std,
      SavedVariable input,
      SavedVariable weight,
      SavedVariable bias)
    : Function(std::move(flags))
    , BatchNormParams(std::move(params))
    , save_mean(std::move(save_mean))
    , save_std(std::move(save_std))
    , input(std::move(input))
    , weight(std::move(weight))
    , bias(std::move(bias)) {}

  virtual variable_list apply(const variable_list& gradOutputs) override;

  virtual void releaseVariables() override;

  std::unique_ptr<thpp::Tensor> save_mean;
  std::unique_ptr<thpp::Tensor> save_std;
  SavedVariable input;
  SavedVariable weight;
  SavedVariable bias;
};

}}
