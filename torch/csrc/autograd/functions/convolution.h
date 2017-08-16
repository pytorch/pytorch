#pragma once

#include <Python.h>
#include <ATen/ATen.h>
#include <memory>
#include <vector>
#include <iostream>

#include "torch/csrc/autograd/function.h"
#include "torch/csrc/autograd/variable.h"
#include "torch/csrc/autograd/primspec.h"

#ifdef WITH_CUDNN
#include "torch/csrc/cudnn/Conv.h"
#else
namespace torch { namespace cudnn {
struct Convolution {};
}}
#endif

namespace torch { namespace autograd {

struct ConvParams {
  std::vector<int> stride;
  std::vector<int> padding;
  std::vector<int> dilation;
  bool transposed;
  std::vector<int> output_padding;
  int groups;
  bool benchmark;
  bool cudnn_enabled;

  bool is_dilated() const;
  bool is_output_padding_neg() const;
  bool is_output_padding_big() const;
  bool is_padding_neg() const;
  void view1d_as_2d();
  bool use_cudnn(const at::Tensor& input) const;
};

struct ConvForward : public Function, public ConvParams, public HasPrimSpec {
  explicit ConvForward(ConvParams params) : ConvParams(std::move(params)) {}

  virtual std::string name() override;
  virtual variable_list apply(const variable_list& inputs) override;
  virtual void primspec(PrimSpecContext* ctx, jit::node_list inputs, jit::node_list outputs);

  std::vector<int64_t> output_size(at::Tensor& input, at::Tensor& weight);
};

struct ConvBackward : public Function, public ConvParams {
  ConvBackward(
      FunctionFlags flags,
      ConvParams params,
      const std::shared_ptr<Variable>& input,
      const std::shared_ptr<Variable>& weight,
      const std::shared_ptr<Variable>& bias,
      tensor_list columns,
      tensor_list ones,
      std::unique_ptr<torch::cudnn::Convolution> convolution)
    : Function(std::move(flags))
    , ConvParams(std::move(params))
    , convolution(std::move(convolution)) {
      if (is_executable) {
        this->input_ = input->save(this);
        this->weight_ = weight->save(this);
        this->bias_ = Variable::save_opt(bias.get(), this);
        this->columns = std::move(columns);
        this->ones = std::move(ones);
      }
    }

  virtual variable_list apply(const variable_list& gradOutputs) override;

  virtual void releaseVariables() override;

  SavedVariable input_;
  SavedVariable weight_;
  SavedVariable bias_;
  tensor_list columns;
  tensor_list ones;
  std::unique_ptr<torch::cudnn::Convolution> convolution;
};

struct ConvBackwardBackward : public Function, public ConvParams {
  ConvBackwardBackward(
      FunctionFlags flags,
      ConvParams params,
      const std::shared_ptr<Variable>& input,
      const std::shared_ptr<Variable>& weight,
      const std::shared_ptr<Variable>& bias,
      const std::shared_ptr<Variable>& grad_output)
    : Function(std::move(flags))
    , ConvParams(std::move(params)) {
      if (is_executable) {
        this->input_ = input->save(this);
        this->weight_ = weight->save(this);
        this->bias_ = Variable::save_opt(bias.get(), this);
        this->grad_output_ = grad_output->save(this);
      }
    }

  virtual variable_list apply(const variable_list& grad_grad_inputs) override;

  virtual void releaseVariables() override;

  SavedVariable input_;
  SavedVariable weight_;
  SavedVariable bias_;
  SavedVariable grad_output_;
};

}} // namespace torch::autograd
