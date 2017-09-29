#pragma once

#include <Python.h>
#include <memory>
#include <ATen/ATen.h>

#include "torch/csrc/autograd/function.h"
#include "torch/csrc/autograd/variable.h"
#include "torch/csrc/autograd/saved_variable.h"

namespace torch { namespace autograd {

// Softmax and LogSoftmax are implemented nearly identically until second
// derivative, so the implementation is contained in a common base class.
template<bool is_log>
struct SoftmaxBase : public ForwardFunction<> {
  SoftmaxBase(int dim)
    : dim(dim) {}

  virtual variable_list apply(const variable_list& inputs) override;

  int dim;
};

template<bool is_log>
struct SoftmaxBackwardBase : public Function {
  SoftmaxBackwardBase(FunctionFlags f, int dim)
    : Function(std::move(f))
    , dim(dim) {}

  virtual variable_list apply(const variable_list& inputs) override;

  SavedVariable saved_output;
  int dim;
};


struct Softmax : public SoftmaxBase<false> {
  using SoftmaxBase::SoftmaxBase;
};
struct SoftmaxBackward : public SoftmaxBackwardBase<false> {
  using SoftmaxBackwardBase::SoftmaxBackwardBase;
};
struct SoftmaxBackwardBackward : public Function {
  using Function::Function;
  virtual variable_list apply(const variable_list& inputs) override;
  SavedVariable saved_output;
  SavedVariable saved_grad_output;
  int dim;
};


struct LogSoftmax : public SoftmaxBase<true> {
  using SoftmaxBase::SoftmaxBase;
};
struct LogSoftmaxBackward : public SoftmaxBackwardBase<true> {
  using SoftmaxBackwardBase::SoftmaxBackwardBase;
};
struct LogSoftmaxBackwardBackward : public Function {
  using Function::Function;
  virtual variable_list apply(const variable_list& inputs) override;
  SavedVariable saved_output;
  SavedVariable saved_grad_output;
  int dim;
};

}}
