#pragma once

#include <Python.h>
#include <memory>
#include <string>

#include "torch/csrc/autograd/function.h"
#include "torch/csrc/autograd/variable.h"
#include "torch/csrc/autograd/symbolic.h"

namespace torch { namespace autograd {

struct Error : public Function {
  Error(std::string msg, FunctionFlags&& flags)
    : Function(std::move(flags))
    , msg(std::move(msg)) {}

  Error(std::string msg)
    : msg(std::move(msg)) {}

  virtual variable_list apply(const variable_list& inputs) override;

  std::string msg;
};

// Identity in forward, Error in backward. Used to implement @once_differentiable
struct DelayedError : public Function {
  DelayedError(std::string msg)
    : msg(std::move(msg)) {};

  virtual variable_list apply(const variable_list& inputs) override;

  std::string msg;
};

struct GraphRoot : public Function {
  GraphRoot(function_list functions, variable_list inputs)
    : outputs(std::move(inputs)) {
      next_functions = std::move(functions);
    };

  virtual variable_list apply(const variable_list& inputs) {
    return outputs;
  }

  variable_list outputs;
};

}}
