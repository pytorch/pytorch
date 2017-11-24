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
      is_executable = true;
    };

  virtual variable_list apply(const variable_list& inputs) {
    return outputs;
  }

  variable_list outputs;
};

struct Add : public ForwardFunction<true>, public HasSymbolic {
  Add() {}

  virtual variable_list apply(const variable_list& inputs) override;
  virtual jit::value_list symbolic(
      SymbolicContext* ctx,
      jit::value_list inputs,
      std::shared_ptr<jit::SourceLocation> sl
  ) override;
};


struct AddBackward_Deprecated : public Function {
  AddBackward_Deprecated(FunctionFlags&& flags)
    : Function(std::move(flags)) {}

  virtual variable_list apply(const variable_list& gradOutputs) override;
  virtual bool is_traceable() override { return true; }
};

struct Mul : public ForwardFunction<> {
  Mul() {}

  virtual variable_list apply(const variable_list& inputs) override;
};

struct MulBackward : public Function {
  MulBackward(FunctionFlags&& flags)
    : Function(std::move(flags)) {}

  virtual variable_list apply(const variable_list& gradOutputs) override;
};

}}
