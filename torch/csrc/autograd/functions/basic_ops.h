#pragma once

#include <memory>
#include <THPP/THPP.h>

#include "torch/csrc/autograd/function.h"
#include "torch/csrc/autograd/variable.h"

namespace torch { namespace autograd {

struct Add : public Function {
  Add() {}

  virtual variable_list apply(const variable_list& inputs) override;
};

struct AddBackward : public Function {
  AddBackward(FunctionFlags flags)
    : Function(std::move(flags)) {}

  virtual variable_list apply(const variable_list& gradOutputs) override;
};

}}

