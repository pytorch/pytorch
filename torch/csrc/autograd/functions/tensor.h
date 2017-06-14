#pragma once

#include <Python.h>
#include <THPP/THPP.h>
#include <memory>

#include "torch/csrc/autograd/function.h"
#include "torch/csrc/autograd/variable.h"

namespace torch { namespace autograd {

struct Identity : public Function {
  Identity(FunctionFlags&& f)
    : Function(std::move(f)) {};

  virtual variable_list apply(const variable_list& inputs) override;
};

struct Clone : public Function {
  Clone() {}

  virtual variable_list apply(const variable_list& inputs) override;
};

}}


