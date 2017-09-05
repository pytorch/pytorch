#pragma once

#include <Python.h>
#include <memory>

#include "torch/csrc/autograd/function.h"
#include "torch/csrc/autograd/variable.h"

namespace torch { namespace autograd {

struct AccumulateGrad : public Function {
  AccumulateGrad(std::shared_ptr<Variable> variable);

  virtual variable_list apply(const variable_list& inputs) override;
  void acc_inplace(std::shared_ptr<Variable>& grad,
    std::shared_ptr<Variable>& new_grad);

  std::weak_ptr<Variable> variable;
  std::weak_ptr<Variable> variable_grad;
};

}}
