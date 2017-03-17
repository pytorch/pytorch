#pragma once

#include <memory>
#include <THPP/THPP.h>

#include "torch/csrc/autograd/function.h"
#include "torch/csrc/autograd/variable.h"

namespace torch { namespace autograd {

struct AccumulateGrad : public Function {
  AccumulateGrad(std::shared_ptr<Variable> variable);

  virtual variable_list apply(const variable_list& inputs) override;

  std::shared_ptr<Variable> variable;
};

}}


