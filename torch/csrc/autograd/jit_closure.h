#pragma once

#include <Python.h>
#include <memory>

#include "torch/csrc/jit/ir.h"
#include "torch/csrc/autograd/function.h"

namespace torch { namespace autograd {

struct AutogradClosure {
  function_list roots;
  std::shared_ptr<Function> output;
};

std::unique_ptr<AutogradClosure> createAutogradClosure(torch::jit::Graph *graph);

}}

