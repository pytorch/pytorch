#pragma once

#include <torch/csrc/autograd/custom_function.h>

namespace torch {
  using Variable = autograd::Variable;
  using AutogradContext = autograd::AutogradContext;
  using variable_list = autograd::variable_list;
} //namespace torch
