#pragma once

#include <torch/csrc/autograd/generated/variable_factories.h>
#include <torch/csrc/jit/custom_operator.h>

#include <ATen/ATen.h>

namespace torch {
using jit::createOperator;
using jit::RegisterOperators;
} // namespace torch
