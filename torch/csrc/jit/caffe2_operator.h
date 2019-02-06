#pragma once

#include <torch/csrc/jit/operator.h>

namespace torch {
namespace jit {

Operator createOperatorFromCaffe2(const std::string& name);

}} // torch::jit
