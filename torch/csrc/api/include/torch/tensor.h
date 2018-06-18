#pragma once

#include <ATen/ATen.h>

#include <torch/csrc/autograd/variable.h>

namespace torch {
// TODO: Rename to `Tensor`.
using Variable = autograd::Variable;
} // namespace torch
