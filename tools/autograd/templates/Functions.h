#pragma once

// ${generated_comment}

#include <ATen/ATen.h>

#include "torch/csrc/autograd/function.h"
#include "torch/csrc/autograd/variable.h"
#include "torch/csrc/autograd/saved_variable.h"

namespace torch { namespace autograd {

using at::Scalar;
using at::Tensor;
using at::IntList;

// avoid mutiply if scalar is 1.
inline Tensor maybe_multiply(const Tensor & t, const Scalar & s);

${autograd_function_declarations}

}} // namespace torch::autograd
