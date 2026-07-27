#pragma once

#include <torch/csrc/lazy/core/tensor.h>

namespace torch::lazy {

//////////////////////////////////////////////////////////////////////////////
// ATEN operators follows here, listed in alphabetical order.
//////////////////////////////////////////////////////////////////////////////

void copy_(torch::lazy::LazyTensorPtr& input, torch::lazy::LazyTensorPtr& src);
// Fills the input with the given value.
void fill_(torch::lazy::LazyTensorPtr& input, const at::Scalar& value);

} // namespace torch::lazy
