#pragma once

#include <torch/csrc/lazy/core/tensor.h>

namespace torch {
namespace lazy {

//////////////////////////////////////////////////////////////////////////////
// ATEN operators follows here, listed in alphabetical order.
//////////////////////////////////////////////////////////////////////////////

torch::lazy::LazyTensorPtr expand(
    const torch::lazy::LazyTensorPtr& input,
    std::vector<int64_t> size);

// Fills the input with the given value.
void fill_(torch::lazy::LazyTensorPtr& input, const at::Scalar& value);

void copy_(torch::lazy::LazyTensorPtr& input, torch::lazy::LazyTensorPtr& src);

} // namespace lazy
} // namespace torch
