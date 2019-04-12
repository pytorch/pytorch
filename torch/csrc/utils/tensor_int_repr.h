#pragma once

#include <torch/csrc/python_headers.h>
#include <ATen/ATen.h>

namespace torch { namespace utils {

at::Tensor int_repr(at::Tensor t);

}} // namespace torch::utils
