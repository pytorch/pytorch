#pragma once

#include <torch/csrc/python_headers.h>
#include <ATen/ATen.h>

namespace torch { namespace utils {

PyObject* tensor_to_list(const at::Tensor& tensor);

}} // namespace torch::utils
