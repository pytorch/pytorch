#pragma once

#include <ATen/ATen.h>
#include <Python.h>

namespace torch { namespace utils {

PyObject* tensor_to_list(const at::Tensor& tensor);

}} // namespace torch::utils
