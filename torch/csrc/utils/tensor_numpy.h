#pragma once

#include <Python.h>
#include <ATen/ATen.h>

namespace torch { namespace utils {

PyObject* tensor_to_numpy(const at::Tensor& tensor);
at::Tensor tensor_from_numpy(PyObject* obj);

}} // namespace torch::utils
