#pragma once

#include <ATen/ATen.h>
#include <Python.h>

namespace torch { namespace utils {

at::Tensor tensor_new(const at::Type& type, PyObject* args, PyObject* kwargs);

}} // namespace torch::utils
