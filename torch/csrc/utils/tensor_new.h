#pragma once

#include <Python.h>
#include <ATen/ATen.h>

namespace torch { namespace utils {

at::Tensor tensor_new(const at::Type& type, PyObject* args, PyObject* kwargs);
at::Tensor variable_data_factory(const at::Type& type, PyObject* args, PyObject* kwargs);

}} // namespace torch::utils
