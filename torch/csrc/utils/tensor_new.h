#pragma once

#include <Python.h>
#include <ATen/ATen.h>

namespace torch { namespace utils {

at::Tensor legacy_tensor_ctor(const at::Type& type, PyObject* args, PyObject* kwargs);
at::Tensor new_tensor(const at::Type& type, PyObject* args, PyObject* kwargs);

}} // namespace torch::utils
