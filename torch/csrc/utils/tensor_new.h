#pragma once

#include <Python.h>
#include <ATen/ATen.h>

namespace torch { namespace utils {

at::Tensor legacy_tensor_ctor(const at::Type& type, PyObject* args, PyObject* kwargs);
at::Tensor legacy_tensor_new(const at::Type& type, PyObject* args, PyObject* kwargs);
at::Tensor legacy_new_from_data(const at::Type& type, int device, PyObject *data);
at::Tensor sparse_coo_tensor_ctor(const at::Type& type, PyObject* args, PyObject* kwargs);
at::Tensor tensor_ctor(const at::Type& type, PyObject* args, PyObject* kwargs);
at::Tensor new_tensor(const at::Type& type, PyObject* args, PyObject* kwargs);
at::Tensor new_empty(const at::Type& type, PyObject* args, PyObject* kwargs);
at::Tensor new_full(const at::Type& type, PyObject* args, PyObject* kwargs);
at::Tensor new_ones(const at::Type& type, PyObject* args, PyObject* kwargs);
at::Tensor new_zeros(const at::Type& type, PyObject* args, PyObject* kwargs);

}} // namespace torch::utils
