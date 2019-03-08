#pragma once

#include <torch/csrc/python_headers.h>

#include <ATen/ATen.h>

namespace torch { namespace utils {

at::Tensor legacy_tensor_ctor(const at::Type& type, at::ScalarType scalar_type, PyObject* args, PyObject* kwargs);
at::Tensor legacy_tensor_new(const at::Type& type, at::ScalarType scalar_type, PyObject* args, PyObject* kwargs);
at::Tensor indexing_tensor_from_data(
    const at::Type& type,
    at::ScalarType scalar_type,
    c10::optional<at::Device> device,
    PyObject* data);
at::Tensor sparse_coo_tensor_ctor(const at::Type& type, at::ScalarType scalar_type, PyObject* args, PyObject* kwargs);
at::Tensor tensor_ctor(const at::Type& type, at::ScalarType scalar_type, PyObject* args, PyObject* kwargs);
at::Tensor as_tensor(const at::Type& type, at::ScalarType scalar_type, PyObject* args, PyObject* kwargs);
at::Tensor new_tensor(const at::Type& type, at::ScalarType scalar_type, PyObject* args, PyObject* kwargs);
at::Tensor new_empty(const at::Type& type, at::ScalarType scalar_type, PyObject* args, PyObject* kwargs);
at::Tensor new_full(const at::Type& type, at::ScalarType scalar_type, PyObject* args, PyObject* kwargs);
at::Tensor new_ones(const at::Type& type, at::ScalarType scalar_type, PyObject* args, PyObject* kwargs);
at::Tensor new_zeros(const at::Type& type, at::ScalarType scalar_type, PyObject* args, PyObject* kwargs);

}} // namespace torch::utils
