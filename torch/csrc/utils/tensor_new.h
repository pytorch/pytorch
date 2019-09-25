#pragma once

#include <torch/csrc/python_headers.h>

#include <ATen/ATen.h>

namespace torch { namespace utils {

at::Tensor legacy_tensor_ctor(c10::TensorTypeId type_id, at::ScalarType scalar_type, PyObject* args, PyObject* kwargs);
at::Tensor legacy_tensor_new(c10::TensorTypeId type_id, at::ScalarType scalar_type, PyObject* args, PyObject* kwargs);
at::Tensor indexing_tensor_from_data(
    c10::TensorTypeId type_id,
    at::ScalarType scalar_type,
    c10::optional<at::Device> device,
    PyObject* data);
at::Tensor sparse_coo_tensor_ctor(c10::TensorTypeId type_id, at::ScalarType scalar_type, PyObject* args, PyObject* kwargs);
at::Tensor tensor_ctor(c10::TensorTypeId type_id, at::ScalarType scalar_type, PyObject* args, PyObject* kwargs);
at::Tensor as_tensor(c10::TensorTypeId type_id, at::ScalarType scalar_type, PyObject* args, PyObject* kwargs);
at::Tensor new_tensor(c10::TensorTypeId type_id, at::ScalarType scalar_type, PyObject* args, PyObject* kwargs);
at::Tensor new_ones(c10::TensorTypeId type_id, at::ScalarType scalar_type, PyObject* args, PyObject* kwargs);
at::Tensor new_zeros(c10::TensorTypeId type_id, at::ScalarType scalar_type, PyObject* args, PyObject* kwargs);

}} // namespace torch::utils
