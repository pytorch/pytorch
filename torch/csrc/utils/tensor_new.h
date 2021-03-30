#pragma once

#include <torch/csrc/python_headers.h>

#include <ATen/ATen.h>

namespace torch { namespace utils {

at::Tensor legacy_tensor_ctor(c10::DispatchKey dispatch_key, at::ScalarType scalar_type, PyObject* args, PyObject* kwargs);
at::Tensor legacy_tensor_new(c10::DispatchKey dispatch_key, at::ScalarType scalar_type, PyObject* args, PyObject* kwargs);
at::Tensor indexing_tensor_from_data(
    c10::TensorOptions options,
    at::ScalarType scalar_type,
    c10::optional<at::Device> device,
    PyObject* data);
at::Tensor sparse_coo_tensor_ctor(c10::DispatchKey dispatch_key, at::ScalarType scalar_type, PyObject* args, PyObject* kwargs);
at::Tensor _sparse_coo_tensor_unsafe_ctor(c10::DispatchKey dispatch_key, at::ScalarType scalar_type, PyObject* args, PyObject* kwargs);
void _validate_sparse_coo_tensor_args(c10::DispatchKey dispatch_key, at::ScalarType scalar_type, PyObject* args, PyObject* kwargs);
at::Tensor tensor_ctor(c10::DispatchKey dispatch_key, at::ScalarType scalar_type, PyObject* args, PyObject* kwargs);
at::Tensor as_tensor(c10::DispatchKey dispatch_key, at::ScalarType scalar_type, PyObject* args, PyObject* kwargs);
at::Tensor new_tensor(c10::DispatchKey dispatch_key, at::ScalarType scalar_type, PyObject* args, PyObject* kwargs);
at::Tensor new_ones(c10::DispatchKey dispatch_key, at::ScalarType scalar_type, PyObject* args, PyObject* kwargs);

}} // namespace torch::utils
