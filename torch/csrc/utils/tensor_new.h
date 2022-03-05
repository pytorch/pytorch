#pragma once

#include <torch/csrc/python_headers.h>

#include <ATen/core/Tensor.h>

namespace torch { namespace utils {

// backend_and_scalar_type handles calls from contexts like FloatTensor or
// CUDAIntTensor
at::Tensor legacy_tensor_ctor(PyObject* args, PyObject* kwargs, c10::optional<std::pair<c10::Backend, at::ScalarType>> backend_and_scalar_type = c10::nullopt);
// new, on the other hand, takes the tensor options of the self tensor the
// method was invoked from
at::Tensor legacy_tensor_new(PyObject* args, PyObject* kwargs, c10::TensorOptions);
at::Tensor indexing_tensor_from_data(
    c10::TensorOptions options,
    at::ScalarType scalar_type,
    c10::optional<at::Device> device,
    PyObject* data);
at::Tensor sparse_coo_tensor_ctor(c10::DispatchKey dispatch_key, at::ScalarType scalar_type, PyObject* args, PyObject* kwargs);
at::Tensor _sparse_coo_tensor_unsafe_ctor(c10::DispatchKey dispatch_key, at::ScalarType scalar_type, PyObject* args, PyObject* kwargs);
void _validate_sparse_coo_tensor_args(c10::DispatchKey dispatch_key, at::ScalarType scalar_type, PyObject* args, PyObject* kwargs);
at::Tensor sparse_csr_tensor_ctor(c10::DispatchKey dispatch_key, at::ScalarType scalar_type, PyObject* args, PyObject* kwargs);
at::Tensor _sparse_csr_tensor_unsafe_ctor(c10::DispatchKey dispatch_key, at::ScalarType scalar_type, PyObject* args, PyObject* kwargs);
void _validate_sparse_csr_tensor_args(c10::DispatchKey dispatch_key, at::ScalarType scalar_type, PyObject* args, PyObject* kwargs);
at::Tensor tensor_ctor(c10::DispatchKey dispatch_key, at::ScalarType scalar_type, PyObject* args, PyObject* kwargs);
at::Tensor as_tensor(c10::DispatchKey dispatch_key, at::ScalarType scalar_type, PyObject* args, PyObject* kwargs);
at::Tensor new_tensor(c10::DispatchKey dispatch_key, at::ScalarType scalar_type, PyObject* args, PyObject* kwargs);
at::Tensor new_ones(c10::DispatchKey dispatch_key, at::ScalarType scalar_type, PyObject* args, PyObject* kwargs);
at::Tensor tensor_frombuffer(PyObject* buffer, at::ScalarType dtype, int64_t count, int64_t offset, bool requires_grad);
at::Tensor tensor_fromDLPack(PyObject *data);
at::Tensor asarray(PyObject* obj, c10::optional<c10::ScalarType> dtype, c10::optional<c10::Device> device, c10::optional<bool> copy, bool requires_grad);
}} // namespace torch::utils
