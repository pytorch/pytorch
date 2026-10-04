#pragma once

#include <torch/csrc/Export.h>
#include <torch/csrc/python_headers.h>
#include <torch/csrc/utils/python_arg_parser.h>

#include <ATen/core/Tensor.h>

namespace torch::utils {

// NOTE: [torch.tensor, lift_fresh, and device movement]
//
// The `only_lift_cpu_tensors` flag controls when torch.tensor moves newly
// constructed data to the requested device. Sequence data normally starts on
// CPU, Meta targets are constructed directly on Meta, and storage inputs start
// on the storage's device.
//
// If false (default):
// - the tensor gets moved to the requested device (via .to)
// - then, we call lift_fresh() on it.
// The move happens with all modes disabled.
//
// If true:
// - the tensor is converted to the correct dtype on its initial device
// - we call lift_fresh() on it
// - finally, we move it to the requested device unless it is already on the
//   canonical CPU or Meta device
// The dtype conversion happens with all modes disabled.
//
// `only_lift_cpu_tensors=true` is useful to prevent CUDA initialization under
// FakeTensorMode because it avoids moving concrete data to CUDA.
TORCH_PYTHON_API bool only_lift_cpu_tensors();
TORCH_PYTHON_API void set_only_lift_cpu_tensors(bool value);

at::Tensor base_tensor_ctor(PyObject* args, PyObject* kwargs);
TORCH_PYTHON_API at::Tensor legacy_tensor_ctor(
    c10::DispatchKey dispatch_key,
    at::ScalarType scalar_type,
    PyObject* args,
    PyObject* kwargs);
at::Tensor legacy_tensor_new(
    c10::DispatchKey dispatch_key,
    at::ScalarType scalar_type,
    PyObject* args,
    PyObject* kwargs);
at::Tensor indexing_tensor_from_data(
    c10::TensorOptions options,
    at::ScalarType scalar_type,
    std::optional<at::Device> device,
    PyObject* data);
at::Tensor sparse_coo_tensor_ctor(
    c10::DispatchKey dispatch_key,
    at::ScalarType scalar_type,
    PythonArgs& r);
void _validate_sparse_coo_tensor_args(
    c10::DispatchKey dispatch_key,
    at::ScalarType scalar_type,
    PyObject* args,
    PyObject* kwargs);

at::Tensor sparse_compressed_tensor_ctor(
    c10::DispatchKey dispatch_key,
    at::ScalarType scalar_type,
    PythonArgs& r);
at::Tensor sparse_csr_tensor_ctor(
    c10::DispatchKey dispatch_key,
    at::ScalarType scalar_type,
    PythonArgs& r);
at::Tensor sparse_csc_tensor_ctor(
    c10::DispatchKey dispatch_key,
    at::ScalarType scalar_type,
    PythonArgs& r);
at::Tensor sparse_bsr_tensor_ctor(
    c10::DispatchKey dispatch_key,
    at::ScalarType scalar_type,
    PythonArgs& r);
at::Tensor sparse_bsc_tensor_ctor(
    c10::DispatchKey dispatch_key,
    at::ScalarType scalar_type,
    PythonArgs& r);

void _validate_sparse_compressed_tensor_args(
    c10::DispatchKey dispatch_key,
    at::ScalarType scalar_type,
    PyObject* args,
    PyObject* kwargs);
void _validate_sparse_csr_tensor_args(
    c10::DispatchKey dispatch_key,
    at::ScalarType scalar_type,
    PyObject* args,
    PyObject* kwargs);
void _validate_sparse_csc_tensor_args(
    c10::DispatchKey dispatch_key,
    at::ScalarType scalar_type,
    PyObject* args,
    PyObject* kwargs);
void _validate_sparse_bsr_tensor_args(
    c10::DispatchKey dispatch_key,
    at::ScalarType scalar_type,
    PyObject* args,
    PyObject* kwargs);
void _validate_sparse_bsc_tensor_args(
    c10::DispatchKey dispatch_key,
    at::ScalarType scalar_type,
    PyObject* args,
    PyObject* kwargs);

at::Tensor tensor_ctor(
    c10::DispatchKey dispatch_key,
    at::ScalarType scalar_type,
    PythonArgs& r);
at::Tensor as_tensor(
    c10::DispatchKey dispatch_key,
    at::ScalarType scalar_type,
    PythonArgs& r);
at::Tensor new_tensor(
    c10::DispatchKey dispatch_key,
    at::ScalarType scalar_type,
    PyObject* args,
    PyObject* kwargs);
at::Tensor new_ones(
    c10::DispatchKey dispatch_key,
    at::ScalarType scalar_type,
    PyObject* args,
    PyObject* kwargs);
at::Tensor tensor_frombuffer(
    PyObject* buffer,
    at::ScalarType dtype,
    int64_t count,
    int64_t offset,
    bool requires_grad);
at::Tensor tensor_fromDLPack(PyObject* data);
at::Tensor asarray(
    PyObject* obj,
    std::optional<c10::ScalarType> dtype,
    std::optional<c10::Device> device,
    std::optional<bool> copy,
    std::optional<bool> requires_grad);
} // namespace torch::utils
