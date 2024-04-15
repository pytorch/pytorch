#pragma once

#include <torch/csrc/python_headers.h>
#include <torch/csrc/utils/python_arg_parser.h>

#include <ATen/core/Tensor.h>

namespace torch {
namespace utils {

// NOTE: [torch.tensor, lift_fresh, and device movement]
//
// The `only_lift_cpu_tensors` flag controls what happens on torch.tensor([1, 2,
// 3], device="cuda") (or any non-CPU devices).
//
// If false (default):
// - the data gets moved into a CPU Tensor
// - then, it gets moved to cuda (via .to)
// - finally, we call lift_fresh() on it.
// Steps 1 and 2 happen with all modes disabled.
//
// If true:
// - the data gets moved into a CPU Tensor (with correct dtype)
// - we call lift_fresh() on it
// - finally, we move it to cuda (via .to)
// Step 1 happens with all modes disabled.
//
// `only_lift_cpu_tensors=true` is useful to prevent CUDA initialization under
// FakeTensorMode because it avoids moving concrete data to CUDA.
TORCH_API bool only_lift_cpu_tensors();
TORCH_API void set_only_lift_cpu_tensors(bool value);

at::Tensor base_tensor_ctor(PyObject* args, PyObject* kwargs);
at::Tensor legacy_tensor_ctor(
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
    c10::optional<at::Device> device,
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
    c10::optional<c10::ScalarType> dtype,
    c10::optional<c10::Device> device,
    c10::optional<bool> copy,
    bool requires_grad);
} // namespace utils
} // namespace torch
