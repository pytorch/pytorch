#pragma once

#include <torch/csrc/python_headers.h>

#include <ATen/core/Tensor.h>

namespace torch { namespace utils {

// Unlike its method brethren legacy_tensor_new, this takes an optional
// pair of backend and scalartype, because this function is called from
// two contexts: Tensor() constructor (nullopt), and
// FloatTensor/CUDAFloatTensor/etc.  Also, technically, this isn't legacy
// anymore, because calling into this constructor is a supported use
// case for tensor subclasses (where being able to interpose on class
// construction is idiomatic.)
at::Tensor legacy_tensor_ctor(c10::optional<std::pair<c10::Backend, at::ScalarType>> backend_and_scalar_type, PyObject* args, PyObject* kwargs);

at::Tensor legacy_tensor_new(c10::TensorOptions self_options, PyObject* args, PyObject* kwargs);
at::Tensor indexing_tensor_from_data(c10::TensorOptions options, PyObject* data);
at::Tensor sparse_coo_tensor_ctor(PyObject* args, PyObject* kwargs);
at::Tensor _sparse_coo_tensor_unsafe_ctor(PyObject* args, PyObject* kwargs);
at::Tensor sparse_csr_tensor_ctor(PyObject* args, PyObject* kwargs);
at::Tensor _sparse_csr_tensor_unsafe_ctor(PyObject* args, PyObject* kwargs);
at::Tensor tensor_ctor(PyObject* args, PyObject* kwargs);
at::Tensor as_tensor(PyObject* args, PyObject* kwargs);
at::Tensor new_tensor(c10::TensorOptions options, PyObject* args, PyObject* kwargs);
at::Tensor tensor_frombuffer(PyObject* buffer, at::ScalarType dtype, int64_t count, int64_t offset, bool requires_grad);
at::Tensor tensor_fromDLPack(PyObject *data);
at::Tensor asarray(PyObject* obj, c10::optional<c10::ScalarType> dtype, c10::optional<c10::Device> device, c10::optional<bool> copy, bool requires_grad);
}} // namespace torch::utils
