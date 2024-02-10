#pragma once

#include <ATen/core/Tensor.h>
#include <torch/csrc/python_headers.h>

namespace torch::utils {

PyObject* tensor_to_numpy(const at::Tensor& tensor, bool force = false);
at::Tensor tensor_from_numpy(PyObject* obj, bool warn_if_not_writeable = true);

int aten_to_numpy_dtype(const at::ScalarType scalar_type);
at::ScalarType numpy_dtype_to_aten(int dtype);

bool is_numpy_available();
bool is_numpy_int(PyObject* obj);
bool is_numpy_bool(PyObject* obj);
bool is_numpy_scalar(PyObject* obj);

void warn_numpy_not_writeable();
at::Tensor tensor_from_cuda_array_interface(PyObject* obj);

void validate_numpy_for_dlpack_deleter_bug();
bool is_numpy_dlpack_deleter_bugged();

} // namespace torch::utils
