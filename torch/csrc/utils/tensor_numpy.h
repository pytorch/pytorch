#pragma once

#include <torch/csrc/python_headers.h>
#include <ATen/core/Tensor.h>

namespace torch { namespace utils {

PyObject* tensor_to_numpy(const at::Tensor& tensor);
at::Tensor tensor_from_numpy(PyObject* obj, bool warn_if_not_writeable=true);

int aten_to_numpy_dtype(const at::ScalarType scalar_type);
at::ScalarType numpy_dtype_to_aten(int dtype);

bool is_numpy_available();
bool is_numpy_int(PyObject* obj);
bool is_numpy_scalar(PyObject* obj);

void warn_numpy_not_writeable();
at::Tensor tensor_from_cuda_array_interface(PyObject* obj);

}} // namespace torch::utils
