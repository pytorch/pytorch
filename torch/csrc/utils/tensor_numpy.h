#pragma once

#include <torch/csrc/python_headers.h>
#include <ATen/ATen.h>

namespace torch { namespace utils {

PyObject* tensor_to_numpy(const at::Tensor& tensor);
at::Tensor tensor_from_numpy(PyObject* obj);

at::ScalarType numpy_dtype_to_aten(int dtype);

bool is_numpy_scalar(PyObject* obj);

at::Tensor tensor_from_cuda_array_interface(PyObject* obj);

}} // namespace torch::utils
