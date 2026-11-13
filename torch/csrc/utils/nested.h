#pragma once

#include <torch/csrc/python_headers.h>
#include <torch/csrc/utils/python_arg_parser.h>

#include <ATen/core/Tensor.h>

namespace torch::utils {

at::Tensor nested_tensor_ctor(
    c10::DispatchKey dispatch_key,
    at::ScalarType scalar_type,
    PythonArgs& r);

} // namespace torch::utils
