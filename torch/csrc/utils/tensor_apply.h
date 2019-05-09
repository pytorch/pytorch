#pragma once

#include <torch/csrc/python_headers.h>
#include <ATen/ATen.h>

namespace torch { namespace utils {

at::Tensor & apply_(at::Tensor & self, PyObject* fn);
at::Tensor & map_(at::Tensor & self, const at::Tensor & other_, PyObject* fn);
at::Tensor & map2_(at::Tensor & self, const at::Tensor & x_,
                   const at::Tensor & y_, PyObject* fn);

}} // namespace torch::utils
