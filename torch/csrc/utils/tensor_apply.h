#pragma once

#include <Python.h>
#include <ATen/ATen.h>

namespace torch { namespace utils {

at::Tensor & apply_(at::Tensor & self, PyObject* fn);
at::Tensor & map_(at::Tensor & self, const at::Tensor & other, PyObject* fn);
at::Tensor & map2_(at::Tensor & self, const at::Tensor & other1,
                   const at::Tensor & other2, PyObject* fn);

}} // namespace torch::utils
