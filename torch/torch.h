#pragma once

#include <Python.h>
#include <ATen/ATen.h>
#include <pybind11/pybind11.h>
#include <torch/csrc/utils/pybind.h>

namespace torch {
at::Tensor as_variable(at::Tensor tensor, bool requires_grad = false);
} // namespace torch
