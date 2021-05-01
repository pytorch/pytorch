#pragma once

#include <torch/csrc/utils/python_strings.h>
#include <ATen/Tensor.h>
#include <torch/library.h>

namespace torch {
namespace fx {


void initFx(PyObject* module);

} // namespace fx
} // namespace torch
