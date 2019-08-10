#pragma once

#include <torch/csrc/python_headers.h>

namespace torch {
namespace autograd {
namespace amp {

PyMethodDef* python_functions();

} // namespace torch
} // namespace autograd
} // namespace amp
