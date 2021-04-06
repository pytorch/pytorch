#pragma once

#include <torch/csrc/python_headers.h>

namespace torch {
namespace distributed {
namespace spmd {

PyMethodDef* python_functions();

} // namespace spmd
} // namespace distributed
} // namespace torch
