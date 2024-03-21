#pragma once

#include <torch/csrc/python_headers.h>

namespace torch {
namespace distributed {
namespace c10d {

PyMethodDef* python_functions();

} // namespace c10d
} // namespace distributed
} // namespace torch
