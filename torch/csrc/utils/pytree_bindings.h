#pragma once

#include <torch/csrc/utils/pybind.h>

namespace torch {
namespace pytree {
void init_bindings(PyObject* module);
}
} // namespace torch
