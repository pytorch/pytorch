#pragma once

#include <torch/csrc/python_headers.h>

namespace torch {
namespace autograd {
void initEnumTag(PyObject* module);
}
} // namespace torch
