#pragma once

#include <torch/csrc/python_headers.h>

namespace torch {
namespace jit {
namespace script {

void initTreeViewBindings(PyObject* module);

}
} // namespace jit
} // namespace torch
