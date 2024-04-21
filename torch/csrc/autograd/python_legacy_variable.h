#pragma once

// Instantiates torch._C._LegacyVariableBase, which defines the Python
// constructor (__new__) for torch.autograd.Variable.

#include <torch/csrc/python_headers.h>

namespace torch::autograd {

void init_legacy_variable(PyObject* module);

}
