#pragma once

// Instantiates torch._C._LegacyVariableBase, which defines the Python
// construct (__new__) for torch.autograd.Variable.

#include <Python.h>

namespace torch { namespace autograd {

void init_legacy_variable(PyObject *module);

}}  // namespace torch::autograd
