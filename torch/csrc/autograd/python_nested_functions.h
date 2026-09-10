#pragma once

#include <torch/csrc/utils/python_compat.h>
namespace torch::autograd {

PyMethodDef* get_nested_functions_manual();

void initNestedFunctions(PyObject* module);

} // namespace torch::autograd
