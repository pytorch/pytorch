#pragma once

namespace torch {
namespace autograd {

PyMethodDef* get_nested_functions_manual();

void initNestedFunctions(PyObject* module);

} // namespace autograd
} // namespace torch
