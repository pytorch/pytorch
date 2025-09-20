#pragma once

#include <Python.h>

namespace torch::distributed {
void initPlacementBindings(PyObject* module);
} // namespace torch::distributed
