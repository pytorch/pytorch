#pragma once

#include <torch/csrc/python_headers.h>

PyMethodDef* THXPModule_methods();

namespace torch::xpu {

void initModule(PyObject* module);

} // namespace torch::xpu
