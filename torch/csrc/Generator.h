#pragma once

#include <ATen/ATen.h>
#include <torch/csrc/python_headers.h>

namespace torch { namespace python {

void initGeneratorBindings(PyObject* module);

}} // namespace torch::python
