#pragma once

#include <ATen/ATen.h>
#include <torch/csrc/python_headers.h>

namespace torch {

void initGeneratorBindings(PyObject* module);

}
