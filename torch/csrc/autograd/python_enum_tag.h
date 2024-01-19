#pragma once

#include <torch/csrc/python_headers.h>

namespace torch::autograd {
void initEnumTag(PyObject* module);
}
