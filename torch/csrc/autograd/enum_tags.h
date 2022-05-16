#pragma once

#include <torch/csrc/python_headers.h>

namespace torch {
    namespace autograd {
    void initEnumTags(PyObject* module);
}} // namespace torch::autograd
