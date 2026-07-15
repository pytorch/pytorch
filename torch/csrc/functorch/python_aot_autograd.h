#pragma once

#include <torch/csrc/python_headers.h>

namespace torch::functorch::impl {

bool InitializeAOTAutogradHelpers(PyObject* module);

} // namespace torch::functorch::impl
