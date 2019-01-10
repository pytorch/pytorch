#pragma once

#include "torch/csrc/python_headers.h"
#include <memory>
#include "torch/csrc/jit/tracer.h"

namespace torch { namespace jit {
void initPythonTracerBindings(PyObject *module);
}} // namespace torch::jit
