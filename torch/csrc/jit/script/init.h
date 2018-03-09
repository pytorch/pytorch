#pragma once

#include "torch/csrc/jit/pybind.h"

namespace torch {
namespace jit {
namespace script {
void initJitScriptBindings(PyObject* module);
std::shared_ptr<Graph> createGraphByTracing(
        py::function func,
        std::vector<tracer::TraceInput> inputs,
        size_t num_inputs);

} // namespace script
} // namespace jit
} // namespace torch
