
copy: fbcode/caffe2/torch/csrc/jit/python/python_tracer.h
copyrev: 4cf00205ba3fa7d514e5ee47d2d6282938675c4b

#pragma once

#include <torch/csrc/jit/tracer.h>
#include <torch/csrc/python_headers.h>
#include <torch/csrc/utils/pybind.h>
#include <torch/csrc/jit/frontend/source_range.h>

#include <memory>
#include <string>

namespace torch {
namespace jit {

namespace script {
  struct Module;
}

namespace tracer {
void initPythonTracerBindings(PyObject* module);

std::string getPythonInterpreterStackTrace();
SourceRange getPythonInterpreterSourceRange();

Node* preRecordPythonTrace(
    THPObjectPtr pyobj,
    const std::string& arg_types,
    at::ArrayRef<autograd::Variable> inputs,
    std::vector<THPObjectPtr> scalar_args);

std::pair<std::shared_ptr<Graph>, Stack> createGraphByTracing(
    const py::function& func,
    Stack inputs,
    const py::function& var_name_lookup_fn,
    bool force_outplace,
    script::Module* self = nullptr);
} // namespace tracer
} // namespace jit
} // namespace torch
