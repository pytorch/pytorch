#pragma once

#include <torch/csrc/jit/tracer.h>
#include <torch/csrc/python_headers.h>
#include <torch/csrc/utils/pybind.h>

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
Node* preRecordPythonTrace(
    THPObjectPtr pyobj,
    const std::string& arg_types,
    at::ArrayRef<autograd::Variable> inputs,
    pyobj_list scalar_args);

std::shared_ptr<Graph> createGraphByTracing(
    const py::function& func,
    TypedStack inputs,
    const py::function& var_name_lookup_fn,
    bool force_outplace,
    const std::shared_ptr<script::Module>& self = nullptr);
} // namespace tracer
} // namespace jit
} // namespace torch
