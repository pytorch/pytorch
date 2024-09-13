#pragma once

#include <torch/csrc/jit/frontend/source_range.h>
#include <torch/csrc/jit/frontend/tracer.h>
#include <torch/csrc/python_headers.h>
#include <torch/csrc/utils/pybind.h>

#include <memory>
#include <string>

namespace torch::jit {

struct Module;

namespace tracer {
void initPythonTracerBindings(PyObject* module);

SourceRange getPythonInterpreterSourceRange();

Node* preRecordPythonTrace(
    THPObjectPtr pyobj,
    const std::string& arg_types,
    at::ArrayRef<autograd::Variable> inputs,
    std::vector<THPObjectPtr> scalar_args);

std::pair<std::shared_ptr<Graph>, Stack> createGraphByTracingWithDict(
    const py::function& func,
    const py::dict& inputs_dict,
    const Stack& inputs,
    const py::function& var_name_lookup_fn,
    bool strict,
    bool force_outplace,
    Module* self = nullptr,
    const std::vector<std::string>& argument_names = {});

std::pair<std::shared_ptr<Graph>, Stack> createGraphByTracing(
    const py::function& func,
    Stack inputs,
    const py::function& var_name_lookup_fn,
    bool strict,
    bool force_outplace,
    Module* self = nullptr,
    const std::vector<std::string>& argument_names = {});
} // namespace tracer
} // namespace torch::jit
