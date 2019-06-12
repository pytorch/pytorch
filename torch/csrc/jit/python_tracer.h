#pragma once

#include "torch/csrc/python_headers.h"
#include <memory>
#include "torch/csrc/jit/tracer.h"
#include "torch/csrc/utils/pybind.h"

namespace torch { namespace jit { namespace tracer {
void initPythonTracerBindings(PyObject *module);


std::string getPythonInterpreterStackTrace();
Node* preRecordPythonTrace(
    THPObjectPtr pyobj, std::string arg_types, at::ArrayRef<autograd::Variable> inputs,
    pyobj_list scalar_args);

std::shared_ptr<Graph> createGraphByTracing(
    py::function func,
    Stack inputs,
    py::function var_name_lookup_fn,
    c10::optional<size_t> num_real_inputs = {});
} // namespace tracer

}} // namespace torch::jit
