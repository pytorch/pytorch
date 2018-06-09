#pragma once

#include "torch/csrc/python_headers.h"
#include <memory>
#include "torch/csrc/jit/tracer.h"
#include "torch/csrc/utils/pybind.h"

namespace torch { namespace jit { namespace tracer {
void initPythonTracerBindings(PyObject *module);


std::string getPythonInterpreterStackTrace();
tracer::PreTraceInfo preRecordPythonTrace(
    THPObjectPtr pyobj, std::string arg_types, at::ArrayRef<autograd::Variable> inputs,
    pyobj_list scalar_args);

std::shared_ptr<Graph> createGraphByTracing(
        py::function func,
        autograd::variable_list inputs,
        size_t num_inputs);
} // namespace tracer

}} // namespace torch::jit
