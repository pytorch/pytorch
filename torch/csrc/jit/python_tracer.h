#pragma once

#include <Python.h>
#include <memory>
#include "torch/csrc/jit/tracer.h"

PyObject * THPTracer_enter(PyObject *_unused, PyObject* args);
PyObject * THPTracer_exit(PyObject *_unused, PyObject* args);
PyObject * THPTracer_createAutogradClosure(PyObject *_unused, PyObject *pygraph);

struct THPTracingState {
  PyObject_HEAD
  std::shared_ptr<torch::jit::tracer::TracingState> cdata;
};

PyObject * THPTracingState_Wrap(const std::shared_ptr<torch::jit::tracer::TracingState> state);

bool THPTracingState_Check(PyObject *obj);

bool THPTracer_initModule(PyObject *module);
