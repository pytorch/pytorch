#pragma once

#include <Python.h>
#include "torch/csrc/jit/tracer.h"

PyObject * THPTracer_enter(PyObject *_unused, PyObject* args);
PyObject * THPTracer_exit(PyObject *_unused, PyObject* args);
PyObject * THPTracer_createAutogradClosure(PyObject *_unused, PyObject *pygraph);
