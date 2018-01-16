#ifndef THP_AUTOGRAD_H
#define THP_AUTOGRAD_H

PyObject * THPAutograd_initExtension(PyObject *_unused);
bool THPAutograd_initFunctions(PyObject* module);

namespace torch { namespace autograd {

void initAutogradClosureBindings(PyObject* module);

PyMethodDef* python_functions();

}}

#include "torch/csrc/autograd/python_function.h"
#include "torch/csrc/autograd/python_variable.h"
#include "torch/csrc/autograd/python_engine.h"

#endif
