#ifndef THP_AUTOGRAD_H
#define THP_AUTOGRAD_H

PyObject* THPAutograd_initExtension(PyObject* _unused, PyObject* unused);
void THPAutograd_initFunctions();

namespace torch::autograd {

PyMethodDef* python_functions();

}

#include <torch/csrc/autograd/python_engine.h>
#include <torch/csrc/autograd/python_function.h>
#include <torch/csrc/autograd/python_variable.h>

#endif
