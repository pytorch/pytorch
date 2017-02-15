#pragma once

#include <Python.h>
#include <memory>
#include <typeinfo>

#include "torch/csrc/autograd/function.h"
#include "torch/csrc/utils/object_ptr.h"

namespace torch { namespace autograd {

struct THPCppFunction {
  PyObject_HEAD
  std::shared_ptr<Function> cdata;
};

template<typename Ctor>
PyObject* CppFunction_pynew(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
  THPObjectPtr obj = type->tp_alloc(type, 0);
  if (!obj) return NULL;
  THPCppFunction* f = (THPCppFunction*)obj.get();
  new (&f->cdata) std::shared_ptr<Function>(Ctor()(args));
  if (!f->cdata) {
    return NULL;
  }
  return obj.release();
}

PyTypeObject* _initFunctionPyTypeObject(PyTypeObject& type, const char* name);

template<typename Ctor>
PyTypeObject* createForwardFunctionPyTypeObject(PyTypeObject& type, const char* name)
{
  type.tp_new = &CppFunction_pynew<Ctor>;
    return _initFunctionPyTypeObject(type, name);
}

// conversion utilities for PyArg_ParseTuple
int TensorConverter(PyObject* obj, std::unique_ptr<thpp::Tensor>* address);

void registerCppFunction(const std::type_info& type, PyTypeObject* pytype);
PyObject* functionToPyObject(std::shared_ptr<Function> cdata);

}} // namespace torch::autograd
