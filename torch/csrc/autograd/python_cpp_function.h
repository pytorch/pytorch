#pragma once

#include <Python.h>
#include <memory>
#include <typeinfo>

#include "torch/csrc/autograd/function.h"
#include "torch/csrc/utils/object_ptr.h"
#include "torch/csrc/Exceptions.h"

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
  HANDLE_TH_ERRORS
  new (&f->cdata) std::shared_ptr<Function>(Ctor()(args));
  END_HANDLE_TH_ERRORS
  if (!f->cdata) {
    return NULL;
  }
  return obj.release();
}

PyTypeObject* _initFunctionPyTypeObject(PyTypeObject& type, const char* name);

PyObject* registerFunctionHook(Function& fn, PyObject* hook);

template<typename Ctor>
PyTypeObject* createForwardFunctionPyTypeObject(PyTypeObject& type, const char* name)
{
  type.tp_new = &CppFunction_pynew<Ctor>;
  return _initFunctionPyTypeObject(type, name);
}

void registerCppFunction(const std::type_info& type, PyTypeObject* pytype);
PyObject* functionToPyObject(std::shared_ptr<Function> cdata);

}} // namespace torch::autograd
