#pragma once

#include <torch/csrc/python_headers.h>
#include <memory>
#include <typeinfo>

#include <torch/csrc/Exceptions.h>
#include <torch/csrc/autograd/function.h>
#include <torch/csrc/utils/object_ptr.h>

namespace torch {
namespace autograd {

struct THPCppFunction {
  PyObject_HEAD std::shared_ptr<Node> cdata;
};

template <typename Ctor>
PyObject* CppFunction_pynew(
    PyTypeObject* type,
    PyObject* args,
    PyObject* kwds) {
  THPObjectPtr obj(type->tp_alloc(type, 0));
  if (!obj)
    return nullptr;
  THPCppFunction* f = (THPCppFunction*)obj.get();
  HANDLE_TH_ERRORS
  new (&f->cdata) std::shared_ptr<Node>(Ctor()(args));
  END_HANDLE_TH_ERRORS
  if (!f->cdata) {
    return nullptr;
  }
  return obj.release();
}

#define THP_FUNCTION_DEFAULT_METHODS                                           \
  {(char*)"_register_hook_dict",                                               \
   THPCppFunction_register_hook_dict,                                          \
   METH_O,                                                                     \
   nullptr},                                                                   \
      {(char*)"register_hook", THPCppFunction_register_hook, METH_O, nullptr}, \
      {(char*)"register_prehook",                                              \
       THPCppFunction_register_prehook,                                        \
       METH_O,                                                                 \
       nullptr},                                                               \
      {(char*)"name", THPCppFunction_name, METH_NOARGS, nullptr}, {            \
    (char*)"_sequence_nr", THPCppFunction_sequence_nr, METH_NOARGS, nullptr    \
  }

#define THP_FUNCTION_DEFAULT_PROPERTIES                                   \
  {(char*)"next_functions",                                               \
   THPCppFunction_next_functions,                                         \
   nullptr,                                                               \
   nullptr,                                                               \
   nullptr},                                                              \
      {(char*)"requires_grad",                                            \
       THPCppFunction_requires_grad,                                      \
       nullptr,                                                           \
       nullptr,                                                           \
       nullptr},                                                          \
  {                                                                       \
    (char*)"metadata", THPCppFunction_metadata, nullptr, nullptr, nullptr \
  }

PyObject* THPCppFunction_next_functions(PyObject* self, void* _unused);
PyObject* THPCppFunction_metadata(PyObject* self, void* _unused);
PyObject* THPCppFunction_requires_grad(PyObject* self, void* _unused);
PyObject* THPCppFunction_register_hook_dict(PyObject* self, PyObject* _var);
PyObject* THPCppFunction_register_hook(PyObject* self, PyObject* hook);
PyObject* THPCppFunction_register_prehook(PyObject* self, PyObject* hook);

PyObject* THPCppFunction_name(PyObject* self, PyObject* noargs);
PyObject* THPCppFunction_sequence_nr(PyObject* self, PyObject* noargs);

PyTypeObject* _initFunctionPyTypeObject(
    PyTypeObject& type,
    const char* name,
    PyGetSetDef* function_properties,
    PyMethodDef* function_methods);

PyObject* registerFunctionHook(Node& fn, PyObject* hook);

PyObject* registerFunctionPreHook(Node& fn, PyObject* hook);

template <typename Ctor>
PyTypeObject* createForwardFunctionPyTypeObject(
    PyTypeObject& type,
    const char* name,
    PyGetSetDef* function_properties = nullptr,
    PyMethodDef* function_methods = nullptr) {
  type.tp_new = &CppFunction_pynew<Ctor>;
  return _initFunctionPyTypeObject(
      type, name, function_properties, function_methods);
}

void registerCppFunction(const std::type_info& type, PyTypeObject* pytype);
PyObject* functionToPyObject(const std::shared_ptr<Node>& cdata);

bool THPCppFunction_Check(PyObject* obj);

} // namespace autograd
} // namespace torch
