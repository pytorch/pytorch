#include "torch/csrc/autograd/python_cpp_function.h"

#include <Python.h>
#include <memory>
#include <stdio.h>
#include <THPP/THPP.h>
#include <typeindex>
#include <unordered_map>

#include "torch/csrc/autograd/python_function.h"
#include "torch/csrc/autograd/python_variable.h"
#include "torch/csrc/utils/auto_gil.h"
#include "torch/csrc/DynamicTypes.h"
#include "torch/csrc/Exceptions.h"

using namespace torch::autograd;

namespace torch { namespace autograd {

namespace {

PyObject* THPCppFunction_call(PyObject* self, PyObject* args, PyObject *kwargs)
{
  if (kwargs && PyDict_Size(kwargs) != 0) {
    return PyErr_Format(PyExc_TypeError, "keyword arguments are not supported");
  }

  int num_inputs = PyTuple_GET_SIZE(args);
  variable_list vars(num_inputs);
  for (int i = 0; i != num_inputs; ++i) {
    PyObject* arg = PyTuple_GET_ITEM(args, i);
    if (arg == Py_None) {
      continue;
    }
    if (!THPVariable_Check(arg)) {
      return PyErr_Format(PyExc_TypeError, "argument %d is not a Variable", i);
    }
    vars[i] = ((THPVariable*)arg)->cdata;
  }

  variable_list output;

  HANDLE_TH_ERRORS {
    AutoNoGIL nogil;
    output = ((THPCppFunction*)self)->cdata->apply(vars);
  }
  END_HANDLE_TH_ERRORS

  int num_outputs = output.size();
  if (num_outputs == 1) {
    // assume we want to unpack one element tuples for now
    return THPVariable_Wrap(output[0]);
  }

  THPObjectPtr tuple = PyTuple_New(num_outputs);
  for (int i = 0; i != num_outputs; ++i) {
    PyTuple_SET_ITEM(tuple.get(), i, THPVariable_Wrap(output[i]));
  }
  return tuple.release();
}

void THPCppFunction_dealloc(PyObject* self)
{
  ((THPCppFunction*)self)->cdata.~shared_ptr();
  Py_TYPE(self)->tp_free(self);
}

} // namespace

int TensorConverter(PyObject* obj, std::unique_ptr<thpp::Tensor>* address)
{
  try {
    *address = createTensor(obj);
  } catch (std::exception& e) {
    PyErr_Format(PyExc_TypeError,
        "expected a tensor, got %s", Py_TYPE(obj)->tp_name);
    return 0;
  }
  return 1;
}

PyTypeObject* _initFunctionPyTypeObject(PyTypeObject& type, const char* name)
{
  type.tp_flags = Py_TPFLAGS_DEFAULT;
  type.tp_name = name;
  type.tp_basicsize = sizeof(THPCppFunction);
  type.tp_call = THPCppFunction_call;
  type.tp_dealloc = THPCppFunction_dealloc;
  if (PyType_Ready(&type) < 0) {
    auto msg = std::string("Unable to instantiate PyTypeObject for ") + name;
    throw std::runtime_error(msg);
  }
  return &type;
}

static std::unordered_map<std::type_index, THPObjectPtr> cpp_function_types;

PyObject* functionToPyObject(std::shared_ptr<Function> cdata)
{
  if (auto pfw = dynamic_cast<PyFunction*>(cdata.get())) {
    PyObject* obj = pfw->obj;
    Py_INCREF(obj);
    return obj;
  }

  if (auto var = std::dynamic_pointer_cast<Variable>(cdata)) {
    return THPVariable_Wrap(var);
  }

  auto it = cpp_function_types.find(std::type_index(typeid(*cdata)));
  if (it == cpp_function_types.end()) {
    return PyErr_Format(PyExc_TypeError,
        "Don't know how to create Python object for %s", typeid(*cdata).name());
  }

  PyTypeObject* type = (PyTypeObject*)it->second.get();
  THPObjectPtr obj = type->tp_alloc(type, 0);
  if (!obj) return NULL;
  THPCppFunction* f = (THPCppFunction*)obj.get();
  new (&f->cdata) std::shared_ptr<Function>(cdata);
  if (!f->cdata) {
    return NULL;
  }
  return obj.release();
}

void registerCppFunction(const std::type_info& type, PyTypeObject* pytype)
{
  Py_INCREF((PyObject*)pytype);
  cpp_function_types[std::type_index(type)] = THPObjectPtr((PyObject*)pytype);
}

}} // namespace torch::autograd
