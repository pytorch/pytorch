#include "python_cpp_function_attr.h"

#include "torch/csrc/autograd/python_variable.h"
#include "torch/csrc/autograd/python_hook.h"

namespace torch { namespace autograd {

namespace attributes {

PyObject* next_functions(THPCppFunction* self, PyObject* hook)
{
  auto& next_functions = self->cdata->next_functions;
  auto num_next = next_functions.size();
  THPObjectPtr py_functions = PyTuple_New(num_next);
  if (!py_functions) return NULL;
  for (size_t i = 0; i < num_next; ++i) {
    auto& c_tuple = next_functions[i];
    THPObjectPtr tuple = PyTuple_New(2);
    if (!tuple) return NULL;
    PyObject *py_fn = functionToPyObject(c_tuple.first);
    if (!py_fn) return NULL;
    PyTuple_SET_ITEM(tuple.get(), 0, py_fn);
    PyObject *py_idx = PyLong_FromLong(c_tuple.second);
    if (!py_idx) return NULL;
    PyTuple_SET_ITEM(tuple.get(), 1, py_idx);
    PyTuple_SET_ITEM(py_functions.get(), i, tuple.release());
  }
  return py_functions.release();
}

PyObject* register_hook_dict(PyObject* self, PyObject* _var)
{
  if (!THPVariable_Check(_var)) {
    return PyErr_Format(PyExc_TypeError, "_register_hook_dict expected a variable");
  }
  auto var = (THPVariable*)_var;
  auto& fn = *((THPCppFunction*)self)->cdata;
  fn.pre_hooks.push_back(std::make_shared<PyFunctionPreHook>(
      var->backward_hooks, var->cdata->output_nr));
  Py_RETURN_NONE;
}

PyObject* register_hook(PyObject* self, PyObject* hook)
{
  auto& fn = *((THPCppFunction*)self)->cdata;
  return registerFunctionHook(fn, hook);
}

}

}}



