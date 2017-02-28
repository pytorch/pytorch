#include "torch/csrc/autograd/python_hook.h"

#include "THP.h"
#include "torch/csrc/autograd/python_variable.h"
#include "torch/csrc/utils/auto_gil.h"
#include "torch/csrc/utils/object_ptr.h"
#include "torch/csrc/Exceptions.h"

namespace torch { namespace autograd {

PyGradHook::PyGradHook(PyObject* dict) : dict(dict) {
  Py_INCREF(dict);
}

PyGradHook::~PyGradHook() {
    AutoGIL gil;
    Py_DECREF(dict);
}

auto PyGradHook::operator()(const std::shared_ptr<Variable>& _grad) -> std::shared_ptr<Variable> {
  AutoGIL gil;

  THPObjectPtr grad = THPVariable_Wrap(_grad);
  if (!grad) throw python_error();

  PyObject *key, *value;
  Py_ssize_t pos = 0;
  while (PyDict_Next(dict, &pos, &key, &value)) {
    THPObjectPtr res = PyObject_CallFunctionObjArgs(value, grad.get(), nullptr);
    if (!res) throw python_error();
    if (res == Py_None) continue;
    if (!PyObject_IsInstance(res.get(), THPVariableClass)) {
      PyErr_Format(PyExc_TypeError, "expected Variable, but hook returned '%s'",
          THPUtils_typename(res.get()));
      throw python_error();
    }
    grad = std::move(res);
  }
  return ((THPVariable*)grad.get())->cdata;
}

}} // namespace torch::autograd
