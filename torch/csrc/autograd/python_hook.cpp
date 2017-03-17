#include "torch/csrc/autograd/python_hook.h"

#include <sstream>

#include "THP.h"
#include "torch/csrc/autograd/python_variable.h"
#include "torch/csrc/utils/auto_gil.h"
#include "torch/csrc/utils/object_ptr.h"
#include "torch/csrc/Exceptions.h"
#include <THPP/THPP.h>

using thpp::Tensor;
using torch::autograd::variable_list;

static THPObjectPtr wrap_variables(const variable_list& grads);
static variable_list unwrap_variables(PyObject* grads);
static std::string hook_name(PyObject* hook);
static void check_result(PyObject* original, PyObject* result, PyObject* hook);
static void check_single_result(PyObject* original, PyObject* result, PyObject* hook);


namespace torch { namespace autograd {

PyFunctionPreHook::PyFunctionPreHook(PyObject* dict, int grad_index)
  : dict(dict)
  , grad_index(grad_index)
{
  Py_INCREF(dict);
}

PyFunctionPreHook::~PyFunctionPreHook() {
  AutoGIL gil;
  Py_DECREF(dict);
}

auto PyFunctionPreHook::operator()(const variable_list& _grads) -> variable_list
{
  AutoGIL gil;

  THPObjectPtr grad = THPVariable_Wrap(_grads.at(grad_index));
  if (!grad) throw python_error();

  PyObject *key, *hook;
  Py_ssize_t pos = 0;
  while (PyDict_Next(dict, &pos, &key, &hook)) {
    THPObjectPtr res = PyObject_CallFunctionObjArgs(hook, grad.get(), nullptr);
    if (!res) throw python_error();
    if (res == Py_None) continue;
    check_single_result(grad.get(), res.get(), hook);
    grad = std::move(res);
  }

  variable_list results(_grads);
  results[grad_index] = ((THPVariable*)grad.get())->cdata;
  return results;
}

PyFunctionPostHook::PyFunctionPostHook(PyObject* dict) : dict(dict) {
  Py_INCREF(dict);
}

PyFunctionPostHook::~PyFunctionPostHook() {
  AutoGIL gil;
  Py_DECREF(dict);
}

auto PyFunctionPostHook::operator()(
    const variable_list& _grad_inputs,
    const variable_list& _grad_outputs) -> variable_list
{
  AutoGIL gil;

  THPObjectPtr grad_inputs = wrap_variables(_grad_inputs);
  THPObjectPtr grad_outputs = wrap_variables(_grad_outputs);

  PyObject *key, *hook;
  Py_ssize_t pos = 0;
  while (PyDict_Next(dict, &pos, &key, &hook)) {
    THPObjectPtr res = PyObject_CallFunctionObjArgs(
        hook, grad_inputs.get(), grad_outputs.get(), nullptr);
    if (!res) throw python_error();
    if (res == Py_None) continue;
    check_result(grad_inputs, res, hook);
    grad_inputs = std::move(res);
  }

  return unwrap_variables(grad_inputs.get());
}

}} // namespace torch::autograd


static THPObjectPtr wrap_variables(const variable_list& grads)
{
  THPObjectPtr tuple = PyTuple_New(grads.size());
  if (!tuple) throw python_error();
  for (size_t i = 0; i < grads.size(); i++) {
    THPObjectPtr grad = THPVariable_Wrap(grads[i]);
    if (!grad) throw python_error();
    PyTuple_SET_ITEM(tuple.get(), i, grad.release());
  }
  return tuple;
}

static variable_list unwrap_variables(PyObject* grads)  {
  variable_list results(PyTuple_GET_SIZE(grads));
  for (size_t i = 0; i < results.size(); i++) {
    PyObject* item = PyTuple_GET_ITEM(grads, i);
    if (item == Py_None) {
      continue;
    } else if (THPVariable_Check(item)) {
      results[i] = ((THPVariable*)item)->cdata;
    } else {
      // this should never happen, but just in case...
      std::stringstream ss;
      ss << "expected variable but got " << Py_TYPE(item)->tp_name;
      throw std::runtime_error(ss.str());
    }
  }
  return results;
}

static void check_result(PyObject* prev, PyObject* result, PyObject* hook) {
  if (!PyTuple_Check(result)) {
    PyErr_Format(PyExc_TypeError, "expected tuple, but hook returned '%s'",
        THPUtils_typename(result));
    throw python_error();
  }

  auto prev_size = PyTuple_GET_SIZE(prev);
  auto result_size = PyTuple_GET_SIZE(result);
  if (prev_size != result_size) {
    std::stringstream ss;
    auto name = hook_name(hook);
    ss << "backward hook '" << name << "' has returned an incorrect number ";
    ss << "of gradients (got " << result_size << ", but expected ";
    ss << prev_size << ")";
    throw std::runtime_error(ss.str());
  }

  for (auto i = 0; i < prev_size; i++) {
    check_single_result(PyTuple_GET_ITEM(prev, i), PyTuple_GET_ITEM(result, i), hook);
  }
}

static void check_single_result(PyObject* _original, PyObject* _result, PyObject* hook) {
  if (!PyObject_IsInstance(_result, THPVariableClass)) {
    PyErr_Format(PyExc_TypeError, "expected Variable, but hook returned '%s'",
        THPUtils_typename(_result));
    throw python_error();
  }

  auto& original = *((THPVariable*)_original)->cdata->data;
  auto& result = *((THPVariable*)_result)->cdata->data;

  if (original.type() != result.type()) {
    std::stringstream ss;
    auto name = hook_name(hook);
    ss << "backward hook '" << name << "' has changed the type of grad_input (";
    ss << "was " << thpp::toString(original.type()) << " got ";
    ss << thpp::toString(result.type()) << ")";
    throw std::runtime_error(ss.str());
  }

  if (original.isCuda() != result.isCuda()) {
    std::stringstream ss;
    auto name = hook_name(hook);
    ss << "backward hook '" << name << "' has changed the type of grad_input";
    if (original.isCuda()) {
      ss << " (was CUDA tensor got CPU tensor)";
    } else {
      ss << " (was CPU tensor got CUDA tensor)";
    }
    throw std::runtime_error(ss.str());
  }

  if (original.sizes() != result.sizes()) {
    std::stringstream ss;
    auto name = hook_name(hook);
    ss << "backward hook '" << name << "' has changed the size of grad_input";
    throw std::runtime_error(ss.str());
  }
}

static std::string hook_name(PyObject* hook) {
  THPObjectPtr name = PyObject_GetAttrString(hook, "__name__");
#if PY_MAJOR_VERSION == 2
  if (name && PyString_Check(name.get())) {
    return std::string(PyString_AS_STRING(name.get()));
  }
#else
  if (name && PyUnicode_Check(name.get())) {
    THPObjectPtr tmp = PyUnicode_AsASCIIString(name.get());
    return std::string(PyBytes_AS_STRING(tmp.get()));
  }
#endif
  return "<unknown>";
}
