#include "torch/csrc/autograd/python_hook.h"

#include <sstream>

#include "THP.h"
#include "torch/csrc/autograd/python_variable.h"
#include "torch/csrc/utils/auto_gil.h"
#include "torch/csrc/utils/object_ptr.h"
#include "torch/csrc/utils/python_strings.h"
#include "torch/csrc/Exceptions.h"

using torch::autograd::variable_list;
using torch::autograd::Variable;

static PyObject* wrap_variables(const variable_list& c_variables);
static variable_list unwrap_variables(PyObject* py_variables);
static std::string hook_name(PyObject* hook);
static void check_result(PyObject* original, PyObject* result, PyObject* hook);
static void check_single_result(PyObject* original, PyObject* result, PyObject* hook);


namespace torch { namespace autograd {

PyFunctionPreHook::PyFunctionPreHook(PyObject* dict, int value_idx)
  : dict(dict)
  , value_idx(value_idx)
{
  Py_INCREF(dict);
}

PyFunctionPreHook::~PyFunctionPreHook() {
  AutoGIL gil;
  Py_DECREF(dict);
}

auto PyFunctionPreHook::operator()(const variable_list& values) -> variable_list
{
  AutoGIL gil;

  THPObjectPtr value(THPVariable_Wrap(values.at(value_idx)));
  if (!value) throw python_error();

  PyObject *key, *hook;
  Py_ssize_t pos = 0;
  while (PyDict_Next(dict, &pos, &key, &hook)) {
    THPObjectPtr res(PyObject_CallFunctionObjArgs(hook, value.get(), nullptr));
    if (!res) throw python_error();
    if (res == Py_None) continue;
    check_single_result(value.get(), res.get(), hook);
    value = std::move(res);
  }

  variable_list results(values);
  if (value != Py_None) results[value_idx] = ((THPVariable*)value.get())->cdata;
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
    const variable_list& _outputs, /* grad_inputs */
    const variable_list& _inputs /* grad_outputs */) -> variable_list
{
  AutoGIL gil;

  THPObjectPtr outputs(wrap_variables(_outputs));
  THPObjectPtr inputs(wrap_variables(_inputs));

  PyObject *key, *hook;
  Py_ssize_t pos = 0;
  while (PyDict_Next(dict, &pos, &key, &hook)) {
    THPObjectPtr res(PyObject_CallFunctionObjArgs(
        hook, outputs.get(), inputs.get(), nullptr));
    if (!res) throw python_error();
    if (res == Py_None) continue;
    check_result(outputs, res, hook);
    outputs = std::move(res);
  }

  return unwrap_variables(outputs.get());
}

}} // namespace torch::autograd


static PyObject *wrap_variables(const variable_list& c_variables)
{
  size_t num_vars = c_variables.size();
  THPObjectPtr tuple(PyTuple_New(num_vars));
  if (!tuple) throw python_error();
  for (size_t i = 0; i < num_vars; ++i) {
    THPObjectPtr var(THPVariable_Wrap(c_variables[i]));
    if (!var) throw python_error();
    PyTuple_SET_ITEM(tuple.get(), i, var.release());
  }
  return tuple.release();
}

static variable_list unwrap_variables(PyObject* py_variables)  {
  variable_list results(PyTuple_GET_SIZE(py_variables));
  for (size_t i = 0; i < results.size(); i++) {
    PyObject* item = PyTuple_GET_ITEM(py_variables, i);
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
    ss << "hook '" << name << "' has returned an incorrect number ";
    ss << "of values (got " << result_size << ", but expected ";
    ss << prev_size << ")";
    throw std::runtime_error(ss.str());
  }

  for (auto i = 0; i < prev_size; i++) {
    check_single_result(PyTuple_GET_ITEM(prev, i), PyTuple_GET_ITEM(result, i), hook);
  }
}

static void check_single_result(PyObject* _original, PyObject* _result, PyObject* hook) {
  if (_result == Py_None) return;

  if (_original == Py_None) {
    throw std::runtime_error("can't replace a None gradient with a non-None value");
  }

  if (!PyObject_IsInstance(_result, THPVariableClass)) {
    PyErr_Format(PyExc_TypeError, "expected Variable, but hook returned '%s'",
        THPUtils_typename(_result));
    throw python_error();
  }

  auto& original = ((THPVariable*)_original)->cdata.data();
  auto& result = ((THPVariable*)_result)->cdata.data();

  if (original.type().ID() != result.type().ID()) {
    std::stringstream ss;
    auto name = hook_name(hook);
    ss << "hook '" << name << "' has changed the type of value (";
    ss << "was " << original.toString() << " got ";
    ss << result.toString() << ")";
    throw std::runtime_error(ss.str());
  }

  if (original.is_cuda() != result.is_cuda()) {
    std::stringstream ss;
    auto name = hook_name(hook);
    ss << "hook '" << name << "' has changed the type of value";
    if (original.is_cuda()) {
      ss << " (was CUDA tensor got CPU tensor)";
    } else {
      ss << " (was CPU tensor got CUDA tensor)";
    }
    throw std::runtime_error(ss.str());
  }

  if (original.sizes().vec() != result.sizes().vec()) {
    std::stringstream ss;
    auto name = hook_name(hook);
    ss << "hook '" << name << "' has changed the size of value";
    throw std::runtime_error(ss.str());
  }
}

static std::string hook_name(PyObject* hook) {
  THPObjectPtr name(PyObject_GetAttrString(hook, "__name__"));
  if (name && THPUtils_checkString(name.get())) {
    return THPUtils_unpackString(name.get());
  }
  return "<unknown>";
}
