#include <torch/csrc/autograd/python_hook.h>

#include <c10/util/irange.h>
#include <pybind11/pybind11.h>
#include <torch/csrc/Exceptions.h>
#include <torch/csrc/PyInterpreter.h>
#include <torch/csrc/THP.h>
#include <torch/csrc/autograd/python_variable.h>
#include <torch/csrc/dynamo/compiled_autograd.h>
#include <torch/csrc/utils/object_ptr.h>
#include <torch/csrc/utils/pybind.h>
#include <torch/csrc/utils/python_strings.h>

#include <iostream>
#include <sstream>

using torch::autograd::Variable;
using torch::autograd::variable_list;

static PyObject* wrap_variables(const variable_list& c_variables);
static variable_list unwrap_variables(PyObject* py_variables);
static std::string hook_name(PyObject* hook);
static void check_result(PyObject* original, PyObject* result, PyObject* hook);
static void check_single_result(
    PyObject* original,
    PyObject* result,
    PyObject* hook);

namespace torch::autograd {

namespace {

// This function is called in 4 different cases:
//   1) TensorPreHook
//   2) PreHook
//   3) PostHook
//   4) TensorPostAccGradHook
//
// Depending on the case, args and res can hold different types of objects:
//
// args:
// TensorPreHook   (Tensor,)
// PreHook         ((Tensor, ...),)                (grad_outputs,)
// PostHook        ((Tensor, ...), (Tensor, ...))  (grad_inputs, grad_outputs)
// TensorPostAccGradHook  ((Tensor), ())                  (tensor,)
//
// res:
// TensorPreHook          Tensor
// PreHook                ((Tensor, ...),)                (grad_outputs,)
// PostHook               ((Tensor, ...),)                (grad_inputs,)
// TensorPostAccGradHook  None
//
// This function returns True if any hook returned non-None value, and False
// otherwise.
bool _call_hooks(PyObject* dict, PyObject* args) {
  // Note: [Extend Hook Lifetime]
  // Hold a reference to hooks till we iterate over them.
  // This is to handle the case when hook calls `handle.remove` inside it
  // and it's refcount goes to `0`, Python is free to GC it.
  // We hold onto a stale pointer and subsequent call to
  // `check_single_result`, which tries to fetch the `hook`'s name segfaults.
  // So, we use `PyDict_Values` which returns a new reference to the values
  // i.e. we hold the reference to the hooks till we have iterated over them.
  // Reference: https://github.com/pytorch/pytorch/issues/58354
  auto hooks = THPObjectPtr{PyDict_Values(dict)};
  bool is_modified = false;
  const auto len = PyList_Size(hooks);
  for (Py_ssize_t idx = 0; idx < len; ++idx) {
    const auto hook = PyList_GetItem(hooks, idx);

    THPObjectPtr res(PyObject_CallObject(hook, args));
    if (!res)
      throw python_error();
    if (res == Py_None)
      continue;

    PyObject* args0 = PyTuple_GetItem(args, 0);
    if (res == args0)
      continue;

    if (PyTuple_CheckExact(args0)) {
      check_result(args0, res, hook);
    } else {
      check_single_result(args0, res, hook);
    }
    PyTuple_SetItem(args, 0, res.release());

    is_modified = true;
  }
  return is_modified;
}

} // namespace

PyFunctionTensorPreHook::PyFunctionTensorPreHook(
    PyObject* dict,
    size_t value_idx)
    : dict(dict), value_idx(value_idx) {
  Py_INCREF(dict);
}

// NOLINTNEXTLINE(bugprone-exception-escape)
PyFunctionTensorPreHook::~PyFunctionTensorPreHook() {
  // If python is already dead, leak the wrapped python objects
  if (Py_IsInitialized()) {
    pybind11::gil_scoped_acquire gil;
    Py_DECREF(dict);
  }
}

auto PyFunctionTensorPreHook::operator()(const variable_list& values)
    -> variable_list {
  pybind11::gil_scoped_acquire gil;
  THPObjectPtr value(THPVariable_Wrap(values.at(value_idx)));
  if (!value)
    throw python_error();
  THPObjectPtr tup(PyTuple_New(1));
  PyTuple_SET_ITEM(tup.get(), 0, value.release());
  bool is_tup_modified = _call_hooks(dict, tup.get());
  variable_list results(values);
  if (is_tup_modified) {
    results[value_idx] = THPVariable_Unpack(PyTuple_GetItem(tup.get(), 0));
  }
  return results;
}

PyFunctionPreHook::PyFunctionPreHook(PyObject* dict) : dict(dict) {
  Py_INCREF(dict);
}

// NOLINTNEXTLINE(bugprone-exception-escape)
PyFunctionPreHook::~PyFunctionPreHook() {
  // If python is already dead, leak the wrapped python objects
  if (Py_IsInitialized()) {
    pybind11::gil_scoped_acquire gil;
    Py_DECREF(dict);
  }
}

auto PyFunctionPreHook::operator()(const variable_list& grad_outputs_)
    -> variable_list {
  pybind11::gil_scoped_acquire gil;
  THPObjectPtr grad_outputs(wrap_variables(grad_outputs_));
  THPObjectPtr tup(PyTuple_New(1));
  PyTuple_SET_ITEM(tup.get(), 0, grad_outputs.release());
  _call_hooks(dict, tup.get());
  return unwrap_variables(PyTuple_GetItem(tup.get(), 0));
}

PyFunctionPostHook::PyFunctionPostHook(PyObject* dict) : dict(dict) {
  Py_INCREF(dict);
}

// NOLINTNEXTLINE(bugprone-exception-escape)
PyFunctionPostHook::~PyFunctionPostHook() {
  // If python is already dead, leak the wrapped python objects
  if (Py_IsInitialized()) {
    pybind11::gil_scoped_acquire gil;
    Py_DECREF(dict);
  }
}

auto PyFunctionPostHook::operator()(
    const variable_list& _outputs, /* grad_inputs */
    const variable_list& _inputs /* grad_outputs */) -> variable_list {
  pybind11::gil_scoped_acquire gil;
  THPObjectPtr grad_inputs(wrap_variables(_outputs));
  THPObjectPtr grad_outputs(wrap_variables(_inputs));
  THPObjectPtr tup(PyTuple_New(2));
  PyTuple_SET_ITEM(tup.get(), 0, grad_inputs.release());
  PyTuple_SET_ITEM(tup.get(), 1, grad_outputs.release());
  _call_hooks(dict, tup.get());
  return unwrap_variables(PyTuple_GetItem(tup.get(), 0));
}

void PyFunctionTensorPreHook::compiled_args(CompiledNodeArgs& args) {
  PyObject *key = nullptr, *value = nullptr;
  Py_ssize_t pos = 0;
  while (PyDict_Next(dict, &pos, &key, &value)) {
    Py_INCREF(value);
    args.add_tensor_pre_hook(
        c10::SafePyObject(value, getPyInterpreter()),
        static_cast<int>(value_idx));
  }
}

void PyFunctionPreHook::compiled_args(CompiledNodeArgs& args) {
  PyObject *key = nullptr, *value = nullptr;
  Py_ssize_t pos = 0;
  while (PyDict_Next(dict, &pos, &key, &value)) {
    Py_INCREF(value);
    args.add_pre_hook(c10::SafePyObject(value, getPyInterpreter()));
  }
}

void PyFunctionPostHook::compiled_args(CompiledNodeArgs& args) {
  PyObject *key = nullptr, *value = nullptr;
  Py_ssize_t pos = 0;
  while (PyDict_Next(dict, &pos, &key, &value)) {
    Py_INCREF(value);
    args.add_post_hook(c10::SafePyObject(value, getPyInterpreter()));
  }
}

PyFunctionTensorPostAccGradHooks::PyFunctionTensorPostAccGradHooks(
    PyObject* dict)
    : dict(dict) {
  Py_INCREF(dict);
}

// NOLINTNEXTLINE(bugprone-exception-escape)
PyFunctionTensorPostAccGradHooks::~PyFunctionTensorPostAccGradHooks() {
  // If python is already dead, leak the wrapped python objects
  if (Py_IsInitialized()) {
    pybind11::gil_scoped_acquire gil;
    Py_DECREF(dict);
  }
}

auto PyFunctionTensorPostAccGradHooks::operator()(const Variable& tensor)
    -> void {
  pybind11::gil_scoped_acquire gil;
  THPObjectPtr tup(PyTuple_New(1));
  PyTuple_SET_ITEM(tup.get(), 0, THPVariable_Wrap(tensor));
  bool returned_none = !_call_hooks(dict, tup.get());
  TORCH_CHECK(
      returned_none, "Tensor post accumulate grad hooks should return None.");
}

void PyFunctionTensorPostAccGradHooks::compiled_args(
    torch::dynamo::autograd::CompiledNodeArgs& args) {
  PyObject *key = nullptr, *value = nullptr;
  Py_ssize_t pos = 0;
  while (PyDict_Next(dict, &pos, &key, &value)) {
    Py_INCREF(value);
    c10::SafePyObject hook_obj(value, getPyInterpreter());
    args.add_post_acc_grad_hook(std::move(hook_obj));
  }
}

void PyFunctionTensorPostAccGradHooks::apply_with_saved(
    Variable& tensor,
    torch::dynamo::autograd::SwapSavedVariables& saved) {
  for (const auto hook : saved.get_curr_node_call().post_acc_grad_hooks) {
    THPObjectPtr py_var(THPVariable_Wrap(tensor));
    PyObject_CallMethod(
        saved.get_py_compiler(),
        "post_acc_grad_hook",
        "Oi",
        py_var.get(),
        hook);
  }
}

} // namespace torch::autograd

static PyObject* wrap_variables(const variable_list& c_variables) {
  size_t num_vars = c_variables.size();
  THPObjectPtr tuple(PyTuple_New(static_cast<Py_ssize_t>(num_vars)));
  if (!tuple)
    throw python_error();
  for (const auto i : c10::irange(num_vars)) {
    THPObjectPtr var(THPVariable_Wrap(c_variables[i]));
    if (!var)
      throw python_error();
    PyTuple_SET_ITEM(tuple.get(), i, var.release());
  }
  return tuple.release();
}

static variable_list unwrap_variables(PyObject* py_variables) {
  variable_list results(PyTuple_GET_SIZE(py_variables));
  for (const auto i : c10::irange(results.size())) {
    PyObject* item = PyTuple_GET_ITEM(py_variables, i);
    if (item == Py_None) {
      continue;
    } else if (THPVariable_Check(item)) {
      results[i] = THPVariable_Unpack(item);
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
    PyErr_Format(
        PyExc_TypeError,
        "expected tuple, but hook returned '%s'",
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

  for (const auto i : c10::irange(prev_size)) {
    check_single_result(
        PyTuple_GET_ITEM(prev, i), PyTuple_GET_ITEM(result, i), hook);
  }
}

static void check_single_result(
    PyObject* _original,
    PyObject* _result,
    PyObject* hook) {
  if (_result == Py_None)
    return;

  if (_original == Py_None) {
    throw std::runtime_error(
        "can't replace a None gradient with a non-None value");
  }

  if (!PyObject_IsInstance(_result, THPVariableClass)) {
    PyErr_Format(
        PyExc_TypeError,
        "expected Variable, but hook returned '%s'",
        THPUtils_typename(_result));
    throw python_error();
  }

  const auto& original = THPVariable_Unpack(_original);
  const auto& result = THPVariable_Unpack(_result);

  torch::autograd::check_variable_result(original, result, hook_name(hook));
}

static std::string hook_name(PyObject* hook) {
  if (PyObject_HasAttrString(hook, "__name__")) {
    THPObjectPtr name(PyObject_GetAttrString(hook, "__name__"));
    if (!name)
      throw python_error();

    if (name && THPUtils_checkString(name.get())) {
      return THPUtils_unpackString(name.get());
    }
  }
  return "<unknown>";
}
