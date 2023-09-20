#pragma once

#include <ATen/core/Tensor.h>
#include <torch/csrc/python_headers.h>
#include <torch/csrc/utils/pythoncapi_compat.h>
#include <memory>

#include <ATen/core/function_schema.h>
#include <pybind11/pybind11.h>
#include <torch/csrc/Exceptions.h>
#include <torch/csrc/Export.h>
#include <torch/csrc/autograd/variable.h>
#include <torch/csrc/utils/pybind.h>

namespace py = pybind11;

// Python object that backs torch.autograd.Variable
struct THPVariable {
  PyObject_HEAD;
  // Payload
  c10::MaybeOwned<at::Tensor> cdata;
  // Hooks to be run on backwards pass (corresponds to Python attr
  // '_backwards_hooks', set by 'register_hook')
  PyObject* backward_hooks = nullptr;
  // Hooks to be run in the backwards pass after accumulate grad,
  // i.e., after the .grad has been set (corresponds to Python attr
  // '_post_accumulate_grad_hooks', set by 'register_post_accumulate_grad_hook')
  PyObject* post_accumulate_grad_hooks = nullptr;
};

TORCH_PYTHON_API void registerPythonTensorClass(
    const std::string& device,
    PyObject* python_tensor_class);

TORCH_PYTHON_API void activateCUDATrace();

TORCH_PYTHON_API extern PyObject* THPVariableClass;
TORCH_PYTHON_API extern PyObject* ParameterClass;

bool THPVariable_initModule(PyObject* module);
TORCH_PYTHON_API PyObject* THPVariable_Wrap(at::TensorBase var);

static inline bool THPVariable_CheckTypeExact(PyTypeObject* tp) {
  // Check that a python object is a `Tensor`, but not a `Tensor` subclass.
  // (A subclass could have different semantics.) The one exception is
  // Parameter, which is used for Python bookkeeping but is equivalent to
  // Tensor as far as C++ is concerned.
  return (
      tp == (PyTypeObject*)THPVariableClass ||
      tp == (PyTypeObject*)ParameterClass);
}

static inline bool THPVariable_CheckExact(PyObject* obj) {
  return THPVariable_CheckTypeExact(Py_TYPE(obj));
}

inline bool THPVariable_Check(PyObject* obj) {
  if (!THPVariableClass)
    return false;

  // Fast path
  if (THPVariable_CheckExact(obj)) {
    return true;
  }

  const auto result = PyObject_IsInstance(obj, THPVariableClass);
  if (result == -1)
    throw python_error();
  return result;
}

inline const at::Tensor& THPVariable_Unpack(THPVariable* var) {
  return *var->cdata;
}

inline const at::Tensor& THPVariable_Unpack(PyObject* obj) {
  return THPVariable_Unpack(reinterpret_cast<THPVariable*>(obj));
}

std::pair<py::object, py::dict> parseIValuesToPyArgsKwargs(
    const c10::OperatorHandle& op,
    const std::vector<c10::IValue>& arguments);

void pushPyOutToStack(
    const c10::OperatorHandle& op,
    torch::jit::Stack* stack,
    py::object out,
    const char* msg);

inline PyObject* THPVariable_WrapList(
    const torch::autograd::variable_list& inputs) {
  PyObject* pyinput = PyList_New(inputs.size());
  for (const auto i : c10::irange(inputs.size())) {
    PyList_SET_ITEM(pyinput, i, THPVariable_Wrap(inputs[i]));
  }
  return pyinput;
}

inline torch::autograd::variable_list THPVariable_UnpackList(
    PyObject* pyresult) {
  TORCH_CHECK(PyList_CheckExact(pyresult));
  auto result_len = PyList_GET_SIZE(pyresult);
  torch::autograd::variable_list result;
  result.reserve(result_len);
  for (const auto i : c10::irange(result_len)) {
    PyObject* item = PyList_GET_ITEM(pyresult, i);
    if (!Py_IsNone(item)) {
      TORCH_INTERNAL_ASSERT_DEBUG_ONLY(THPVariable_Check(item));
      result.emplace_back(THPVariable_Unpack(item));
    } else {
      result.emplace_back();
    }
  }
  return result;
}
