#pragma once

#include <ATen/core/Tensor.h>
#include <torch/csrc/python_headers.h>
#include <torch/csrc/utils/pythoncapi_compat.h>

#include <ATen/core/function_schema.h>
#include <pybind11/pybind11.h>
#include <torch/csrc/Exceptions.h>
#include <torch/csrc/Export.h>
#include <torch/csrc/autograd/variable.h>
#include <torch/csrc/utils/pybind.h>

namespace py = pybind11;

// Python object that backs torch.autograd.Variable
struct THPVariable {
  PyObject_HEAD
  // Payload
  at::Tensor cdata;
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

TORCH_PYTHON_API void activateGPUTrace();

TORCH_PYTHON_API extern PyObject* THPVariableClass;
TORCH_PYTHON_API extern PyObject* ParameterClass;

bool THPVariable_initModule(PyObject* module);
TORCH_PYTHON_API PyObject* THPVariable_Wrap(at::TensorBase&& var);
TORCH_PYTHON_API PyObject* THPVariable_Wrap(const at::TensorBase& var);
TORCH_PYTHON_API PyObject* THPVariable_Wrap(
    const at::TensorBase& var,
    PyTypeObject* type);

inline bool THPVariable_CheckTypeExact(PyTypeObject* tp) {
  // Check that a python object is a `Tensor`, but not a `Tensor` subclass.
  // (A subclass could have different semantics.) The one exception is
  // Parameter, which is used for Python bookkeeping but is equivalent to
  // Tensor as far as C++ is concerned.
  return (
      tp == (PyTypeObject*)THPVariableClass ||
      tp == (PyTypeObject*)ParameterClass);
}

inline bool THPVariable_CheckExact(PyObject* obj) {
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
  TORCH_CHECK_PYTHON(result != -1);
  return result;
}

inline const at::Tensor& THPVariable_Unpack(THPVariable* var) {
  return var->cdata;
}

inline const at::Tensor& THPVariable_Unpack(PyObject* obj) {
  return THPVariable_Unpack(reinterpret_cast<THPVariable*>(obj));
}

std::pair<py::object, py::dict> parseIValuesToPyArgsKwargs(
    const c10::OperatorHandle& op,
    const std::vector<c10::IValue>& arguments);

// How the C++ Dispatcher passes an operator's arguments to a Python kernel
// (parseIValuesToPyArgsKwargs): arguments [0, num_positional) positionally,
// i.e. up to the last positional argument that isn't equal to its default,
// and, as kwargs, the kwarg-only arguments (from kwarg_only_start on) that
// aren't equal to their defaults.
struct PythonKernelArgSplit {
  size_t num_positional;
  size_t kwarg_only_start;
};

// `is_default(i)` says whether argument i is equal to its default.
template <typename IsDefault>
PythonKernelArgSplit splitPythonKernelArgs(
    const c10::FunctionSchema& schema,
    const IsDefault& is_default) {
  const auto& arguments = schema.arguments();
  size_t kwarg_only_start = arguments.size();
  while (kwarg_only_start > 0 && arguments[kwarg_only_start - 1].kwarg_only()) {
    --kwarg_only_start;
  }
  size_t num_positional = kwarg_only_start;
  while (num_positional > 0 && is_default(num_positional - 1)) {
    --num_positional;
  }
  return {num_positional, kwarg_only_start};
}

void pushPyOutToStack(
    const c10::OperatorHandle& op,
    torch::jit::Stack* stack,
    py::object out,
    const char* msg);

py::handle get_dtensor_class();

py::object dispatchDTensorOp(
    const c10::OperatorHandle& op,
    py::handle py_op,
    py::handle args,
    py::handle kwargs,
    torch::jit::Stack* stack);

inline PyObject* THPVariable_WrapList(
    const torch::autograd::variable_list& inputs) {
  PyObject* pyinput = PyList_New(static_cast<Py_ssize_t>(inputs.size()));
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
