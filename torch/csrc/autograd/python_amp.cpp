#include <torch/csrc/python_headers.h>
#include <torch/csrc/autograd/python_amp.h>
#include <torch/csrc/autograd/amp_mode.h>
#include <torch/csrc/Exceptions.h>

#include <ATen/ATen.h>
// #include <torch/csrc/autograd/utils/wrap_outputs.h>

namespace torch {
namespace autograd {
namespace amp {

// Following the patterns in
// torch/csrc/autograd/autograd.h
// init.cpp
// grad_mode.h
// grad_mode.cpp
// I don't think I need any INCREFS/DECREFS/XDECREFS in the code below.

// This is exposed directly from Aten.
// static PyObject* THPAmp_getAmpOverflowState(PyObject* self, PyObject* arg) {
//   HANDLE_TH_ERRORS
//   return torch::autograd::utils::wrap(at::native::_get_amp_overflow_state());
//   END_HANDLE_TH_ERRORS
// }
// Come to think of it, why not just expose all stuff here through Aten instead of manual bindings?
// Don't forget to argue about this in the PR.

static PyObject* THPAmp_getGradScalingEnabled(PyObject* self, PyObject* arg) {
  HANDLE_TH_ERRORS
  // Py_RETURN* may be multiline macros (involving increfs) so we need braces.
  if(AmpMode::is_grad_scaling_enabled()) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  //
  // https://docs.python.org/3/extending/extending.html#intermezzo-errors-and-exceptions
  END_HANDLE_TH_ERRORS
}

static PyObject* THPAmp_setGradScalingEnabled(PyObject* self, PyObject* arg) {
  HANDLE_TH_ERRORS
  // int new_enabled;
  // if(!PyArg_ParseTuple(args, "p", &new_enabled);
  //   return nullptr;
  // setGradScalingEnabled(new_enabled);
  if(!PyBool_Check(arg)) {
    throw torch::TypeError("enabled must be a bool (got %s)", Py_TYPE(arg)->tp_name);
  }
  AmpMode::set_grad_scaling_enabled(arg == Py_True);
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

static PyObject* THPAmp_getGradScale(PyObject* self, PyObject* arg) {
  HANDLE_TH_ERRORS
  return PyFloat_FromDouble(AmpMode::get_grad_scale());
  END_HANDLE_TH_ERRORS
}

PyObject* THPAmp_setGradScale(PyObject* self, PyObject* arg) {
  HANDLE_TH_ERRORS
  // float new_scale;
  // if(!PyArg_ParseTuple(args, "i", &new_scale))
  //   return nullptr;
  if(!PyFloat_Check(arg)) {
    throw torch::TypeError("enabled must be a float (got %s)", Py_TYPE(arg)->tp_name);
  }
  AmpMode::set_grad_scale(PyFloat_AsDouble(arg));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// c10d methods on torch._C
static PyMethodDef methods[] = {
    // {"get_amp_overflow_state", (PyCFunction)THPAmp_getAmpOverflowState, METH_NOARGS, nullptr},
    {"is_grad_scaling_enabled", (PyCFunction)THPAmp_getGradScalingEnabled, METH_NOARGS, nullptr},
    {"set_grad_scaling_enabled", (PyCFunction)THPAmp_setGradScalingEnabled, METH_O, nullptr},
    {"get_grad_scale", (PyCFunction)THPAmp_getGradScale, METH_NOARGS, nullptr},
    {"set_grad_scale", (PyCFunction)THPAmp_setGradScale, METH_O, nullptr},
    {nullptr, nullptr, 0, nullptr}};

PyMethodDef* python_functions() {
  return methods;
}

} // namespace torch
} // namespace autograd
} // namespace amp
