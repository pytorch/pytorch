// ${generated_comment}

#include <Python.h>

#include "torch/csrc/Exceptions.h"
#include "torch/csrc/autograd/python_variable.h"
#include "torch/csrc/autograd/utils/wrap_outputs.h"
#include "torch/csrc/utils/python_arg_parser.h"
#include "torch/csrc/utils/python_numbers.h"

#include "python_variable_methods_dispatch.h"

using at::Tensor;
using at::Scalar;
using namespace torch::autograd::utils;

namespace torch { namespace autograd {

static PyObject * THPVariable_detach(PyObject* self, PyObject* args)
{
  HANDLE_TH_ERRORS
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  return THPVariable_Wrap(self_.detach());
  END_HANDLE_TH_ERRORS
}

static PyObject * THPVariable_detach_(PyObject* self, PyObject* args)
{
  HANDLE_TH_ERRORS
  reinterpret_cast<THPVariable*>(self)->cdata.detach_();
  Py_INCREF(self);
  return self;
  END_HANDLE_TH_ERRORS
}

static Tensor dispatch_contiguous(const Tensor & self) {
  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.contiguous();
}

static PyObject * THPVariable_contiguous(PyObject* self, PyObject* args)
{
  HANDLE_TH_ERRORS
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  // avoids touching the GIL or current device if self is already contiguous
  if (self_.is_contiguous()) {
    Py_INCREF(self);
    return self;
  }
  return THPVariable_Wrap(dispatch_contiguous(self_));
  END_HANDLE_TH_ERRORS
}

// generated methods start here

${py_methods}

PyMethodDef variable_methods[] = {
  {"__add__", (PyCFunction)THPVariable_add, METH_VARARGS | METH_KEYWORDS, NULL},
  {"__radd__", (PyCFunction)THPVariable_add, METH_VARARGS | METH_KEYWORDS, NULL},
  {"__iadd__", (PyCFunction)THPVariable_add_, METH_VARARGS | METH_KEYWORDS, NULL},
  {"__rmul__", (PyCFunction)THPVariable_mul, METH_VARARGS | METH_KEYWORDS, NULL},
  {"__mul__", (PyCFunction)THPVariable_mul, METH_VARARGS | METH_KEYWORDS, NULL},
  {"__imul__", (PyCFunction)THPVariable_mul_, METH_VARARGS | METH_KEYWORDS, NULL},
  {"__sub__", (PyCFunction)THPVariable_sub, METH_VARARGS | METH_KEYWORDS, NULL},
  {"__isub__", (PyCFunction)THPVariable_sub_, METH_VARARGS | METH_KEYWORDS, NULL},
  {"__div__", (PyCFunction)THPVariable_div, METH_VARARGS | METH_KEYWORDS, NULL},
  {"__truediv__", (PyCFunction)THPVariable_div, METH_VARARGS | METH_KEYWORDS, NULL},
  {"__idiv__", (PyCFunction)THPVariable_div_, METH_VARARGS | METH_KEYWORDS, NULL},
  {"__mod__", (PyCFunction)THPVariable_remainder, METH_VARARGS | METH_KEYWORDS, NULL},
  {"contiguous", (PyCFunction)THPVariable_contiguous, METH_NOARGS, NULL},
  {"detach", (PyCFunction)THPVariable_detach, METH_NOARGS, NULL},
  {"detach_", (PyCFunction)THPVariable_detach_, METH_NOARGS, NULL},
  ${py_method_defs}
  {NULL}
};

}} // namespace torch::autograd
