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
  Variable detached = make_variable(self_.data());
  detached.is_volatile() = self_.is_volatile();
  detached.version_counter() = self_.version_counter();
  return THPVariable_Wrap(std::move(detached));
  END_HANDLE_TH_ERRORS
}

static PyObject * THPVariable_detach_(PyObject* self, PyObject* args)
{
  HANDLE_TH_ERRORS
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  if (self_.is_view()) {
    throw std::runtime_error("Can't detach views in-place. Use detach() instead");
  }
  self_.get()->requires_grad = false;
  self_.output_nr() = 0;
  self_.get()->_grad_fn = nullptr;
  Py_INCREF(self);
  return self;
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
  {"detach", (PyCFunction)THPVariable_detach, METH_NOARGS, NULL},
  {"detach_", (PyCFunction)THPVariable_detach_, METH_NOARGS, NULL},
  ${py_method_defs}
  {NULL}
};

}} // namespace torch::autograd
