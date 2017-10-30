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
  ${py_method_defs}
  {NULL}
};

}} // namespace torch::autograd
