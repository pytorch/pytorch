#include "python_nn_functions.h"

// ${generated_comment}

#include "torch/csrc/Exceptions.h"
#include "torch/csrc/autograd/python_variable.h"
#include "torch/csrc/autograd/utils/wrap_outputs.h"
#include "torch/csrc/utils/python_arg_parser.h"

#include "python_nn_functions_dispatch.h"

using at::Tensor;
using at::Scalar;
using namespace torch::autograd::utils;

namespace torch { namespace autograd {

static PyObject * THPVariable__parse_to(PyObject* module, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "to(Device device, ScalarType dtype=None, bool non_blocking=False)",
    "to(ScalarType dtype, bool non_blocking=False)",
    "to(Tensor tensor, bool non_blocking=False)",
  });
  PyObject *tuple = PyTuple_New(3);
  if (!tuple) throw python_error();
  ParsedArgs<3> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    PyTuple_SET_ITEM(tuple, 0, r.pyobject(0));
    PyTuple_SET_ITEM(tuple, 1, r.pyobject(1));
    PyTuple_SET_ITEM(tuple, 2, r.toBool(2) ? Py_True : Py_False);
  } else if (r.idx == 1) {
    PyTuple_SET_ITEM(tuple, 0, Py_None);
    PyTuple_SET_ITEM(tuple, 1, r.pyobject(0));
    PyTuple_SET_ITEM(tuple, 2, r.toBool(1) ? Py_True : Py_False);
  } else if (r.idx == 2) {
    auto tensor = reinterpret_cast<THPVariable*>(r.pyobject(0));
    PyTuple_SET_ITEM(tuple, 0, THPVariable_device(tensor));
    PyTuple_SET_ITEM(tuple, 1, THPVariable_dtype(tensor));
    PyTuple_SET_ITEM(tuple, 2, r.toBool(1) ? Py_True : Py_False);
  }
  return tuple;
  END_HANDLE_TH_ERRORS
}

${py_methods}

static PyMethodDef nn_functions[] = {
  {"_parse_to", (PyCFunction)THPVariable__parse_to, METH_VARARGS | METH_KEYWORDS, NULL},
  ${py_method_defs}
  {NULL}
};

void initNNFunctions(PyObject* module) {
#if PY_MAJOR_VERSION == 2
  PyObject* nn = Py_InitModule("torch._C._nn", nn_functions);
  Py_XINCREF(nn);  // Py_InitModule returns "borrowed" reference
#else
  static struct PyModuleDef def = {
     PyModuleDef_HEAD_INIT,
     "torch._C._nn",
     NULL,
     -1,
     nn_functions
  };
  PyObject* nn = PyModule_Create(&def);
#endif
  if (!nn) {
    throw python_error();
  }
  // steals a reference to nn
  if (PyModule_AddObject(module, "_nn", nn) != 0) {
    throw python_error();
  }
}

}} // namespace torch::autograd
