#include "python_nn_functions.h"

// ${generated_comment}

#include "torch/csrc/Device.h"
#include "torch/csrc/DynamicTypes.h"
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
  auto parsed = parse_to_conversion(args, kwargs);
  auto& device = std::get<0>(parsed);
  auto& scalarType = std::get<1>(parsed);
  auto non_blocking = std::get<2>(parsed);
  PyObject *tuple = PyTuple_New(3);
  PyTuple_SET_ITEM(tuple, 0, device       ? THPDevice_New(*device) : Py_None);
  PyTuple_SET_ITEM(tuple, 1, scalarType   ? torch::autograd::utils::wrap(torch::getDtype(*scalarType)) : Py_None);
  PyTuple_SET_ITEM(tuple, 2, non_blocking ? Py_True : Py_False);
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
