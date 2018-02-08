// ${generated_comment}

// Python bindings for torch.* functions implemented through ATen.
//
// The functions are bound as static methods on a class
// torch._C._VariableFunctions which is also aliased as Variable._torch.

#include <Python.h>

#include "torch/csrc/Exceptions.h"
#include "torch/csrc/autograd/python_variable.h"
#include "torch/csrc/autograd/utils/wrap_outputs.h"
#include "torch/csrc/utils/python_arg_parser.h"
#include "torch/csrc/utils/tensor_new.h"
#include "torch/csrc/utils/tensor_numpy.h"

#include "python_torch_functions_dispatch.h"

using at::Tensor;
using at::Scalar;
using at::ScalarType;
using at::Backend;
using namespace torch::autograd::utils;

namespace torch { namespace autograd {

static Tensor set_requires_grad(Tensor self, bool requires_grad) {
  static_cast<Variable&>(self).get()->_requires_grad = requires_grad;
  return self;
}

static Tensor dispatch_clamp(const Tensor & self, Scalar min, Scalar max) {
  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.clamp(min, max);
}
static Tensor dispatch_clamp_min(const Tensor & self, Scalar min) {
  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.clamp_min(min);
}
static Tensor dispatch_clamp_max(const Tensor & self, Scalar max) {
  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.clamp_max(max);
}

// The Python clamp() syntax has to be mapped to one of three C++ functions
static PyObject * THPVariable_clamp(PyObject* module, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "clamp(Tensor input, Scalar min=None, Scalar max=None)",
  });
  PyObject* parsed_args[4];
  auto r = parser.parse(args, kwargs, parsed_args);
  if (!r.isNone(1) && !r.isNone(2)) {
    return THPVariable_Wrap(dispatch_clamp(r.tensor(0), r.scalar(1), r.scalar(2)));
  } else if (!r.isNone(1)) {
    return THPVariable_Wrap(dispatch_clamp_min(r.tensor(0), r.scalar(1)));
  } else if (!r.isNone(2)) {
    return THPVariable_Wrap(dispatch_clamp_max(r.tensor(0), r.scalar(2)));
  } else {
    throw std::runtime_error("At least one of 'min' or 'max' must not be None");
  }
  END_HANDLE_TH_ERRORS
}

static PyObject * THPVariable_from_numpy(PyObject* module, PyObject* arg)
{
  HANDLE_TH_ERRORS
  auto data = torch::utils::tensor_from_numpy(arg);
  return THPVariable_Wrap(make_variable(std::move(data)));
  END_HANDLE_TH_ERRORS
}

static PyObject * THPVariable_variable(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  return THPVariable_Wrap(torch::utils::new_tensor(default_type(), args, kwargs));
  END_HANDLE_TH_ERRORS
}

// generated methods start here

${py_methods}

static PyMethodDef torch_functions[] = {
  {"clamp", (PyCFunction)THPVariable_clamp, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"from_numpy", (PyCFunction)THPVariable_from_numpy, METH_STATIC | METH_O, NULL},
  {"variable", (PyCFunction)THPVariable_variable, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  ${py_method_defs}
  {NULL}
};

static PyTypeObject THPVariableFunctions = {
  PyVarObject_HEAD_INIT(NULL, 0)
  "torch._C._VariableFunctions",         /* tp_name */
  0,                                     /* tp_basicsize */
  0,                                     /* tp_itemsize */
  0,                                     /* tp_dealloc */
  0,                                     /* tp_print */
  0,                                     /* tp_getattr */
  0,                                     /* tp_setattr */
  0,                                     /* tp_reserved */
  0,                                     /* tp_repr */
  0,                                     /* tp_as_number */
  0,                                     /* tp_as_sequence */
  0,                                     /* tp_as_mapping */
  0,                                     /* tp_hash  */
  0,                                     /* tp_call */
  0,                                     /* tp_str */
  0,                                     /* tp_getattro */
  0,                                     /* tp_setattro */
  0,                                     /* tp_as_buffer */
  Py_TPFLAGS_DEFAULT,                    /* tp_flags */
  NULL,                                  /* tp_doc */
  0,                                     /* tp_traverse */
  0,                                     /* tp_clear */
  0,                                     /* tp_richcompare */
  0,                                     /* tp_weaklistoffset */
  0,                                     /* tp_iter */
  0,                                     /* tp_iternext */
  torch_functions,                       /* tp_methods */
  0,                                     /* tp_members */
  0,                                     /* tp_getset */
  0,                                     /* tp_base */
  0,                                     /* tp_dict */
  0,                                     /* tp_descr_get */
  0,                                     /* tp_descr_set */
  0,                                     /* tp_dictoffset */
  0,                                     /* tp_init */
  0,                                     /* tp_alloc */
  0                                      /* tp_new */
};

void initTorchFunctions(PyObject* module) {
  if (PyType_Ready(&THPVariableFunctions) < 0) {
    throw python_error();
  }
  Py_INCREF(&THPVariableFunctions);
  if (PyModule_AddObject(module, "_VariableFunctions", (PyObject*)&THPVariableFunctions) < 0) {
    throw python_error();
  }
}

}} // namespace torch::autograd
