#include <Python.h>

#include <vector>
#include <map>
#include <string>

#include "torch/csrc/autograd/generated/python_return_types.h"
#include "torch/csrc/utils/structseq.h"
#include "torch/csrc/Exceptions.h"

namespace torch::autograd::generated {

${py_return_types}

} // namespace torch::autograd::generated

namespace torch::autograd {

static void addReturnType(
    PyObject* module,
    const char* name,
    PyTypeObject* type) {
  // hold onto the TypeObject for the unlikely case of user
  // deleting or overriding it.
  Py_INCREF(type);
  if (PyModule_AddObject(
          module,
          name,
          (PyObject*)type) != 0) {
    Py_DECREF(type);
    // @allow-raw-throw: the reference must be dropped before unwinding
    throw python_error();
  }
}

void initReturnTypes(PyObject* module) {
  static struct PyModuleDef def = {
      PyModuleDef_HEAD_INIT, "torch._C._return_types", nullptr, -1, {}};
  PyObject* return_types_module = PyModule_Create(&def);
  TORCH_CHECK_PYTHON(return_types_module);

  ${py_return_types_registrations}

  // steals a reference to return_types on success
  if (PyModule_AddObject(module, "_return_types", return_types_module) != 0) {
    Py_DECREF(return_types_module);
    // @allow-raw-throw: the reference must be dropped before unwinding
    throw python_error();
  }
}

} // namespace torch::autograd
