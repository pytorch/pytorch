#include <Python.h>

#include <vector>
#include <map>
#include <string>

#include "torch/csrc/autograd/generated/python_return_types.h"
#include "torch/csrc/utils/structseq.h"
#include "torch/csrc/Exceptions.h"
#include "torch/csrc/utils/object_ptr.h"

namespace torch::autograd::generated {

${py_return_types}

} // namespace torch::autograd::generated

namespace torch::autograd {

static void addReturnType(
    PyObject* module,
    const char* name,
    PyTypeObject* type) {
  // AddObjectRef takes its own reference, so the module keeps the TypeObject
  // alive for the unlikely case of a user deleting or overriding it.
  TORCH_CHECK_PYTHON(
      PyModule_AddObjectRef(module, name, (PyObject*)type) == 0);
}

void initReturnTypes(PyObject* module) {
  static struct PyModuleDef def = {
      PyModuleDef_HEAD_INIT, "torch._C._return_types", nullptr, -1, {}};
  // AddObjectRef takes its own reference rather than stealing ours, so the
  // module we create is owned here and released on the way out.
  THPObjectPtr return_types_module(PyModule_Create(&def));
  TORCH_CHECK_PYTHON(return_types_module);

  ${py_return_types_registrations}

  TORCH_CHECK_PYTHON(
      PyModule_AddObjectRef(module, "_return_types", return_types_module) == 0);
}

} // namespace torch::autograd
