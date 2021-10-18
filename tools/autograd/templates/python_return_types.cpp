#include <Python.h>

#include <vector>
#include <map>

#include "torch/csrc/autograd/python_return_types.h"
#include "torch/csrc/utils/structseq.h"
#include "torch/csrc/Exceptions.h"

namespace {
${py_return_types}
}

namespace torch {
namespace autograd {

static std::map<const char*, PyTypeObject*> return_types_map = {
    ${py_return_types_map}
};

PyTypeObject* get_namedtuple(const char* name) {
  return return_types_map.at(name);
}

void initReturnTypes(PyObject* module) {
  static struct PyModuleDef def = {
      PyModuleDef_HEAD_INIT, "torch._C._return_types", nullptr, -1, {}};
  PyObject* return_types_module = PyModule_Create(&def);
  if (!return_types_module) {
    throw python_error();
  }

  for (const auto& return_type : return_types_map) {
    if (PyModule_AddObject(
            return_types_module,
            return_type.first,
            (PyObject*)return_type.second) != 0) {
      throw python_error();
    }
  }

  // steals a reference to return_types
  if (PyModule_AddObject(module, "_return_types", return_types_module) != 0) {
    throw python_error();
  }
}

} // namespace autograd
} // namespace torch
