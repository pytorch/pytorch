#include <Python.h>

#include <vector>
#include <map>
#include <string>

#include "torch/csrc/autograd/python_return_types.h"
#include "torch/csrc/utils/structseq.h"
#include "torch/csrc/Exceptions.h"

namespace {
${py_return_types}
}

namespace torch {
namespace autograd {

static std::map<std::string, PyTypeObject*> return_types_map = {
    ${py_return_types_map}
};

PyTypeObject* get_namedtuple(std::string name) {
  return return_types_map[name];
}

void initReturnTypes(PyObject* module) {
  static struct PyModuleDef def = {
      PyModuleDef_HEAD_INIT, "torch._C._return_types", nullptr, -1, {}};
  PyObject* return_types_module = PyModule_Create(&def);
  if (!return_types_module) {
    throw python_error();
  }

  for (const auto& return_type_pair : return_types_map) {
    if (PyModule_AddObject(
            return_types_module,
            return_type_pair.first.c_str(),
            (PyObject*)return_type_pair.second) != 0) {
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
