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

std::map<std::string, PyTypeObject*>& get_namedtuple_types_map() {
  static std::map<std::string, PyTypeObject*> namedtuple_types_map = {
    ${py_return_types_map}
  };
  return namedtuple_types_map;
}

PyTypeObject* get_namedtuple(std::string name) {
  static auto& namedtuple_types_map = get_namedtuple_types_map();
  return namedtuple_types_map[name];
}

void initReturnTypes(PyObject* module) {
  static struct PyModuleDef def = {
      PyModuleDef_HEAD_INIT, "torch._C._return_types", nullptr, -1, {}};
  PyObject* return_types_module = PyModule_Create(&def);
  if (!return_types_module) {
    throw python_error();
  }

  for (const auto& return_type_pair : get_namedtuple_types_map()) {
    // hold onto the TypeObject for the unlikely case of user
    // deleting or overriding it.
    Py_INCREF(return_type_pair.second);
    if (PyModule_AddObject(
            return_types_module,
            return_type_pair.first.c_str(),
            (PyObject*)return_type_pair.second) != 0) {
      Py_DECREF((PyObject*)return_type_pair.second);
      throw python_error();
    }
  }

  // steals a reference to return_types on success
  if (PyModule_AddObject(module, "_return_types", return_types_module) != 0) {
    Py_DECREF(return_types_module);
    throw python_error();
  }
}

} // namespace autograd
} // namespace torch
