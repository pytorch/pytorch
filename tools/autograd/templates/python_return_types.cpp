#include <Python.h>

#include <vector>
#include <map>

#include "torch/csrc/autograd/python_return_types.h"
#include "torch/csrc/utils/structseq.h"
#include "torch/csrc/Exceptions.h"

namespace {
  PyTypeObject get_namedtuple_helper(const char* name, PyStructSequence_Field fields[], size_t len) {
      PyTypeObject NamedTuple;
      PyStructSequence_Desc desc = { name, nullptr, fields, len };
      PyStructSequence_InitType(&NamedTuple, &desc);
      NamedTuple.tp_repr = (reprfunc)torch::utils::returned_structseq_repr;  
      return NamedTuple;
  }

  ${py_return_types}

}

namespace torch {
namespace autograd {

static std::map<const char*, PyTypeObject*> return_types_map = {
    ${py_return_types_map}
};

PyTypeObject* get_namedtuple(const char* name) {
  // hold onto generated return type.
  return return_types_map.at(name);
}

void initReturnTypes(PyObject* module) {
  static struct PyModuleDef def = {
      PyModuleDef_HEAD_INIT, "torch._C._return_types", nullptr, -1, {}};
  PyObject* return_types = PyModule_Create(&def);
  if (!return_types) {
    throw python_error();
  }
  // steals a reference to return_types
  if (PyModule_AddObject(module, "_return_types", return_types) != 0) {
    throw python_error();
  }

  auto return_module = PyObject_GetAttrString(module, "_return_types");
  for (const auto& return_type: return_types_map) {
    PyModule_AddObject(return_module, return_type.first, (PyObject*)return_type.second);
  }
}

} // namespace autograd
} // namespace torch
