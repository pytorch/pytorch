#pragma once

#include <pybind11/pybind11.h>
#include "torch/csrc/utils/structseq.h"

namespace six {

// Usually instances of PyStructSequence is also an instance of tuple
// but in some py2 environment it is not, so we have to manually check
// the name of the type to determine if it is a namedtupled returned
// by a pytorch operator.

inline bool isTuple(pybind11::handle input) {
  if (PyTuple_Check(input.ptr())) {
    return true;
  }
#if PY_MAJOR_VERSION == 2
  return pybind11::cast<std::string>(input.get_type().attr("__module__")) == "torch.return_types";
#else
  return false;
#endif
}

inline bool isTuple(PyObject* obj) {
  return isTuple(pybind11::handle(obj));
}

inline PyObject *toTuple(PyStructSequence *obj) {
  // create a new tuple object on python 2, or increase
  // the ref count of the current object on python 3.
#if PY_MAJOR_VERSION == 2
  return torch::utils::structseq_slice(obj, 0, Py_SIZE(obj));
#else
  Py_INCREF(obj);
  return (PyObject *)obj;
#endif
}

}  // namespace six
