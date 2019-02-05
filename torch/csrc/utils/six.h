#pragma once

#include <pybind11/pybind11.h>

namespace six {

// Usually instances of PyStructSequence is also an instance of tuple
// but in some py2 environment it is not, so we have to manually check
// the name of the type to determine if it is a namedtupled returned
// by a pytorch operator.

inline bool isTuple(pybind11::handle input) {
  std::string m = pybind11::str(input.get_type().attr("__module__"));
  return pybind11::isinstance<pybind11::tuple>(input) || m == "torch.return_types";
}

inline bool isTuple(PyObject* obj) {
  return isTuple(pybind11::handle(obj));
}

}  // namespace six
