#pragma once

#include <pybind11/pybind11.h>
#include <torch/csrc/utils/object_ptr.h>
#include <torch/csrc/utils/pybind.h>
#include <torch/csrc/utils/structseq.h>

namespace six {

// Usually instances of PyStructSequence is also an instance of tuple
// but in some py2 environment it is not, so we have to manually check
// the name of the type to determine if it is a namedtupled returned
// by a pytorch operator.

inline bool isStructSeq(pybind11::handle input) {
  return pybind11::cast<std::string>(pybind11::type::handle_of(input).attr(
             "__module__")) == "torch.return_types";
}

inline bool isStructSeq(PyObject* obj) {
  return isStructSeq(pybind11::handle(obj));
}

inline bool isTuple(pybind11::handle input) {
  if (PyTuple_Check(input.ptr())) {
    return true;
  }
  return false;
}

inline bool isTuple(PyObject* obj) {
  return isTuple(pybind11::handle(obj));
}

// maybeAsTuple: if the input is a structseq, then convert it to a tuple
//
// On Python 3, structseq is a subtype of tuple, so these APIs could be used
// directly. But on Python 2, structseq is not a subtype of tuple, so we need to
// manually create a new tuple object from structseq.
inline THPObjectPtr maybeAsTuple(PyStructSequence* obj) {
  Py_INCREF(obj);
  return THPObjectPtr((PyObject*)obj);
}

inline THPObjectPtr maybeAsTuple(PyObject* obj) {
  if (isStructSeq(obj))
    return maybeAsTuple((PyStructSequence*)obj);
  Py_INCREF(obj);
  return THPObjectPtr(obj);
}

} // namespace six
