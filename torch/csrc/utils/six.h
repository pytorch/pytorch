#pragma once

#include <pybind11/pybind11.h>
#include <torch/csrc/utils/structseq.h>

namespace six {

// Usually instances of PyStructSequence is also an instance of tuple
// but in some py2 environment it is not, so we have to manually check
// the name of the type to determine if it is a namedtupled returned
// by a pytorch operator.

inline bool isStructSeq(pybind11::handle input) {
  return pybind11::cast<std::string>(input.get_type().attr("__module__")) == "torch.return_types";
}

inline bool isStructSeq(PyObject* obj) {
  return isStructSeq(pybind11::handle(obj));
}

inline bool isTuple(pybind11::handle input) {
  if (PyTuple_Check(input.ptr())) {
    return true;
  }
#if PY_MAJOR_VERSION == 2
  return isStructSeq(input);
#else
  return false;
#endif
}

inline bool isTuple(PyObject* obj) {
  return isTuple(pybind11::handle(obj));
}

// toTuple: enable PyTuple API for PyStructSequence
//
// The input of this function is assumed to be either a tuple or a structseq. The caller
// is responsible for this check.
//
// On Python 3, structseq is a subtype of tuple, so these APIs could be used directly.
//
// But on Python 2, structseq is not a subtype of tuple, so we need to manually create a
// new tuple object from structseq. The new object has ref count 1, so when finish using
// the returned object, we need to manually decrease ref count of the returned object.
//
// Instead of making the caller writing codes like:
//
// tuple = six::toTuple(obj);
// use_tuple(tuple);
// #if PY_MAJOR_VERSION == 2
// Py_DECREF(tup);
// #endif
//
// We decide to increase the ref count of the input object on python 3, so that the caller
// could write cleaner code like below:
//
// tuple = six::toTuple(obj);
// use_tuple(tuple);
// Py_DECREF(tup);

inline PyObject *toTuple(PyStructSequence *obj) {
#if PY_MAJOR_VERSION == 2
  return torch::utils::structseq_slice(obj, 0, Py_SIZE(obj));
#else
  Py_INCREF(obj);
  return (PyObject *)obj;
#endif
}

inline PyObject *toTuple(PyObject *obj) {
  if (isStructSeq(obj))
    return toTuple((PyStructSequence *)obj);
  Py_INCREF(obj);
  return obj;
}

}  // namespace six
