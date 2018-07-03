#pragma once

#include <Python.h>
#include <TH/THStorage.hpp>
#include <torch/csrc/utils/object_ptr.h>

namespace torch {

struct PyObjectFinalizer : public THFinalizer {
  THPObjectPtr pyobj_;
  // TODO: This recursive structure can lead to a stack overflow if you
  // put too many finalizers on the same object
  std::unique_ptr<THFinalizer> next_;
  PyObjectFinalizer(PyObject* pyobj) {
    Py_XINCREF(pyobj);
    pyobj_ = pyobj;
  }
  void operator()() override {
    if (next_) { (*next_)(); }
  }
};

} // namespace torch
