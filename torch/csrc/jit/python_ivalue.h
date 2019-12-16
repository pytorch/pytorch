#pragma once
#include <pybind11/pybind11.h>
#include <torch/csrc/python_headers.h>

namespace py = pybind11;

namespace c10 {
namespace ivalue {

// concrete ivalue Holder that hold a PyObject*
struct C10_EXPORT ConcretePyObjectHolder final : PyObjectHolder {
 public:
  static c10::intrusive_ptr<PyObjectHolder> steal(PyObject* py_obj) {
    Py_INCREF(py_obj);
    return c10::make_intrusive<ConcretePyObjectHolder>(py_obj);
  }

  PyObject* getPyObject() override {
    return py_obj_;
  }

  ~ConcretePyObjectHolder() {
    Py_DECREF(getPyObject());
  }
  // explicit construction to avoid errornous implicit conversion and
  // copy-initialization
  explicit ConcretePyObjectHolder(PyObject* py_obj) : py_obj_(py_obj) {}

 private:
  PyObject* py_obj_;
};

} // namespace ivalue
} // namespace c10