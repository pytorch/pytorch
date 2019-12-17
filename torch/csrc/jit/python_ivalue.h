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
    return c10::make_intrusive<ConcretePyObjectHolder>(py_obj);
  }

  PyObject* getPyObject() override {
    return py_obj_;
  }

  ~ConcretePyObjectHolder() {
    // decrease the ref_count to avoid memory leak
    Py_DECREF(py_obj_);
  }
  // explicit construction to avoid errornous implicit conversion and
  // copy-initialization
  explicit ConcretePyObjectHolder(PyObject* py_obj) : py_obj_(py_obj) {
    // increase the ref_count to avoid dangling behavior
    Py_INCREF(py_obj_);
  }

 private:
  PyObject* py_obj_;
};

} // namespace ivalue
} // namespace c10