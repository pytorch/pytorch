#pragma once
#include <pybind11/pybind11.h>
#include <torch/csrc/python_headers.h>

namespace py = pybind11;

namespace c10 {
namespace ivalue {

// concrete ivalue Holder that hold a py::object
struct C10_EXPORT ConcretePyObjectHolder final : PyObjectHolder {
 public:
  static c10::intrusive_ptr<PyObjectHolder> create(py::object py_obj) {
    return c10::make_intrusive<ConcretePyObjectHolder>(std::move(py_obj));
  }

  static c10::intrusive_ptr<PyObjectHolder> create(const py::handle& handle) {
    py::gil_scoped_acquire ag;
    return c10::make_intrusive<ConcretePyObjectHolder>(
        handle.cast<py::object>());
  }

  PyObject* getPyObject() override {
    return py_obj_.ptr();
  }

  ~ConcretePyObjectHolder() {
    pybind11::gil_scoped_acquire ag;
    py_obj_ = py::none();
  }
  // explicit construction to avoid errornous implicit conversion and
  // copy-initialization
  explicit ConcretePyObjectHolder(py::object py_obj)
      : py_obj_(std::move(py_obj)) {}

 private:
  py::object py_obj_;
};

} // namespace ivalue
} // namespace c10
