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

  // Note [Destructing py::object]
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~
  //
  // (1) Why py_obj_ = py::none(); does not work. Because we also need to
  // acquire GIL when destructing py::object of None that de-references None.
  // https://docs.python.org/3/c-api/none.html#c.Py_RETURN_NONE
  //
  // https://stackoverflow.com/questions/15287590/why-should-py-increfpy-none-be-required-before-returning-py-none-in-c
  //
  // (2) Why we need to call dec_ref() explicitly. Because py::object of
  // nullptr, on destruction, effectively does nothing because of it calls
  // Py_XDECREF(NULL) underlying.
  // https://docs.python.org/3/c-api/refcounting.html#c.Py_XDECREF
  ~ConcretePyObjectHolder() {
    pybind11::gil_scoped_acquire ag;
    py_obj_.dec_ref();
    // explicitly setting PyObject* to nullptr to prevent py::object's dtor to
    // decref on the PyObject again.
    py_obj_.ptr() = nullptr;
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
