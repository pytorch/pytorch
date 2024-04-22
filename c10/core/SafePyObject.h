#pragma once

#include <c10/core/impl/PyInterpreter.h>
#include <c10/macros/Export.h>
#include <c10/util/python_stub.h>
#include <utility>

namespace c10 {

// This is an safe owning holder for a PyObject, akin to pybind11's
// py::object, with two major differences:
//
//  - It is in c10/core; i.e., you can use this type in contexts where
//    you do not have a libpython dependency
//
//  - It is multi-interpreter safe (ala torchdeploy); when you fetch
//    the underlying PyObject* you are required to specify what the current
//    interpreter context is and we will check that you match it.
//
// It is INVALID to store a reference to a Tensor object in this way;
// you should just use TensorImpl directly in that case!
struct C10_API SafePyObject {
  // Steals a reference to data
  SafePyObject(PyObject* data, c10::impl::PyInterpreter* pyinterpreter)
      : data_(data), pyinterpreter_(pyinterpreter) {}
  SafePyObject(SafePyObject&& other) noexcept
      : data_(std::exchange(other.data_, nullptr)),
        pyinterpreter_(other.pyinterpreter_) {}

  // In principle this could be copyable if we add an incref to PyInterpreter
  // but for now it's easier to just disallow it.
  SafePyObject(SafePyObject const&) = delete;
  SafePyObject& operator=(SafePyObject const&) = delete;

  ~SafePyObject() {
    if (data_ != nullptr) {
      (*pyinterpreter_)->decref(data_, /*has_pyobj_slot*/ false);
    }
  }

  c10::impl::PyInterpreter& pyinterpreter() const {
    return *pyinterpreter_;
  }
  PyObject* ptr(const c10::impl::PyInterpreter*) const;

  // stop tracking the current object, and return it
  PyObject* release() {
    auto rv = data_;
    data_ = nullptr;
    return rv;
  }

 private:
  PyObject* data_;
  c10::impl::PyInterpreter* pyinterpreter_;
};

// Like SafePyObject, but non-owning.  Good for references to global PyObjects
// that will be leaked on interpreter exit.  You get a copy constructor/assign
// this way.
struct C10_API SafePyHandle {
  SafePyHandle() : data_(nullptr), pyinterpreter_(nullptr) {}
  SafePyHandle(PyObject* data, c10::impl::PyInterpreter* pyinterpreter)
      : data_(data), pyinterpreter_(pyinterpreter) {}

  c10::impl::PyInterpreter& pyinterpreter() const {
    return *pyinterpreter_;
  }
  PyObject* ptr(const c10::impl::PyInterpreter*) const;
  void reset() {
    data_ = nullptr;
    pyinterpreter_ = nullptr;
  }
  operator bool() {
    return data_;
  }

 private:
  PyObject* data_;
  c10::impl::PyInterpreter* pyinterpreter_;
};

} // namespace c10
