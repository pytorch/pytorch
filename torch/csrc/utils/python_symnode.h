#pragma once

#include <c10/core/SafePyObject.h>
#include <c10/core/SymNodeImpl.h>

#include <torch/csrc/autograd/python_variable.h>
#include <torch/csrc/utils/pybind.h>

namespace torch {

TORCH_PYTHON_API py::handle get_symint_class();
TORCH_PYTHON_API py::handle get_symfloat_class();

// NB: These functions must not be called too early, otherwise torch not setup.
// Alternate design is to have torch "register" the object to us
inline bool is_symint(py::handle obj) {
  return py::isinstance(obj, get_symint_class());
}
inline bool is_symfloat(py::handle obj) {
  return py::isinstance(obj, get_symfloat_class());
}

namespace impl {

// This c10::SymNodeImpl simply backends to a Python object that
// implements the API.   The Python object is the source of truth,
// this is just an adapter so C++ calls can get to the object.
class PythonSymNodeImpl : public c10::SymNodeImpl {
 public:
  PythonSymNodeImpl(py::object pyobj) : c10::SymNodeImpl() {
    pyobj_ = std::make_shared<c10::SafePyObject>(
        pyobj.release().ptr(), getPyInterpreter());
  };

  c10::SymNode wrap_int(int64_t num) override {
    py::gil_scoped_acquire acquire;
    auto r = getPyObj().attr("wrap_int")(num);
    return c10::make_intrusive<PythonSymNodeImpl>(r);
  }

  c10::SymNode wrap_float(double num) override {
    py::gil_scoped_acquire acquire;
    auto r = getPyObj().attr("wrap_float")(num);
    return c10::make_intrusive<PythonSymNodeImpl>(r);
  }

  bool bool_() override {
    py::gil_scoped_acquire acquire;
    return getPyObj().attr("bool_")().is(py::handle(Py_True));
  }

  bool is_int() override {
    py::gil_scoped_acquire acquire;
    return getPyObj().attr("is_int")().is(py::handle(Py_True));
  }

  bool is_float() override {
    py::gil_scoped_acquire acquire;
    return getPyObj().attr("is_float")().is(py::handle(Py_True));
  }

  int64_t guard_int(const char* file, int64_t line) override {
    py::gil_scoped_acquire acquire;
    return getPyObj().attr("guard_int")(file, line).cast<int64_t>();
  }

  double guard_float(const char* file, int64_t line) override {
    py::gil_scoped_acquire acquire;
    return getPyObj().attr("guard_float")(file, line).cast<double>();
  }

  int64_t int_() override {
    py::gil_scoped_acquire acquire;
    return getPyObj().attr("int_")().cast<int64_t>();
  }

  std::string str() override {
    py::gil_scoped_acquire acquire;
    return getPyObj().attr("str")().cast<std::string>();
  }

  c10::SymNode dispatch_common_(const char* fname, const c10::SymNode& other) {
    auto pother = dynamic_cast<PythonSymNodeImpl*>(other.get());
    TORCH_CHECK(pother);
    py::gil_scoped_acquire acquire;
    auto r = getPyObj().attr(fname)(pother->getPyObj());
    return c10::make_intrusive<PythonSymNodeImpl>(r);
  }

  c10::SymNode dispatch_common_(const char* fname) {
    py::gil_scoped_acquire acquire;
    auto r = getPyObj().attr(fname)();
    return c10::make_intrusive<PythonSymNodeImpl>(r);
  }

  c10::SymNode add(const c10::SymNode& other) override {
    return dispatch_common_(__FUNCTION__, other);
  }

  c10::SymNode sub(const c10::SymNode& other) override {
    return dispatch_common_(__FUNCTION__, other);
  }

  c10::SymNode mul(const c10::SymNode& other) override {
    return dispatch_common_(__FUNCTION__, other);
  }

  c10::SymNode truediv(const c10::SymNode& other) override {
    return dispatch_common_(__FUNCTION__, other);
  }

  c10::SymNode pow(const c10::SymNode& other) override {
    return dispatch_common_(__FUNCTION__, other);
  }

  c10::SymNode floordiv(const c10::SymNode& other) override {
    return dispatch_common_(__FUNCTION__, other);
  }

  c10::SymNode mod(const c10::SymNode& other) override {
    return dispatch_common_(__FUNCTION__, other);
  }

  c10::SymNode eq(const c10::SymNode& other) override {
    return dispatch_common_(__FUNCTION__, other);
  }

  c10::SymNode gt(const c10::SymNode& other) override {
    return dispatch_common_(__FUNCTION__, other);
  }

  c10::SymNode lt(const c10::SymNode& other) override {
    return dispatch_common_(__FUNCTION__, other);
  }

  c10::SymNode le(const c10::SymNode& other) override {
    return dispatch_common_(__FUNCTION__, other);
  }

  c10::SymNode ge(const c10::SymNode& other) override {
    return dispatch_common_(__FUNCTION__, other);
  }

  c10::SymNode min(const c10::SymNode& other) override {
    return dispatch_common_(__FUNCTION__, other);
  }
  c10::SymNode max(const c10::SymNode& other) override {
    return dispatch_common_(__FUNCTION__, other);
  }

  c10::SymNode ceil() override {
    return dispatch_common_(__FUNCTION__);
  }

  c10::SymNode floor() override {
    return dispatch_common_(__FUNCTION__);
  }

  c10::SymNode neg() override {
    return dispatch_common_(__FUNCTION__);
  }

  c10::SymNode clone() override {
    return dispatch_common_(__FUNCTION__);
  }

  c10::SymNode sym_float() override {
    return dispatch_common_(__FUNCTION__);
  }

  py::handle getPyObj() {
    return py::handle(pyobj_.get()->ptr(getPyInterpreter()));
  }
  std::shared_ptr<c10::SafePyObject> pyobj_ = nullptr;
};

} // namespace impl
} // namespace torch
