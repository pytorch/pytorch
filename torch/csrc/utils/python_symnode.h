#pragma once

#include <c10/core/SafePyObject.h>
#include <c10/core/SymNodeImpl.h>

#include <torch/csrc/PyInterpreter.h>
#include <torch/csrc/autograd/python_variable.h>
#include <torch/csrc/utils/pybind.h>

namespace torch {

TORCH_PYTHON_API py::handle get_symint_class();
TORCH_PYTHON_API py::handle get_symfloat_class();
TORCH_PYTHON_API py::handle get_symbool_class();
TORCH_PYTHON_API py::handle get_dynint_class();

// NB: These functions must not be called too early, otherwise torch not setup.
// Alternate design is to have torch "register" the object to us
inline bool is_symint(py::handle obj) {
  return py::isinstance(obj, get_symint_class());
}
inline bool is_symfloat(py::handle obj) {
  return py::isinstance(obj, get_symfloat_class());
}
inline bool is_symbool(py::handle obj) {
  return py::isinstance(obj, get_symbool_class());
}
inline bool is_dynint(py::handle obj) {
  return py::isinstance(obj, get_dynint_class());
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
  }

  c10::SymNode wrap_int(int64_t num) override {
    py::gil_scoped_acquire acquire;
    auto r = getPyObj().attr("wrap_int")(num);
    return c10::make_intrusive<PythonSymNodeImpl>(std::move(r));
  }

  c10::SymNode wrap_float(double num) override {
    py::gil_scoped_acquire acquire;
    auto r = getPyObj().attr("wrap_float")(num);
    return c10::make_intrusive<PythonSymNodeImpl>(std::move(r));
  }

  c10::SymNode wrap_bool(bool num) override {
    py::gil_scoped_acquire acquire;
    auto r = getPyObj().attr("wrap_bool")(num);
    return c10::make_intrusive<PythonSymNodeImpl>(std::move(r));
  }

#define TORCH_SYMNODE_SIZES_STRIDES(n)                                        \
  c10::SymNode n(                                                             \
      c10::ArrayRef<c10::SymNode> sizes, c10::ArrayRef<c10::SymNode> strides) \
      override {                                                              \
    py::gil_scoped_acquire acquire;                                           \
    auto r = getPyObj().attr(#n)(sizes, strides);                             \
    return c10::make_intrusive<PythonSymNodeImpl>(std::move(r));              \
  }

  // clang-format off
    TORCH_SYMNODE_SIZES_STRIDES(is_contiguous)
    TORCH_SYMNODE_SIZES_STRIDES(is_channels_last_contiguous_2d)
    TORCH_SYMNODE_SIZES_STRIDES(is_channels_last_contiguous_3d)
    TORCH_SYMNODE_SIZES_STRIDES(is_channels_last_strides_2d)
    TORCH_SYMNODE_SIZES_STRIDES(is_channels_last_strides_3d)
    TORCH_SYMNODE_SIZES_STRIDES(is_non_overlapping_and_dense)
  // clang-format on

#undef TORCH_SYMNODE_SIZES_STRIDES

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

  bool is_bool() override {
    py::gil_scoped_acquire acquire;
    return getPyObj().attr("is_bool")().is(py::handle(Py_True));
  }

  bool is_nested_int() const override {
    py::gil_scoped_acquire acquire;
    return getPyObj().attr("is_nested_int")().is(py::handle(Py_True));
  }

  bool has_hint() override {
    py::gil_scoped_acquire acquire;
    return getPyObj().attr("has_hint")().is(py::handle(Py_True));
  }

  int64_t guard_int(const char* file, int64_t line) override {
    py::gil_scoped_acquire acquire;
    return getPyObj().attr("guard_int")(file, line).cast<int64_t>();
  }

  double guard_float(const char* file, int64_t line) override {
    py::gil_scoped_acquire acquire;
    return getPyObj().attr("guard_float")(file, line).cast<double>();
  }

  bool guard_bool(const char* file, int64_t line) override {
    py::gil_scoped_acquire acquire;
    return getPyObj().attr("guard_bool")(file, line).cast<bool>();
  }

  bool expect_true(const char* file, int64_t line) override {
    py::gil_scoped_acquire acquire;
    return getPyObj().attr("expect_true")(file, line).cast<bool>();
  }

  bool expect_size(const char* file, int64_t line) override {
    py::gil_scoped_acquire acquire;
    return getPyObj().attr("expect_size")(file, line).cast<bool>();
  }

  bool guard_size_oblivious(const char* file, int64_t line) override {
    py::gil_scoped_acquire acquire;
    return getPyObj().attr("guard_size_oblivious")(file, line).cast<bool>();
  }

  bool guard_or_false(const char* file, int64_t line) override {
    py::gil_scoped_acquire acquire;
    return getPyObj().attr("guard_or_false")(file, line).cast<bool>();
  }

  bool statically_known_true(const char* file, int64_t line) override {
    py::gil_scoped_acquire acquire;
    return getPyObj().attr("statically_known_true")(file, line).cast<bool>();
  }

  bool guard_or_true(const char* file, int64_t line) override {
    py::gil_scoped_acquire acquire;
    return getPyObj().attr("guard_or_true")(file, line).cast<bool>();
  }

  int64_t int_() override {
    py::gil_scoped_acquire acquire;
    return getPyObj().attr("int_")().cast<int64_t>();
  }

  std::optional<int64_t> maybe_as_int() override {
    py::gil_scoped_acquire acquire;
    const auto& r = getPyObj().attr("maybe_as_int")();
    if (r.is_none()) {
      return std::nullopt;
    } else {
      return r.cast<int64_t>();
    }
  }

  std::string str() override {
    py::gil_scoped_acquire acquire;
    return getPyObj().attr("str")().cast<std::string>();
  }

  std::string _graph_repr() override {
    py::gil_scoped_acquire acquire;
    return getPyObj().attr("_graph_repr")().cast<std::string>();
  }

  c10::SymNode dispatch_sym_ite_(
      const char* fname,
      const c10::SymNode& other,
      const c10::SymNode& third) {
    auto pother = dynamic_cast<PythonSymNodeImpl*>(other.get());
    auto pthird = dynamic_cast<PythonSymNodeImpl*>(third.get());
    TORCH_CHECK(pother);
    TORCH_CHECK(pthird);
    py::gil_scoped_acquire acquire;
    auto r = getPyObj().attr(fname)(pother->getPyObj(), pthird->getPyObj());
    return c10::make_intrusive<PythonSymNodeImpl>(r);
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
    return dispatch_common_(__func__, other);
  }

  c10::SymNode sub(const c10::SymNode& other) override {
    return dispatch_common_(__func__, other);
  }

  c10::SymNode mul(const c10::SymNode& other) override {
    return dispatch_common_(__func__, other);
  }

  c10::SymNode truediv(const c10::SymNode& other) override {
    return dispatch_common_(__func__, other);
  }

  c10::SymNode float_truediv(const c10::SymNode& other) override {
    return dispatch_common_(__func__, other);
  }

  c10::SymNode int_truediv(const c10::SymNode& other) override {
    return dispatch_common_(__func__, other);
  }

  c10::SymNode pow(const c10::SymNode& other) override {
    return dispatch_common_(__func__, other);
  }

  c10::SymNode float_pow(const c10::SymNode& other) override {
    return dispatch_common_(__func__, other);
  }

  c10::SymNode pow_by_natural(const c10::SymNode& other) override {
    return dispatch_common_(__func__, other);
  }

  c10::SymNode floordiv(const c10::SymNode& other) override {
    return dispatch_common_(__func__, other);
  }

  c10::SymNode int_floordiv(const c10::SymNode& other) override {
    return dispatch_common_(__func__, other);
  }

  c10::SymNode mod(const c10::SymNode& other) override {
    return dispatch_common_(__func__, other);
  }

  c10::SymNode eq(const c10::SymNode& other) override {
    return dispatch_common_(__func__, other);
  }

  c10::SymNode ne(const c10::SymNode& other) override {
    return dispatch_common_(__func__, other);
  }

  c10::SymNode gt(const c10::SymNode& other) override {
    return dispatch_common_(__func__, other);
  }

  c10::SymNode lt(const c10::SymNode& other) override {
    return dispatch_common_(__func__, other);
  }

  c10::SymNode le(const c10::SymNode& other) override {
    return dispatch_common_(__func__, other);
  }

  c10::SymNode ge(const c10::SymNode& other) override {
    return dispatch_common_(__func__, other);
  }

  c10::SymNode sym_min(const c10::SymNode& other) override {
    return dispatch_common_(__func__, other);
  }
  c10::SymNode sym_max(const c10::SymNode& other) override {
    return dispatch_common_(__func__, other);
  }

  c10::SymNode sym_and(const c10::SymNode& other) override {
    return dispatch_common_(__func__, other);
  }

  c10::SymNode sym_or(const c10::SymNode& other) override {
    return dispatch_common_(__func__, other);
  }

  c10::SymNode sym_ite(const c10::SymNode& other, const c10::SymNode& third)
      override {
    return dispatch_sym_ite_(__func__, other, third);
  }

  c10::SymNode sym_not() override {
    return dispatch_common_(__func__);
  }

  c10::SymNode ceil() override {
    return dispatch_common_(__func__);
  }

  c10::SymNode floor() override {
    return dispatch_common_(__func__);
  }

  c10::SymNode neg() override {
    return dispatch_common_(__func__);
  }

  c10::SymNode clone() override {
    return dispatch_common_(__func__);
  }

  c10::SymNode sym_float() override {
    return dispatch_common_(__func__);
  }

  py::handle getPyObj() const {
    return py::handle(pyobj_->ptr(getPyInterpreter()));
  }
  std::shared_ptr<c10::SafePyObject> pyobj_ = nullptr;
};

} // namespace impl
} // namespace torch
