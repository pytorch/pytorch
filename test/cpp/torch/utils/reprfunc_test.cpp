#include <torch/csrc/utils/reprfunc.h>

#include <Python.h>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cassert>
#include <string_view>

namespace torch {

// Simple type to create a reprfunc from.
class PyType {
 public:
  explicit PyType(std::string repr) : repr_(std::move(repr)) {}

  static auto repr_impl(PyType* self) -> PyObject* {
    return PyUnicode_FromStringAndSize(
        self->repr_.data(), self->repr_.length());
  }

  auto repr() const -> std::string_view {
    return repr_;
  }

 private:
  std::string repr_;
};

namespace {

// Gets a std::string_view from a PyObject.
std::string_view as_string_view(PyObject* obj) {
  assert(PyUnicode_Check(obj));
  const char* data = PyUnicode_AsUTF8(obj);
  assert(data != nullptr);
  return std::string_view(data);
}

// Matcher that verifies a PyObject is equivalent to a string.
MATCHER_P(EqPyString, want, "") {
  std::string_view got = as_string_view(arg);
  *result_listener << "i.e. " << got;
  return got == want;
}

TEST(reprfunc_test, adapter) {
  reprfunc func = as_reprfunc<PyType, &PyType::repr_impl>();

  PyType self("PyType('value')");
  auto obj = reinterpret_cast<PyObject const&>(self);

  ASSERT_THAT(func(&obj), EqPyString(self.repr()));
}

TEST(reprfunc_test, adapter_macro) {
  reprfunc func = TORCH_AS_REPRFUNC(&PyType::repr_impl);

  PyType self("PyType('value')");
  auto obj = reinterpret_cast<PyObject const&>(self);

  ASSERT_THAT(func(&obj), EqPyString(self.repr()));
}

} // namespace
} // namespace torch
