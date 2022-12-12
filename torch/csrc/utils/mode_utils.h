#pragma once
#include <c10/core/ModePyObjTrampoline.h>
#include <torch/csrc/Exceptions.h>
#include <torch/csrc/autograd/python_variable.h>
#include <torch/csrc/utils/python_strings.h>

namespace torch {
struct ConcreteModePyObjTrampoline : virtual c10::ModePyObjTrampoline {
  using ModePyObjTrampoline::ModePyObjTrampoline;

  void mode_state_push_trampoline() const override {
    PyObject* mode_obj = this->ptr(getPyInterpreter());
    const char* check_mode_push_name = "check_mode_state_push";

    // if the guard is the one pushing the mode, there might be active exceptions.
    // This temporarily removes them in order to check before putting the mode
    // back on the stack

    // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
    PyObject *type, *value, *traceback;
    PyErr_Fetch(&type, &value, &traceback);

    py::object run_function =
        PyObject_FastGetAttrString(mode_obj, check_mode_push_name);
    if (!run_function) {
      TORCH_INTERNAL_ASSERT(0);
    }

    const auto ret = py::reinterpret_steal<py::object>(
        PyObject_CallMethod(mode_obj, check_mode_push_name, ""));
    if (ret.ptr() == nullptr) {
      throw python_error();
    }
    PyErr_Restore(type, value, traceback);
  }

  void mode_state_pop_trampoline() const override {
    const char* check_mode_pop_name = "check_mode_state_pop";
    PyObject* mode_obj = this->ptr(getPyInterpreter());

    const auto run_function =
        PyObject_FastGetAttrString(mode_obj, check_mode_pop_name);
    if (!run_function) {
      TORCH_INTERNAL_ASSERT(0);
    }

    const auto ret = py::reinterpret_steal<py::object>(
        PyObject_CallMethod(mode_obj, check_mode_pop_name, ""));
    if (ret.ptr() == nullptr) {
      throw python_error();
    }
  }
};

} // namespace torch