#include <torch/csrc/utils/torch_dispatch_mode.h>

#include <torch/csrc/Exceptions.h>
#include <torch/csrc/autograd/python_variable.h>
#include <torch/csrc/utils/python_strings.h>

namespace torch {
namespace torch_dispatch_mode {
void push_onto_dispatch_stack(std::shared_ptr<at::SafePyObject> mode) {
  PyObject* mode_obj = mode->ptr(getPyInterpreter());
  const char* check_mode_push_name = "check_mode_state_push";
  py::object run_function = PyObject_FastGetAttrString(mode_obj, check_mode_push_name);
  if (!run_function) {
    TORCH_INTERNAL_ASSERT(0);
  }

  const auto ret = py::reinterpret_steal<py::object>(PyObject_CallMethod(
      mode_obj,
      check_mode_push_name,
      ""));
  if (ret.ptr() == nullptr) {
    throw python_error();
  }

  c10::impl::TorchDispatchModeTLS::unsafe_push_onto_stack(mode);
}

std::shared_ptr<at::SafePyObject> pop_dispatch_stack() {
  std::shared_ptr<at::SafePyObject> mode = c10::impl::TorchDispatchModeTLS::unsafe_pop_stack();
  PyObject* mode_obj = mode->ptr(getPyInterpreter());

  const auto run_function = PyObject_FastGetAttrString(mode_obj, "check_mode_state_pop");
  if (!run_function) {
    TORCH_INTERNAL_ASSERT(0);
  }

  const auto ret = py::reinterpret_steal<py::object>(PyObject_CallMethod(
      mode_obj,
      "check_mode_state_pop",
      ""));
  if (ret.ptr() == nullptr) {
    throw python_error();
  }

  return mode;
}

} // namespace torch_dispatch_mode
} // namespace torch
