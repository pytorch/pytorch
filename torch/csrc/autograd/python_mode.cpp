#include <torch/csrc/autograd/python_mode.h>
#include <torch/csrc/python_headers.h>
#include <torch/csrc/autograd/python_variable.h>
#include <ATen/core/PythonModeTLS.h>
#include <c10/core/TensorImpl.h>

namespace torch { namespace autograd {

void PythonMode::enter(PyObject* type) {
  if (at::impl::PythonModeTLS::get_state()) {
    TORCH_CHECK(
        false,
        "python mode has already been set. We do not yet support nested python ",
        "mode. Please file us an issue and reset it before setting it again.")
  }
  // TorchDispatchTypeObject steals a reference, See NOTE [What is TorchDispatchTypeObject?]
  Py_INCREF(type);
  auto state = std::make_shared<c10::TorchDispatchTypeObject>(type, getPyInterpreter());
  at::impl::PythonModeTLS::set_state(state);
}

void PythonMode::exit() {
  TORCH_INTERNAL_ASSERT(at::impl::PythonModeTLS::get_state(), "exiting Python Mode but it wasn't set!");
  at::impl::PythonModeTLS::reset_state();
}

}}
