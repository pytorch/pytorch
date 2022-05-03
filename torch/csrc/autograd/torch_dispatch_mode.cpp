#include <ATen/core/TorchDispatchModeTLS.h>
#include <c10/core/SafePyObject.h>
#include <c10/core/TensorImpl.h>
#include <torch/csrc/autograd/torch_dispatch_mode.h>
#include <torch/csrc/autograd/python_variable.h>
#include <torch/csrc/python_headers.h>

namespace torch { namespace autograd {

void TorchDispatchMode::enter(PyObject* type) {
  if (at::impl::TorchDispatchModeTLS::get_state()) {
    TORCH_CHECK(
        false,
        "torch dispatch mode has already been set. We do not yet support nested torch dispatch ",
        "mode. Please file us an issue and reset it before setting it again.")
  }
  // SafePyObject steals a reference, See NOTE [What is SafePyObject?]
  Py_INCREF(type);
  at::impl::TorchDispatchModeTLS::set_state(
      std::make_shared<c10::SafePyObject>(type, getPyInterpreter()));
}

void TorchDispatchMode::exit() {
  TORCH_INTERNAL_ASSERT(at::impl::TorchDispatchModeTLS::get_state(), "exiting TorchDispatch Mode but it wasn't set!");
  at::impl::TorchDispatchModeTLS::reset_state();
}

}}
