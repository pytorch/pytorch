#include <torch/csrc/autograd/python_mode.h>
#include <torch/csrc/python_headers.h>
#include <torch/csrc/autograd/python_variable.h>
#include <ATen/core/PythonModeTLS.h>
#include <c10/core/TensorImpl.h>

namespace torch { namespace autograd {

void PythonMode::enter(PyObject* type, PyObject* mode_ctx) {
  // TorchDispatchTypeObject steals a reference, See NOTE [What is TorchDispatchTypeObject?]
  Py_INCREF(type);
  Py_INCREF(mode_ctx);
  auto state = std::make_shared<c10::TorchDispatchTypeObject>(type, getPyInterpreter(), mode_ctx);
  at::impl::PythonModeTLS::push_mode(state);
}

std::tuple<PyObject*,PyObject*> PythonMode::exit() {
  TORCH_INTERNAL_ASSERT(at::impl::PythonModeTLS::get_state().size() > 0,
                        "exiting Python Mode but it wasn't set!");
  auto result = at::impl::PythonModeTLS::pop_mode();
  return std::make_tuple(result->ptr(), result->mode_ctx());
}

}}
