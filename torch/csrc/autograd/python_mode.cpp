#include <torch/csrc/autograd/python_mode.h>
#include <torch/csrc/python_headers.h>
#include <torch/csrc/autograd/python_variable.h>
#include <ATen/PythonModeTLS.h>

namespace torch { namespace autograd {

PythonTorchDispatchTypeObject::PythonTorchDispatchTypeObject(
    PyObject* torch_dispatch_type_object,
    c10::impl::PyInterpreter* pyinterpreter) {
  data_ = torch_dispatch_type_object;
  pyinterpreter_ = pyinterpreter;
  Py_INCREF(data_);
}

PythonTorchDispatchTypeObject::~PythonTorchDispatchTypeObject() {
  Py_DECREF(data_);
}

PyObject* PythonTorchDispatchTypeObject::ptr() const {
  return data_;
}

c10::impl::PyInterpreter* PythonTorchDispatchTypeObject::pyinterpreter() const {
  return pyinterpreter_;
}

void PythonMode::enter(PyObject* type) {
  if (at::impl::PythonModeTLS::get_state()) {
    TORCH_CHECK(
        false,
        "python mode has already been set. We do not yet support nested python ",
        "mode. Please reset it before setting it again.")
  }
  auto state = std::make_shared<PythonTorchDispatchTypeObject>(type, getPyInterpreter());
  at::impl::PythonModeTLS::set_state(state);
}

void PythonMode::exit() {
  TORCH_INTERNAL_ASSERT(at::impl::PythonModeTLS::get_state(), "exiting Python Mode but it wasn't set!");
  at::impl::PythonModeTLS::reset_state();
}

}}
