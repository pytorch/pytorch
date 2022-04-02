#include <c10/core/SafePyObject.h>

namespace c10 {

TorchDispatchTypeObject::TorchDispatchTypeObject(
    PyObject* type_object,
    c10::impl::PyInterpreter* pyinterpreter)
    : data_(type_object), pyinterpreter_(pyinterpreter) {}

TorchDispatchTypeObject::~TorchDispatchTypeObject() {
  pyinterpreter_->decref(data_, /*is_tensor*/ false);
}

PyObject* SafePyObject::ptr(const c10::impl::PyInterpreter* interpreter) const {
  TORCH_INTERNAL_ASSERT(interpreter == pyinterpreter_);
  return data_;
}

} // namespace c10
