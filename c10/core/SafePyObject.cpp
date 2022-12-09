#include <c10/core/SafePyObject.h>
#include <c10/core/TensorImpl.h>

namespace c10 {

PyObject* SafePyObject::ptr(const c10::impl::PyInterpreter* interpreter) const {
  TORCH_INTERNAL_ASSERT(interpreter == pyinterpreter_);
  return data_;
}

PyObject* SafePyHandle::ptr(const c10::impl::PyInterpreter* interpreter) const {
  TORCH_INTERNAL_ASSERT(interpreter == pyinterpreter_);
  return data_;
}

} // namespace c10
