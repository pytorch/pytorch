#include <c10/core/SafePyObject.h>
#include <c10/core/TensorImpl.h>

namespace c10 {

bool SafePyObject::has_same_interpreter(const c10::impl::PyInterpreter* interpreter) const {
  return interpreter == pyinterpreter_;
}

PyObject* SafePyObject::ptr(const c10::impl::PyInterpreter* interpreter) const {
  TORCH_INTERNAL_ASSERT(has_same_interpreter(interpreter));
  return data_;
}

} // namespace c10
