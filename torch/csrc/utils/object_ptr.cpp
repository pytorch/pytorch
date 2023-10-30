#include <c10/macros/Macros.h>
#include <torch/csrc/utils/object_ptr.h>

#include <torch/csrc/python_headers.h>

template <>
void THPPointer<PyObject>::free() {
  if (ptr && C10_LIKELY(Py_IsInitialized()))
    Py_DECREF(ptr);
}

template class THPPointer<PyObject>;

template <>
void THPPointer<PyCodeObject>::free() {
  if (ptr && C10_LIKELY(Py_IsInitialized()))
    Py_DECREF(ptr);
}

template class THPPointer<PyCodeObject>;

template <>
void THPPointer<PyFrameObject>::free() {
  if (ptr && C10_LIKELY(Py_IsInitialized()))
    Py_DECREF(ptr);
}

template class THPPointer<PyFrameObject>;
