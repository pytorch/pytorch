#include <torch/csrc/utils/object_ptr.h>

#include <torch/csrc/python_headers.h>

template <>
void THPPointer<PyObject>::free() {
  if (ptr)
    Py_DECREF(ptr);
}

template class TORCH_PYTHON_API THPPointer<PyObject>;

template <>
void THPPointer<PyCodeObject>::free() {
  if (ptr)
    Py_DECREF(ptr);
}

template class THPPointer<PyCodeObject>;

template <>
void THPPointer<PyFrameObject>::free() {
  if (ptr)
    Py_DECREF(ptr);
}

template class THPPointer<PyFrameObject>;
