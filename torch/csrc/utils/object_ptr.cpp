#include "torch/csrc/utils/object_ptr.h"

#include <Python.h>

template<>
void THPPointer<PyObject>::free() {
  if (ptr)
    Py_DECREF(ptr);
}

template class THPPointer<PyObject>;
