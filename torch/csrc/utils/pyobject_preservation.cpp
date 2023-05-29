#include <torch/csrc/utils/pyobject_preservation.h>

#include <structmember.h>

void clear_slots(PyTypeObject* type, PyObject* self) {
  Py_ssize_t n = Py_SIZE(type);
  PyMemberDef* mp = type->tp_members;

  for (Py_ssize_t i = 0; i < n; i++, mp++) {
    if (mp->type == T_OBJECT_EX && !(mp->flags & READONLY)) {
      char* addr = (char*)self + mp->offset;
      PyObject* obj = *(PyObject**)addr;
      if (obj != nullptr) {
        *(PyObject**)addr = nullptr;
        Py_DECREF(obj);
      }
    }
  }
}
