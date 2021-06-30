#ifndef THP_STREAM_INC
#define THP_STREAM_INC

#include <torch/csrc/python_headers.h>

struct THPStream {
  PyObject_HEAD
  uint64_t cdata;
};
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
extern PyTypeObject *THPStreamClass;

void THPStream_init(PyObject *module);

inline bool THPStream_Check(PyObject* obj) {
  return THPStreamClass && PyObject_IsInstance(obj, (PyObject*)THPStreamClass);
}

#endif // THP_STREAM_INC
