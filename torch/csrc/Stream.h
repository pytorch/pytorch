#ifndef THP_STREAM_INC
#define THP_STREAM_INC

#include <c10/core/Stream.h>
#include <c10/macros/Export.h>
#include <torch/csrc/python_headers.h>

struct THPStream {
  PyObject_HEAD
  int64_t stream_id;
  int64_t device_type;
  int64_t device_index;
  // Used to switch stream context management, initialized lazily.
  PyObject* context;
};
extern TORCH_API PyTypeObject* THPStreamClass;

void THPStream_init(PyObject* module);

inline bool THPStream_Check(PyObject* obj) {
  return THPStreamClass && PyObject_IsInstance(obj, (PyObject*)THPStreamClass);
}

PyObject* THPStream_Wrap(const c10::Stream& stream);

#endif // THP_STREAM_INC
