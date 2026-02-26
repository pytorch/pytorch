#ifndef THP_EVENT_INC
#define THP_EVENT_INC

#include <c10/core/Event.h>
#include <torch/csrc/python_headers.h>

struct TORCH_API THPEvent {
  PyObject_HEAD
  c10::Event event;
  PyObject* weakreflist;
};
TORCH_API extern PyTypeObject* THPEventClass;
TORCH_API extern PyTypeObject THPEventType;

TORCH_API void THPEvent_init(PyObject* module);
TORCH_API PyObject* THPEvent_new(
    c10::DeviceType device_type,
    c10::EventFlag flag);
inline bool THPEvent_Check(PyObject* obj) {
  return THPEventClass && PyObject_IsInstance(obj, (PyObject*)THPEventClass);
}

#endif // THP_EVENT_INC
