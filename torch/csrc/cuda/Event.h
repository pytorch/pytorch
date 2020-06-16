#ifndef THCP_EVENT_INC
#define THCP_EVENT_INC

#include <ATen/hip/HIPEvent.h>
#include <torch/csrc/python_headers.h>
#include <THH/THH.h>

struct THCPEvent {
  PyObject_HEAD
  at::cuda::CUDAEvent cuda_event;
};
extern PyObject *THCPEventClass;

void THCPEvent_init(PyObject *module);

inline bool THCPEvent_Check(PyObject* obj) {
  return THCPEventClass && PyObject_IsInstance(obj, THCPEventClass);
}

#endif // THCP_EVENT_INC
