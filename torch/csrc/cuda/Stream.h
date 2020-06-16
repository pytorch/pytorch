#ifndef THCP_STREAM_INC
#define THCP_STREAM_INC

#include <ATen/hip/impl/HIPStreamMasqueradingAsCUDA.h>
#include <torch/csrc/python_headers.h>
#include <THH/THH.h>

struct THCPStream {
  PyObject_HEAD
  uint64_t cdata;
  at::hip::HIPStreamMasqueradingAsCUDA cuda_stream;
};
extern PyObject *THCPStreamClass;

void THCPStream_init(PyObject *module);

inline bool THCPStream_Check(PyObject* obj) {
  return THCPStreamClass && PyObject_IsInstance(obj, THCPStreamClass);
}

#endif // THCP_STREAM_INC
