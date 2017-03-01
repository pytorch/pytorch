#ifndef THCP_STREAM_INC
#define THCP_STREAM_INC

#include <Python.h>
#include <THC/THC.h>


struct THCPStream {
  PyObject_HEAD
  THCStream *cdata;
  int device;
  cudaStream_t cuda_stream;
};
extern PyObject *THCPStreamClass;

bool THCPStream_init(PyObject *module);

inline bool THCPStream_Check(PyObject* obj) {
  return THCPStreamClass && PyObject_IsInstance(obj, THCPStreamClass);
}

#endif // THCP_STREAM_INC
