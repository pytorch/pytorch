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

bool THCPStream_init(PyObject *module);

#endif // THCP_STREAM_INC
