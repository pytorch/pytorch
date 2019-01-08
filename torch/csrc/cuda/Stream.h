#ifndef THCP_STREAM_INC
#define THCP_STREAM_INC

#include <c10/cuda/CUDAStream.h>
#include <torch/csrc/python_headers.h>
#include <THC/THC.h>

struct THCPStream {
  PyObject_HEAD
  // Can't conveniently put an actual c10::Stream here, because the
  // class is not POD.  (We could put it here, but then we'd be on
  // the hook for placement-new'ing/delete'ing it; simpler to just
  // rely on the packed representation...)
  uint64_t cdata;
  at::cuda::CUDAStream cuda_stream;
};
extern PyObject *THCPStreamClass;

bool THCPStream_init(PyObject *module);

inline bool THCPStream_Check(PyObject* obj) {
  return THCPStreamClass && PyObject_IsInstance(obj, THCPStreamClass);
}

#endif // THCP_STREAM_INC
