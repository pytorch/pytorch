#ifndef THCP_STREAM_INC
#define THCP_STREAM_INC

#include <c10/cuda/CUDAStream.h>
#include <torch/csrc/Stream.h>
#include <torch/csrc/python_headers.h>

// NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
struct THCPStream : THPStream {
  at::cuda::CUDAStream cuda_stream;
};
extern PyObject* THCPStreamClass;

void THCPStream_init(PyObject* module);

inline bool THCPStream_Check(PyObject* obj) {
  return THCPStreamClass && PyObject_IsInstance(obj, THCPStreamClass);
}

#endif // THCP_STREAM_INC
