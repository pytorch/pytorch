#pragma once

#include <ATen/mps/MPSStream.h>
#include <torch/csrc/Stream.h>
#include <torch/csrc/python_headers.h>

// NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
struct THMPStream : THPStream {
  // Non-owning pointer to one of the streams in the MPS stream pool
  at::mps::MPSStream* mps_stream;
};
extern PyObject* THMPStreamClass;

void THMPStream_init(PyObject* module);

inline bool THMPStream_Check(PyObject* obj) {
  return THMPStreamClass && PyObject_IsInstance(obj, THMPStreamClass);
}
