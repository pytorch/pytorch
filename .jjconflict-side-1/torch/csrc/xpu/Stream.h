#pragma once

#include <c10/xpu/XPUStream.h>
#include <torch/csrc/Stream.h>
#include <torch/csrc/python_headers.h>

// NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
struct THXPStream : THPStream {
  at::xpu::XPUStream xpu_stream;
};
extern PyObject* THXPStreamClass;

void THXPStream_init(PyObject* module);

inline bool THXPStream_Check(PyObject* obj) {
  return THXPStreamClass && PyObject_IsInstance(obj, THXPStreamClass);
}
