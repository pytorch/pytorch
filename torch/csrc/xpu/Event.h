#pragma once

#include <ATen/xpu/XPUEvent.h>
#include <torch/csrc/python_headers.h>

struct THXPEvent {
  PyObject_HEAD at::xpu::XPUEvent xpu_event;
};
extern PyObject* THXPEventClass;

void THXPEvent_init(PyObject* module);

inline bool THXPEvent_Check(PyObject* obj) {
  return THXPEventClass && PyObject_IsInstance(obj, THXPEventClass);
}
