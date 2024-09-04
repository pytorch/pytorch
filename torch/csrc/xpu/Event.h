#pragma once

#include <ATen/xpu/XPUEvent.h>
#include <torch/csrc/Event.h>
#include <torch/csrc/python_headers.h>

struct THXPEvent : THPEvent {
  at::xpu::XPUEvent xpu_event;
};
extern PyObject* THXPEventClass;

void THXPEvent_init(PyObject* module);

inline bool THXPEvent_Check(PyObject* obj) {
  return THXPEventClass && PyObject_IsInstance(obj, THXPEventClass);
}
