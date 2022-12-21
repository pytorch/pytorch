#pragma once

#include <torch/csrc/python_headers.h>

// Compat macros macros taken from
// https://docs.python.org/3.11/whatsnew/3.11.html

#if PY_VERSION_HEX < 0x030900B1
static inline PyCodeObject* PyFrame_GetCode(PyFrameObject* frame) {
  Py_INCREF(frame->f_code);
  return frame->f_code;
}

static inline PyFrameObject* PyFrame_GetBack(PyFrameObject* frame) {
  Py_XINCREF(frame->f_back);
  return frame->f_back;
}

static inline PyFrameObject* PyThreadState_GetFrame(PyThreadState* state) {
  Py_XINCREF(state->frame);
  return state->frame;
}
#endif

#if PY_VERSION_HEX < 0x030900A4 && !defined(Py_SET_TYPE)
static inline void _Py_SET_TYPE(PyObject* ob, PyTypeObject* type) {
  ob->ob_type = type;
}
#define Py_SET_TYPE(ob, type) _Py_SET_TYPE((PyObject*)(ob), type)
#endif

#if PY_VERSION_HEX < ((3 << 24) | (11 << 16) | (0 << 8) | (0xA << 4) | (4 << 0))
static inline PyObject* PyFrame_GetLocals(PyFrameObject* frame) {
  PyFrame_FastToLocals(frame);
  auto res = frame->f_locals;

  // To match PyFrame_GetLocals, return a new reference
  Py_INCREF(res);
  return res;
}

static inline int PyFrame_GetLasti(PyFrameObject* frame) {
  return frame->f_lasti;
}
#endif
