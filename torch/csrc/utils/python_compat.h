#pragma once

#include <torch/csrc/python_headers.h>

#define MAYBE_METH_FASTCALL METH_FASTCALL
#define MAYBE_WRAP_FASTCALL(f) (PyCFunction)(void (*)(void)) f

// PyPy 3.6 does not yet have PySlice_Unpack
#if PY_VERSION_HEX < 0x03060100 || defined(PYPY_VERSION)

// PySlice_Unpack not introduced till python 3.6.1
// included here for backwards compatibility
// https://docs.python.org/3/c-api/slice.html#c.PySlice_Unpack
// https://github.com/python/cpython/blob/master/Objects/sliceobject.c#L196

inline int __PySlice_Unpack(
    PyObject* _r,
    Py_ssize_t* start,
    Py_ssize_t* stop,
    Py_ssize_t* step) {
  PySliceObject* r = (PySliceObject*)_r;
  /* this is harder to get right than you might think */

  // Py_BUILD_ASSERT replaced because it is not available in all versions
  static_assert(PY_SSIZE_T_MIN + 1 <= -PY_SSIZE_T_MAX, "Build failed");

  if (r->step == Py_None) {
    *step = 1;
  } else {
    if (!_PyEval_SliceIndex(r->step, step))
      return -1;
    if (*step == 0) {
      PyErr_SetString(PyExc_ValueError, "slice step cannot be zero");
      return -1;
    }
    /* Here *step might be -PY_SSIZE_T_MAX-1; in this case we replace it
     * with -PY_SSIZE_T_MAX.  This doesn't affect the semantics, and it
     * guards against later undefined behaviour resulting from code that
     * does "step = -step" as part of a slice reversal.
     */
    if (*step < -PY_SSIZE_T_MAX)
      *step = -PY_SSIZE_T_MAX;
  }

  if (r->start == Py_None) {
    *start = *step < 0 ? PY_SSIZE_T_MAX : 0;
  } else {
    if (!_PyEval_SliceIndex(r->start, start))
      return -1;
  }

  if (r->stop == Py_None) {
    *stop = *step < 0 ? PY_SSIZE_T_MIN : PY_SSIZE_T_MAX;
  } else {
    if (!_PyEval_SliceIndex(r->stop, stop))
      return -1;
  }

  return 0;
}

#define THPUtils_unpackSlice(SLICE, START, STOP, STEP) \
  (__PySlice_Unpack(SLICE, START, STOP, STEP) == 0)
#else
#define THPUtils_unpackSlice(SLICE, START, STOP, STEP) \
  (PySlice_Unpack(SLICE, START, STOP, STEP) == 0)
#endif

#define THPUtils_parseSlice(SLICE, LEN, START, STOP, LENGTH, STEP) \
  (PySlice_GetIndicesEx(SLICE, LEN, START, STOP, LENGTH, STEP) == 0)

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
