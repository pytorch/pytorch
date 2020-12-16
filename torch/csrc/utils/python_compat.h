#pragma once

#include <torch/csrc/python_headers.h>

#if PY_VERSION_HEX < 0x03070000
// METH_FASTCALL was introduced in Python 3.7, so we wrap _PyCFunctionFast
// signatures for earlier versions.

template <PyObject* (*f)(PyObject*, PyObject *const *, Py_ssize_t)>
PyObject* maybe_wrap_fastcall(PyObject *module, PyObject *args) {
  return f(
    module,

    // _PyTuple_ITEMS
    //   Because this is only a compat shim for Python 3.6, we don't have
    //   to worry about the representation changing.
    ((PyTupleObject *)args)->ob_item,
    PySequence_Fast_GET_SIZE(args)
  );
}

#define MAYBE_METH_FASTCALL METH_VARARGS
#define MAYBE_WRAP_FASTCALL(f) maybe_wrap_fastcall<f>

#else

#define MAYBE_METH_FASTCALL METH_FASTCALL
#define MAYBE_WRAP_FASTCALL(f) (PyCFunction)(void(*)(void))f

#endif

// PyPy 3.6 does not yet have PySlice_Unpack
#if PY_VERSION_HEX < 0x03060100 || defined(PYPY_VERSION)

// PySlice_Unpack not introduced till python 3.6.1
// included here for backwards compatibility
// https://docs.python.org/3/c-api/slice.html#c.PySlice_Unpack
// https://github.com/python/cpython/blob/master/Objects/sliceobject.c#L196

inline int
__PySlice_Unpack(PyObject *_r,
               Py_ssize_t *start, Py_ssize_t *stop, Py_ssize_t *step)
{
    PySliceObject *r = (PySliceObject*)_r;
    /* this is harder to get right than you might think */

    // Py_BUILD_ASSERT replaced because it is not available in all versions
    static_assert(PY_SSIZE_T_MIN + 1 <= -PY_SSIZE_T_MAX, "Build failed");

    if (r->step == Py_None) {
        *step = 1;
    }
    else {
        if (!_PyEval_SliceIndex(r->step, step)) return -1;
        if (*step == 0) {
            PyErr_SetString(PyExc_ValueError,
                            "slice step cannot be zero");
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
    }
    else {
        if (!_PyEval_SliceIndex(r->start, start)) return -1;
    }

    if (r->stop == Py_None) {
        *stop = *step < 0 ? PY_SSIZE_T_MIN : PY_SSIZE_T_MAX;
    }
    else {
        if (!_PyEval_SliceIndex(r->stop, stop)) return -1;
    }

    return 0;
}

#define THPUtils_unpackSlice(SLICE, START, STOP, STEP) \
  (__PySlice_Unpack(SLICE, START, STOP, STEP) == 0)
#else
#define THPUtils_unpackSlice(SLICE, START, STOP, STEP) \
  (PySlice_Unpack(SLICE, START, STOP, STEP) == 0)
#endif

// https://bugsfiles.kde.org/attachment.cgi?id=61186
#if PY_VERSION_HEX >= 0x03020000
#define THPUtils_parseSlice(SLICE, LEN, START, STOP, LENGTH, STEP) \
  (PySlice_GetIndicesEx(SLICE, LEN, START, STOP, LENGTH, STEP) == 0)
#else
#define THPUtils_parseSlice(SLICE, LEN, START, STOP, LENGTH, STEP) \
  (PySlice_GetIndicesEx((PySliceObject*)SLICE, LEN, START, STOP, LENGTH, STEP) == 0)
#endif

// This function was introduced in Python 3.4
#if PY_VERSION_HEX < 0x03040000
inline int
PyGILState_Check() {
  PyThreadState * tstate = _PyThreadState_Current;
  return tstate && (tstate == PyGILState_GetThisThreadState());
}
#endif
