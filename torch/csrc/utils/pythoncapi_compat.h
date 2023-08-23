// Header file providing new C API functions to old Python versions.
//
// File distributed under the Zero Clause BSD (0BSD) license.
// Copyright Contributors to the pythoncapi_compat project.
//
// Homepage:
// https://github.com/python/pythoncapi_compat
//
// Latest version:
// https://raw.githubusercontent.com/python/pythoncapi_compat/master/pythoncapi_compat.h
//
// SPDX-License-Identifier: 0BSD

#ifndef PYTHONCAPI_COMPAT
#define PYTHONCAPI_COMPAT

#ifdef __cplusplus
extern "C" {
#endif

#include <Python.h>
#include "frameobject.h"          // PyFrameObject, PyFrame_GetBack()


// Compatibility with Visual Studio 2013 and older which don't support
// the inline keyword in C (only in C++): use __inline instead.
#if (defined(_MSC_VER) && _MSC_VER < 1900 \
     && !defined(__cplusplus) && !defined(inline))
#  define PYCAPI_COMPAT_STATIC_INLINE(TYPE) static __inline TYPE
#else
#  define PYCAPI_COMPAT_STATIC_INLINE(TYPE) static inline TYPE
#endif


#ifndef _Py_CAST
#  define _Py_CAST(type, expr) ((type)(expr))
#endif

// On C++11 and newer, _Py_NULL is defined as nullptr on C++11,
// otherwise it is defined as NULL.
#ifndef _Py_NULL
#  if defined(__cplusplus) && __cplusplus >= 201103
#    define _Py_NULL nullptr
#  else
#    define _Py_NULL NULL
#  endif
#endif

// Cast argument to PyObject* type.
#ifndef _PyObject_CAST
#  define _PyObject_CAST(op) _Py_CAST(PyObject*, op)
#endif


// bpo-42262 added Py_NewRef() to Python 3.10.0a3
#if PY_VERSION_HEX < 0x030A00A3 && !defined(Py_NewRef)
PYCAPI_COMPAT_STATIC_INLINE(PyObject*)
_Py_NewRef(PyObject *obj)
{
    Py_INCREF(obj);
    return obj;
}
#define Py_NewRef(obj) _Py_NewRef(_PyObject_CAST(obj))
#endif


// bpo-42262 added Py_XNewRef() to Python 3.10.0a3
#if PY_VERSION_HEX < 0x030A00A3 && !defined(Py_XNewRef)
PYCAPI_COMPAT_STATIC_INLINE(PyObject*)
_Py_XNewRef(PyObject *obj)
{
    Py_XINCREF(obj);
    return obj;
}
#define Py_XNewRef(obj) _Py_XNewRef(_PyObject_CAST(obj))
#endif


// bpo-39573 added Py_SET_REFCNT() to Python 3.9.0a4
#if PY_VERSION_HEX < 0x030900A4 && !defined(Py_SET_REFCNT)
PYCAPI_COMPAT_STATIC_INLINE(void)
_Py_SET_REFCNT(PyObject *ob, Py_ssize_t refcnt)
{
    ob->ob_refcnt = refcnt;
}
#define Py_SET_REFCNT(ob, refcnt) _Py_SET_REFCNT(_PyObject_CAST(ob), refcnt)
#endif


// Py_SETREF() and Py_XSETREF() were added to Python 3.5.2.
// It is excluded from the limited C API.
#if (PY_VERSION_HEX < 0x03050200 && !defined(Py_SETREF)) && !defined(Py_LIMITED_API)
#define Py_SETREF(dst, src)                                     \
    do {                                                        \
        PyObject **_tmp_dst_ptr = _Py_CAST(PyObject**, &(dst)); \
        PyObject *_tmp_dst = (*_tmp_dst_ptr);                   \
        *_tmp_dst_ptr = _PyObject_CAST(src);                    \
        Py_DECREF(_tmp_dst);                                    \
    } while (0)

#define Py_XSETREF(dst, src)                                    \
    do {                                                        \
        PyObject **_tmp_dst_ptr = _Py_CAST(PyObject**, &(dst)); \
        PyObject *_tmp_dst = (*_tmp_dst_ptr);                   \
        *_tmp_dst_ptr = _PyObject_CAST(src);                    \
        Py_XDECREF(_tmp_dst);                                   \
    } while (0)
#endif


// bpo-43753 added Py_Is(), Py_IsNone(), Py_IsTrue() and Py_IsFalse()
// to Python 3.10.0b1.
#if PY_VERSION_HEX < 0x030A00B1 && !defined(Py_Is)
#  define Py_Is(x, y) ((x) == (y))
#endif
#if PY_VERSION_HEX < 0x030A00B1 && !defined(Py_IsNone)
#  define Py_IsNone(x) Py_Is(x, Py_None)
#endif
#if PY_VERSION_HEX < 0x030A00B1 && !defined(Py_IsTrue)
#  define Py_IsTrue(x) Py_Is(x, Py_True)
#endif
#if PY_VERSION_HEX < 0x030A00B1 && !defined(Py_IsFalse)
#  define Py_IsFalse(x) Py_Is(x, Py_False)
#endif


// bpo-39573 added Py_SET_TYPE() to Python 3.9.0a4
#if PY_VERSION_HEX < 0x030900A4 && !defined(Py_SET_TYPE)
PYCAPI_COMPAT_STATIC_INLINE(void)
_Py_SET_TYPE(PyObject *ob, PyTypeObject *type)
{
    ob->ob_type = type;
}
#define Py_SET_TYPE(ob, type) _Py_SET_TYPE(_PyObject_CAST(ob), type)
#endif


// bpo-39573 added Py_SET_SIZE() to Python 3.9.0a4
#if PY_VERSION_HEX < 0x030900A4 && !defined(Py_SET_SIZE)
PYCAPI_COMPAT_STATIC_INLINE(void)
_Py_SET_SIZE(PyVarObject *ob, Py_ssize_t size)
{
    ob->ob_size = size;
}
#define Py_SET_SIZE(ob, size) _Py_SET_SIZE((PyVarObject*)(ob), size)
#endif


// bpo-40421 added PyFrame_GetCode() to Python 3.9.0b1
#if PY_VERSION_HEX < 0x030900B1 || defined(PYPY_VERSION)
PYCAPI_COMPAT_STATIC_INLINE(PyCodeObject*)
PyFrame_GetCode(PyFrameObject *frame)
{
    assert(frame != _Py_NULL);
    assert(frame->f_code != _Py_NULL);
    return _Py_CAST(PyCodeObject*, Py_NewRef(frame->f_code));
}
#endif

PYCAPI_COMPAT_STATIC_INLINE(PyCodeObject*)
_PyFrame_GetCodeBorrow(PyFrameObject *frame)
{
    PyCodeObject *code = PyFrame_GetCode(frame);
    Py_DECREF(code);
    return code;
}


// bpo-40421 added PyFrame_GetBack() to Python 3.9.0b1
#if PY_VERSION_HEX < 0x030900B1 && !defined(PYPY_VERSION)
PYCAPI_COMPAT_STATIC_INLINE(PyFrameObject*)
PyFrame_GetBack(PyFrameObject *frame)
{
    assert(frame != _Py_NULL);
    return _Py_CAST(PyFrameObject*, Py_XNewRef(frame->f_back));
}
#endif

#if !defined(PYPY_VERSION)
PYCAPI_COMPAT_STATIC_INLINE(PyFrameObject*)
_PyFrame_GetBackBorrow(PyFrameObject *frame)
{
    PyFrameObject *back = PyFrame_GetBack(frame);
    Py_XDECREF(back);
    return back;
}
#endif


// bpo-40421 added PyFrame_GetLocals() to Python 3.11.0a7
#if PY_VERSION_HEX < 0x030B00A7 && !defined(PYPY_VERSION)
PYCAPI_COMPAT_STATIC_INLINE(PyObject*)
PyFrame_GetLocals(PyFrameObject *frame)
{
#if PY_VERSION_HEX >= 0x030400B1
    if (PyFrame_FastToLocalsWithError(frame) < 0) {
        return NULL;
    }
#else
    PyFrame_FastToLocals(frame);
#endif
    return Py_NewRef(frame->f_locals);
}
#endif


// bpo-40421 added PyFrame_GetGlobals() to Python 3.11.0a7
#if PY_VERSION_HEX < 0x030B00A7 && !defined(PYPY_VERSION)
PYCAPI_COMPAT_STATIC_INLINE(PyObject*)
PyFrame_GetGlobals(PyFrameObject *frame)
{
    return Py_NewRef(frame->f_globals);
}
#endif


// bpo-40421 added PyFrame_GetBuiltins() to Python 3.11.0a7
#if PY_VERSION_HEX < 0x030B00A7 && !defined(PYPY_VERSION)
PYCAPI_COMPAT_STATIC_INLINE(PyObject*)
PyFrame_GetBuiltins(PyFrameObject *frame)
{
    return Py_NewRef(frame->f_builtins);
}
#endif


// bpo-40421 added PyFrame_GetLasti() to Python 3.11.0b1
#if PY_VERSION_HEX < 0x030B00B1 && !defined(PYPY_VERSION)
PYCAPI_COMPAT_STATIC_INLINE(int)
PyFrame_GetLasti(PyFrameObject *frame)
{
#if PY_VERSION_HEX >= 0x030A00A7
    // bpo-27129: Since Python 3.10.0a7, f_lasti is an instruction offset,
    // not a bytes offset anymore. Python uses 16-bit "wordcode" (2 bytes)
    // instructions.
    if (frame->f_lasti < 0) {
        return -1;
    }
    return frame->f_lasti * 2;
#else
    return frame->f_lasti;
#endif
}
#endif


// gh-91248 added PyFrame_GetVar() to Python 3.12.0a2
#if PY_VERSION_HEX < 0x030C00A2 && !defined(PYPY_VERSION)
PYCAPI_COMPAT_STATIC_INLINE(PyObject*)
PyFrame_GetVar(PyFrameObject *frame, PyObject *name)
{
    PyObject *locals, *value;

    locals = PyFrame_GetLocals(frame);
    if (locals == NULL) {
        return NULL;
    }
#if PY_VERSION_HEX >= 0x03000000
    value = PyDict_GetItemWithError(locals, name);
#else
    value = PyDict_GetItem(locals, name);
#endif
    Py_DECREF(locals);

    if (value == NULL) {
        if (PyErr_Occurred()) {
            return NULL;
        }
#if PY_VERSION_HEX >= 0x03000000
        PyErr_Format(PyExc_NameError, "variable %R does not exist", name);
#else
        PyErr_SetString(PyExc_NameError, "variable does not exist");
#endif
        return NULL;
    }
    return Py_NewRef(value);
}
#endif


// gh-91248 added PyFrame_GetVarString() to Python 3.12.0a2
#if PY_VERSION_HEX < 0x030C00A2 && !defined(PYPY_VERSION)
PYCAPI_COMPAT_STATIC_INLINE(PyObject*)
PyFrame_GetVarString(PyFrameObject *frame, const char *name)
{
    PyObject *name_obj, *value;
    name_obj = PyUnicode_FromString(name);
    if (name_obj == NULL) {
        return NULL;
    }
    value = PyFrame_GetVar(frame, name_obj);
    Py_DECREF(name_obj);
    return value;
}
#endif


// bpo-39947 added PyThreadState_GetInterpreter() to Python 3.9.0a5
#if PY_VERSION_HEX < 0x030900A5 || defined(PYPY_VERSION)
PYCAPI_COMPAT_STATIC_INLINE(PyInterpreterState *)
PyThreadState_GetInterpreter(PyThreadState *tstate)
{
    assert(tstate != _Py_NULL);
    return tstate->interp;
}
#endif


// bpo-40429 added PyThreadState_GetFrame() to Python 3.9.0b1
#if PY_VERSION_HEX < 0x030900B1 && !defined(PYPY_VERSION)
PYCAPI_COMPAT_STATIC_INLINE(PyFrameObject*)
PyThreadState_GetFrame(PyThreadState *tstate)
{
    assert(tstate != _Py_NULL);
    return _Py_CAST(PyFrameObject *, Py_XNewRef(tstate->frame));
}
#endif

#if !defined(PYPY_VERSION)
PYCAPI_COMPAT_STATIC_INLINE(PyFrameObject*)
_PyThreadState_GetFrameBorrow(PyThreadState *tstate)
{
    PyFrameObject *frame = PyThreadState_GetFrame(tstate);
    Py_XDECREF(frame);
    return frame;
}
#endif


// bpo-39947 added PyInterpreterState_Get() to Python 3.9.0a5
#if PY_VERSION_HEX < 0x030900A5 || defined(PYPY_VERSION)
PYCAPI_COMPAT_STATIC_INLINE(PyInterpreterState*)
PyInterpreterState_Get(void)
{
    PyThreadState *tstate;
    PyInterpreterState *interp;

    tstate = PyThreadState_GET();
    if (tstate == _Py_NULL) {
        Py_FatalError("GIL released (tstate is NULL)");
    }
    interp = tstate->interp;
    if (interp == _Py_NULL) {
        Py_FatalError("no current interpreter");
    }
    return interp;
}
#endif


// bpo-39947 added PyInterpreterState_Get() to Python 3.9.0a6
#if 0x030700A1 <= PY_VERSION_HEX && PY_VERSION_HEX < 0x030900A6 && !defined(PYPY_VERSION)
PYCAPI_COMPAT_STATIC_INLINE(uint64_t)
PyThreadState_GetID(PyThreadState *tstate)
{
    assert(tstate != _Py_NULL);
    return tstate->id;
}
#endif

// bpo-43760 added PyThreadState_EnterTracing() to Python 3.11.0a2
#if PY_VERSION_HEX < 0x030B00A2 && !defined(PYPY_VERSION)
PYCAPI_COMPAT_STATIC_INLINE(void)
PyThreadState_EnterTracing(PyThreadState *tstate)
{
    tstate->tracing++;
#if PY_VERSION_HEX >= 0x030A00A1
    tstate->cframe->use_tracing = 0;
#else
    tstate->use_tracing = 0;
#endif
}
#endif

// bpo-43760 added PyThreadState_LeaveTracing() to Python 3.11.0a2
#if PY_VERSION_HEX < 0x030B00A2 && !defined(PYPY_VERSION)
PYCAPI_COMPAT_STATIC_INLINE(void)
PyThreadState_LeaveTracing(PyThreadState *tstate)
{
    int use_tracing = (tstate->c_tracefunc != _Py_NULL
                       || tstate->c_profilefunc != _Py_NULL);
    tstate->tracing--;
#if PY_VERSION_HEX >= 0x030A00A1
    tstate->cframe->use_tracing = use_tracing;
#else
    tstate->use_tracing = use_tracing;
#endif
}
#endif


// bpo-37194 added PyObject_CallNoArgs() to Python 3.9.0a1
// PyObject_CallNoArgs() added to PyPy 3.9.16-v7.3.11
#if !defined(PyObject_CallNoArgs) && PY_VERSION_HEX < 0x030900A1
PYCAPI_COMPAT_STATIC_INLINE(PyObject*)
PyObject_CallNoArgs(PyObject *func)
{
    return PyObject_CallFunctionObjArgs(func, NULL);
}
#endif


// bpo-39245 made PyObject_CallOneArg() public (previously called
// _PyObject_CallOneArg) in Python 3.9.0a4
// PyObject_CallOneArg() added to PyPy 3.9.16-v7.3.11
#if !defined(PyObject_CallOneArg) && PY_VERSION_HEX < 0x030900A4
PYCAPI_COMPAT_STATIC_INLINE(PyObject*)
PyObject_CallOneArg(PyObject *func, PyObject *arg)
{
    return PyObject_CallFunctionObjArgs(func, arg, NULL);
}
#endif


// bpo-1635741 added PyModule_AddObjectRef() to Python 3.10.0a3
#if PY_VERSION_HEX < 0x030A00A3
PYCAPI_COMPAT_STATIC_INLINE(int)
PyModule_AddObjectRef(PyObject *module, const char *name, PyObject *value)
{
    int res;
    Py_XINCREF(value);
    res = PyModule_AddObject(module, name, value);
    if (res < 0) {
        Py_XDECREF(value);
    }
    return res;
}
#endif


// bpo-40024 added PyModule_AddType() to Python 3.9.0a5
#if PY_VERSION_HEX < 0x030900A5
PYCAPI_COMPAT_STATIC_INLINE(int)
PyModule_AddType(PyObject *module, PyTypeObject *type)
{
    const char *name, *dot;

    if (PyType_Ready(type) < 0) {
        return -1;
    }

    // inline _PyType_Name()
    name = type->tp_name;
    assert(name != _Py_NULL);
    dot = strrchr(name, '.');
    if (dot != _Py_NULL) {
        name = dot + 1;
    }

    return PyModule_AddObjectRef(module, name, _PyObject_CAST(type));
}
#endif


// bpo-40241 added PyObject_GC_IsTracked() to Python 3.9.0a6.
// bpo-4688 added _PyObject_GC_IS_TRACKED() to Python 2.7.0a2.
#if PY_VERSION_HEX < 0x030900A6 && !defined(PYPY_VERSION)
PYCAPI_COMPAT_STATIC_INLINE(int)
PyObject_GC_IsTracked(PyObject* obj)
{
    return (PyObject_IS_GC(obj) && _PyObject_GC_IS_TRACKED(obj));
}
#endif

// bpo-40241 added PyObject_GC_IsFinalized() to Python 3.9.0a6.
// bpo-18112 added _PyGCHead_FINALIZED() to Python 3.4.0 final.
#if PY_VERSION_HEX < 0x030900A6 && PY_VERSION_HEX >= 0x030400F0 && !defined(PYPY_VERSION)
PYCAPI_COMPAT_STATIC_INLINE(int)
PyObject_GC_IsFinalized(PyObject *obj)
{
    PyGC_Head *gc = _Py_CAST(PyGC_Head*, obj) - 1;
    return (PyObject_IS_GC(obj) && _PyGCHead_FINALIZED(gc));
}
#endif


// bpo-39573 added Py_IS_TYPE() to Python 3.9.0a4
#if PY_VERSION_HEX < 0x030900A4 && !defined(Py_IS_TYPE)
PYCAPI_COMPAT_STATIC_INLINE(int)
_Py_IS_TYPE(PyObject *ob, PyTypeObject *type) {
    return Py_TYPE(ob) == type;
}
#define Py_IS_TYPE(ob, type) _Py_IS_TYPE(_PyObject_CAST(ob), type)
#endif


// bpo-46906 added PyFloat_Pack2() and PyFloat_Unpack2() to Python 3.11a7.
// bpo-11734 added _PyFloat_Pack2() and _PyFloat_Unpack2() to Python 3.6.0b1.
// Python 3.11a2 moved _PyFloat_Pack2() and _PyFloat_Unpack2() to the internal
// C API: Python 3.11a2-3.11a6 versions are not supported.
#if 0x030600B1 <= PY_VERSION_HEX && PY_VERSION_HEX <= 0x030B00A1 && !defined(PYPY_VERSION)
PYCAPI_COMPAT_STATIC_INLINE(int)
PyFloat_Pack2(double x, char *p, int le)
{ return _PyFloat_Pack2(x, (unsigned char*)p, le); }

PYCAPI_COMPAT_STATIC_INLINE(double)
PyFloat_Unpack2(const char *p, int le)
{ return _PyFloat_Unpack2((const unsigned char *)p, le); }
#endif


// bpo-46906 added PyFloat_Pack4(), PyFloat_Pack8(), PyFloat_Unpack4() and
// PyFloat_Unpack8() to Python 3.11a7.
// Python 3.11a2 moved _PyFloat_Pack4(), _PyFloat_Pack8(), _PyFloat_Unpack4()
// and _PyFloat_Unpack8() to the internal C API: Python 3.11a2-3.11a6 versions
// are not supported.
#if PY_VERSION_HEX <= 0x030B00A1 && !defined(PYPY_VERSION)
PYCAPI_COMPAT_STATIC_INLINE(int)
PyFloat_Pack4(double x, char *p, int le)
{ return _PyFloat_Pack4(x, (unsigned char*)p, le); }

PYCAPI_COMPAT_STATIC_INLINE(int)
PyFloat_Pack8(double x, char *p, int le)
{ return _PyFloat_Pack8(x, (unsigned char*)p, le); }

PYCAPI_COMPAT_STATIC_INLINE(double)
PyFloat_Unpack4(const char *p, int le)
{ return _PyFloat_Unpack4((const unsigned char *)p, le); }

PYCAPI_COMPAT_STATIC_INLINE(double)
PyFloat_Unpack8(const char *p, int le)
{ return _PyFloat_Unpack8((const unsigned char *)p, le); }
#endif


// gh-92154 added PyCode_GetCode() to Python 3.11.0b1
#if PY_VERSION_HEX < 0x030B00B1 && !defined(PYPY_VERSION)
PYCAPI_COMPAT_STATIC_INLINE(PyObject*)
PyCode_GetCode(PyCodeObject *code)
{
    return Py_NewRef(code->co_code);
}
#endif


// gh-95008 added PyCode_GetVarnames() to Python 3.11.0rc1
#if PY_VERSION_HEX < 0x030B00C1 && !defined(PYPY_VERSION)
PYCAPI_COMPAT_STATIC_INLINE(PyObject*)
PyCode_GetVarnames(PyCodeObject *code)
{
    return Py_NewRef(code->co_varnames);
}
#endif

// gh-95008 added PyCode_GetFreevars() to Python 3.11.0rc1
#if PY_VERSION_HEX < 0x030B00C1 && !defined(PYPY_VERSION)
PYCAPI_COMPAT_STATIC_INLINE(PyObject*)
PyCode_GetFreevars(PyCodeObject *code)
{
    return Py_NewRef(code->co_freevars);
}
#endif

// gh-95008 added PyCode_GetCellvars() to Python 3.11.0rc1
#if PY_VERSION_HEX < 0x030B00C1 && !defined(PYPY_VERSION)
PYCAPI_COMPAT_STATIC_INLINE(PyObject*)
PyCode_GetCellvars(PyCodeObject *code)
{
    return Py_NewRef(code->co_cellvars);
}
#endif


// Py_UNUSED() was added to Python 3.4.0b2.
#if PY_VERSION_HEX < 0x030400B2 && !defined(Py_UNUSED)
#  if defined(__GNUC__) || defined(__clang__)
#    define Py_UNUSED(name) _unused_ ## name __attribute__((unused))
#  else
#    define Py_UNUSED(name) _unused_ ## name
#  endif
#endif


// gh-105922 added PyImport_AddModuleRef() to Python 3.13.0a1
#if PY_VERSION_HEX < 0x030D00A0
PYCAPI_COMPAT_STATIC_INLINE(PyObject*)
PyImport_AddModuleRef(const char *name)
{
    return Py_XNewRef(PyImport_AddModule(name));
}
#endif


// gh-105927 added PyWeakref_GetRef() to Python 3.13.0a1
#if PY_VERSION_HEX < 0x030D0000
PYCAPI_COMPAT_STATIC_INLINE(int)
PyWeakref_GetRef(PyObject *ref, PyObject **pobj)
{
    PyObject *obj;
    if (ref != NULL && !PyWeakref_Check(ref)) {
        *pobj = NULL;
        PyErr_SetString(PyExc_TypeError, "expected a weakref");
        return -1;
    }
    obj = PyWeakref_GetObject(ref);
    if (obj == NULL) {
        // SystemError if ref is NULL
        *pobj = NULL;
        return -1;
    }
    if (obj == Py_None) {
        *pobj = NULL;
        return 0;
    }
    *pobj = Py_NewRef(obj);
    return (*pobj != NULL);
}
#endif


// bpo-36974 added PY_VECTORCALL_ARGUMENTS_OFFSET to Python 3.8b1
#ifndef PY_VECTORCALL_ARGUMENTS_OFFSET
#  define PY_VECTORCALL_ARGUMENTS_OFFSET (_Py_CAST(size_t, 1) << (8 * sizeof(size_t) - 1))
#endif

// bpo-36974 added PyVectorcall_NARGS() to Python 3.8b1
#if PY_VERSION_HEX < 0x030800B1
static inline Py_ssize_t
PyVectorcall_NARGS(size_t n)
{
    return n & ~PY_VECTORCALL_ARGUMENTS_OFFSET;
}
#endif


// gh-105922 added PyObject_Vectorcall() to Python 3.9.0a4
#if PY_VERSION_HEX < 0x030900A4
PYCAPI_COMPAT_STATIC_INLINE(PyObject*)
PyObject_Vectorcall(PyObject *callable, PyObject *const *args,
                     size_t nargsf, PyObject *kwnames)
{
#if PY_VERSION_HEX >= 0x030800B1 && !defined(PYPY_VERSION)
    // bpo-36974 added _PyObject_Vectorcall() to Python 3.8.0b1
    return _PyObject_Vectorcall(callable, args, nargsf, kwnames);
#else
    PyObject *posargs = NULL, *kwargs = NULL;
    PyObject *res;
    Py_ssize_t nposargs, nkwargs, i;

    if (nargsf != 0 && args == NULL) {
        PyErr_BadInternalCall();
        goto error;
    }
    if (kwnames != NULL && !PyTuple_Check(kwnames)) {
        PyErr_BadInternalCall();
        goto error;
    }

    nposargs = (Py_ssize_t)PyVectorcall_NARGS(nargsf);
    if (kwnames) {
        nkwargs = PyTuple_GET_SIZE(kwnames);
    }
    else {
        nkwargs = 0;
    }

    posargs = PyTuple_New(nposargs);
    if (posargs == NULL) {
        goto error;
    }
    if (nposargs) {
        for (i=0; i < nposargs; i++) {
            PyTuple_SET_ITEM(posargs, i, Py_NewRef(*args));
            args++;
        }
    }

    if (nkwargs) {
        kwargs = PyDict_New();
        if (kwargs == NULL) {
            goto error;
        }

        for (i = 0; i < nkwargs; i++) {
            PyObject *key = PyTuple_GET_ITEM(kwnames, i);
            PyObject *value = *args;
            args++;
            if (PyDict_SetItem(kwargs, key, value) < 0) {
                goto error;
            }
        }
    }
    else {
        kwargs = NULL;
    }

    res = PyObject_Call(callable, posargs, kwargs);
    Py_DECREF(posargs);
    Py_XDECREF(kwargs);
    return res;

error:
    Py_DECREF(posargs);
    Py_XDECREF(kwargs);
    return NULL;
#endif
}
#endif


#ifdef __cplusplus
}
#endif
#endif  // PYTHONCAPI_COMPAT
