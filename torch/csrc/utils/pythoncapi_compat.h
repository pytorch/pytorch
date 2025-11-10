// Header file providing new C API functions to old Python versions.
//
// File distributed under the Zero Clause BSD (0BSD) license.
// Copyright Contributors to the pythoncapi_compat project.
//
// Homepage:
// https://github.com/python/pythoncapi_compat
//
// Latest version:
// https://raw.githubusercontent.com/python/pythoncapi-compat/main/pythoncapi_compat.h
//
// SPDX-License-Identifier: 0BSD

#ifndef PYTHONCAPI_COMPAT
#define PYTHONCAPI_COMPAT

#ifdef __cplusplus
extern "C" {
#endif

#include <Python.h>
#include <stddef.h>               // offsetof()

// Python 3.11.0b4 added PyFrame_Back() to Python.h
#if PY_VERSION_HEX < 0x030b00B4 && !defined(PYPY_VERSION)
#  include "frameobject.h"        // PyFrameObject, PyFrame_GetBack()
#endif


#ifndef _Py_CAST
#  define _Py_CAST(type, expr) ((type)(expr))
#endif

// Static inline functions should use _Py_NULL rather than using directly NULL
// to prevent C++ compiler warnings. On C23 and newer and on C++11 and newer,
// _Py_NULL is defined as nullptr.
#ifndef _Py_NULL
#  if (defined (__STDC_VERSION__) && __STDC_VERSION__ > 201710L) \
          || (defined(__cplusplus) && __cplusplus >= 201103)
#    define _Py_NULL nullptr
#  else
#    define _Py_NULL NULL
#  endif
#endif

// Cast argument to PyObject* type.
#ifndef _PyObject_CAST
#  define _PyObject_CAST(op) _Py_CAST(PyObject*, op)
#endif

#ifndef Py_BUILD_ASSERT
#  define Py_BUILD_ASSERT(cond) \
        do { \
            (void)sizeof(char [1 - 2 * !(cond)]); \
        } while(0)
#endif


// bpo-42262 added Py_NewRef() to Python 3.10.0a3
#if PY_VERSION_HEX < 0x030A00A3 && !defined(Py_NewRef)
static inline PyObject* _Py_NewRef(PyObject *obj)
{
    Py_INCREF(obj);
    return obj;
}
#define Py_NewRef(obj) _Py_NewRef(_PyObject_CAST(obj))
#endif


// bpo-42262 added Py_XNewRef() to Python 3.10.0a3
#if PY_VERSION_HEX < 0x030A00A3 && !defined(Py_XNewRef)
static inline PyObject* _Py_XNewRef(PyObject *obj)
{
    Py_XINCREF(obj);
    return obj;
}
#define Py_XNewRef(obj) _Py_XNewRef(_PyObject_CAST(obj))
#endif


// bpo-39573 added Py_SET_REFCNT() to Python 3.9.0a4
#if PY_VERSION_HEX < 0x030900A4 && !defined(Py_SET_REFCNT)
static inline void _Py_SET_REFCNT(PyObject *ob, Py_ssize_t refcnt)
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
#if (PY_VERSION_HEX < 0x030A00B1 || defined(PYPY_VERSION)) && !defined(Py_IsTrue)
#  define Py_IsTrue(x) Py_Is(x, Py_True)
#endif
#if (PY_VERSION_HEX < 0x030A00B1 || defined(PYPY_VERSION)) && !defined(Py_IsFalse)
#  define Py_IsFalse(x) Py_Is(x, Py_False)
#endif


// bpo-39573 added Py_SET_TYPE() to Python 3.9.0a4
#if PY_VERSION_HEX < 0x030900A4 && !defined(Py_SET_TYPE)
static inline void _Py_SET_TYPE(PyObject *ob, PyTypeObject *type)
{
    ob->ob_type = type;
}
#define Py_SET_TYPE(ob, type) _Py_SET_TYPE(_PyObject_CAST(ob), type)
#endif


// bpo-39573 added Py_SET_SIZE() to Python 3.9.0a4
#if PY_VERSION_HEX < 0x030900A4 && !defined(Py_SET_SIZE)
static inline void _Py_SET_SIZE(PyVarObject *ob, Py_ssize_t size)
{
    ob->ob_size = size;
}
#define Py_SET_SIZE(ob, size) _Py_SET_SIZE((PyVarObject*)(ob), size)
#endif


// bpo-40421 added PyFrame_GetCode() to Python 3.9.0b1
#if PY_VERSION_HEX < 0x030900B1 || defined(PYPY_VERSION)
static inline PyCodeObject* PyFrame_GetCode(PyFrameObject *frame)
{
    assert(frame != _Py_NULL);
    assert(frame->f_code != _Py_NULL);
    return _Py_CAST(PyCodeObject*, Py_NewRef(frame->f_code));
}
#endif

static inline PyCodeObject* _PyFrame_GetCodeBorrow(PyFrameObject *frame)
{
    PyCodeObject *code = PyFrame_GetCode(frame);
    Py_DECREF(code);
    return code;
}


// bpo-40421 added PyFrame_GetBack() to Python 3.9.0b1
#if PY_VERSION_HEX < 0x030900B1 && !defined(PYPY_VERSION)
static inline PyFrameObject* PyFrame_GetBack(PyFrameObject *frame)
{
    assert(frame != _Py_NULL);
    return _Py_CAST(PyFrameObject*, Py_XNewRef(frame->f_back));
}
#endif

#if !defined(PYPY_VERSION)
static inline PyFrameObject* _PyFrame_GetBackBorrow(PyFrameObject *frame)
{
    PyFrameObject *back = PyFrame_GetBack(frame);
    Py_XDECREF(back);
    return back;
}
#endif


// bpo-40421 added PyFrame_GetLocals() to Python 3.11.0a7
#if PY_VERSION_HEX < 0x030B00A7 && !defined(PYPY_VERSION)
static inline PyObject* PyFrame_GetLocals(PyFrameObject *frame)
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
static inline PyObject* PyFrame_GetGlobals(PyFrameObject *frame)
{
    return Py_NewRef(frame->f_globals);
}
#endif


// bpo-40421 added PyFrame_GetBuiltins() to Python 3.11.0a7
#if PY_VERSION_HEX < 0x030B00A7 && !defined(PYPY_VERSION)
static inline PyObject* PyFrame_GetBuiltins(PyFrameObject *frame)
{
    return Py_NewRef(frame->f_builtins);
}
#endif


// bpo-40421 added PyFrame_GetLasti() to Python 3.11.0b1
#if PY_VERSION_HEX < 0x030B00B1 && !defined(PYPY_VERSION)
static inline int PyFrame_GetLasti(PyFrameObject *frame)
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
static inline PyObject* PyFrame_GetVar(PyFrameObject *frame, PyObject *name)
{
    PyObject *locals, *value;

    locals = PyFrame_GetLocals(frame);
    if (locals == NULL) {
        return NULL;
    }
#if PY_VERSION_HEX >= 0x03000000
    value = PyDict_GetItemWithError(locals, name);
#else
    value = _PyDict_GetItemWithError(locals, name);
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
static inline PyObject*
PyFrame_GetVarString(PyFrameObject *frame, const char *name)
{
    PyObject *name_obj, *value;
#if PY_VERSION_HEX >= 0x03000000
    name_obj = PyUnicode_FromString(name);
#else
    name_obj = PyString_FromString(name);
#endif
    if (name_obj == NULL) {
        return NULL;
    }
    value = PyFrame_GetVar(frame, name_obj);
    Py_DECREF(name_obj);
    return value;
}
#endif


// bpo-39947 added PyThreadState_GetInterpreter() to Python 3.9.0a5
#if PY_VERSION_HEX < 0x030900A5 || (defined(PYPY_VERSION) && PY_VERSION_HEX < 0x030B0000)
static inline PyInterpreterState *
PyThreadState_GetInterpreter(PyThreadState *tstate)
{
    assert(tstate != _Py_NULL);
    return tstate->interp;
}
#endif


// bpo-40429 added PyThreadState_GetFrame() to Python 3.9.0b1
#if PY_VERSION_HEX < 0x030900B1 && !defined(PYPY_VERSION)
static inline PyFrameObject* PyThreadState_GetFrame(PyThreadState *tstate)
{
    assert(tstate != _Py_NULL);
    return _Py_CAST(PyFrameObject *, Py_XNewRef(tstate->frame));
}
#endif

#if !defined(PYPY_VERSION)
static inline PyFrameObject*
_PyThreadState_GetFrameBorrow(PyThreadState *tstate)
{
    PyFrameObject *frame = PyThreadState_GetFrame(tstate);
    Py_XDECREF(frame);
    return frame;
}
#endif


// bpo-39947 added PyInterpreterState_Get() to Python 3.9.0a5
#if PY_VERSION_HEX < 0x030900A5 || defined(PYPY_VERSION)
static inline PyInterpreterState* PyInterpreterState_Get(void)
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
static inline uint64_t PyThreadState_GetID(PyThreadState *tstate)
{
    assert(tstate != _Py_NULL);
    return tstate->id;
}
#endif

// bpo-43760 added PyThreadState_EnterTracing() to Python 3.11.0a2
#if PY_VERSION_HEX < 0x030B00A2 && !defined(PYPY_VERSION)
static inline void PyThreadState_EnterTracing(PyThreadState *tstate)
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
static inline void PyThreadState_LeaveTracing(PyThreadState *tstate)
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
static inline PyObject* PyObject_CallNoArgs(PyObject *func)
{
    return PyObject_CallFunctionObjArgs(func, NULL);
}
#endif


// bpo-39245 made PyObject_CallOneArg() public (previously called
// _PyObject_CallOneArg) in Python 3.9.0a4
// PyObject_CallOneArg() added to PyPy 3.9.16-v7.3.11
#if !defined(PyObject_CallOneArg) && PY_VERSION_HEX < 0x030900A4
static inline PyObject* PyObject_CallOneArg(PyObject *func, PyObject *arg)
{
    return PyObject_CallFunctionObjArgs(func, arg, NULL);
}
#endif


// bpo-1635741 added PyModule_AddObjectRef() to Python 3.10.0a3
#if PY_VERSION_HEX < 0x030A00A3
static inline int
PyModule_AddObjectRef(PyObject *module, const char *name, PyObject *value)
{
    int res;

    if (!value && !PyErr_Occurred()) {
        // PyModule_AddObject() raises TypeError in this case
        PyErr_SetString(PyExc_SystemError,
                        "PyModule_AddObjectRef() must be called "
                        "with an exception raised if value is NULL");
        return -1;
    }

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
static inline int PyModule_AddType(PyObject *module, PyTypeObject *type)
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
static inline int PyObject_GC_IsTracked(PyObject* obj)
{
    return (PyObject_IS_GC(obj) && _PyObject_GC_IS_TRACKED(obj));
}
#endif

// bpo-40241 added PyObject_GC_IsFinalized() to Python 3.9.0a6.
// bpo-18112 added _PyGCHead_FINALIZED() to Python 3.4.0 final.
#if PY_VERSION_HEX < 0x030900A6 && PY_VERSION_HEX >= 0x030400F0 && !defined(PYPY_VERSION)
static inline int PyObject_GC_IsFinalized(PyObject *obj)
{
    PyGC_Head *gc = _Py_CAST(PyGC_Head*, obj) - 1;
    return (PyObject_IS_GC(obj) && _PyGCHead_FINALIZED(gc));
}
#endif


// bpo-39573 added Py_IS_TYPE() to Python 3.9.0a4
#if PY_VERSION_HEX < 0x030900A4 && !defined(Py_IS_TYPE)
static inline int _Py_IS_TYPE(PyObject *ob, PyTypeObject *type) {
    return Py_TYPE(ob) == type;
}
#define Py_IS_TYPE(ob, type) _Py_IS_TYPE(_PyObject_CAST(ob), type)
#endif


// bpo-46906 added PyFloat_Pack2() and PyFloat_Unpack2() to Python 3.11a7.
// bpo-11734 added _PyFloat_Pack2() and _PyFloat_Unpack2() to Python 3.6.0b1.
// Python 3.11a2 moved _PyFloat_Pack2() and _PyFloat_Unpack2() to the internal
// C API: Python 3.11a2-3.11a6 versions are not supported.
#if 0x030600B1 <= PY_VERSION_HEX && PY_VERSION_HEX <= 0x030B00A1 && !defined(PYPY_VERSION)
static inline int PyFloat_Pack2(double x, char *p, int le)
{ return _PyFloat_Pack2(x, (unsigned char*)p, le); }

static inline double PyFloat_Unpack2(const char *p, int le)
{ return _PyFloat_Unpack2((const unsigned char *)p, le); }
#endif


// bpo-46906 added PyFloat_Pack4(), PyFloat_Pack8(), PyFloat_Unpack4() and
// PyFloat_Unpack8() to Python 3.11a7.
// Python 3.11a2 moved _PyFloat_Pack4(), _PyFloat_Pack8(), _PyFloat_Unpack4()
// and _PyFloat_Unpack8() to the internal C API: Python 3.11a2-3.11a6 versions
// are not supported.
#if PY_VERSION_HEX <= 0x030B00A1 && !defined(PYPY_VERSION)
static inline int PyFloat_Pack4(double x, char *p, int le)
{ return _PyFloat_Pack4(x, (unsigned char*)p, le); }

static inline int PyFloat_Pack8(double x, char *p, int le)
{ return _PyFloat_Pack8(x, (unsigned char*)p, le); }

static inline double PyFloat_Unpack4(const char *p, int le)
{ return _PyFloat_Unpack4((const unsigned char *)p, le); }

static inline double PyFloat_Unpack8(const char *p, int le)
{ return _PyFloat_Unpack8((const unsigned char *)p, le); }
#endif


// gh-92154 added PyCode_GetCode() to Python 3.11.0b1
#if PY_VERSION_HEX < 0x030B00B1 && !defined(PYPY_VERSION)
static inline PyObject* PyCode_GetCode(PyCodeObject *code)
{
    return Py_NewRef(code->co_code);
}
#endif


// gh-95008 added PyCode_GetVarnames() to Python 3.11.0rc1
#if PY_VERSION_HEX < 0x030B00C1 && !defined(PYPY_VERSION)
static inline PyObject* PyCode_GetVarnames(PyCodeObject *code)
{
    return Py_NewRef(code->co_varnames);
}
#endif

// gh-95008 added PyCode_GetFreevars() to Python 3.11.0rc1
#if PY_VERSION_HEX < 0x030B00C1 && !defined(PYPY_VERSION)
static inline PyObject* PyCode_GetFreevars(PyCodeObject *code)
{
    return Py_NewRef(code->co_freevars);
}
#endif

// gh-95008 added PyCode_GetCellvars() to Python 3.11.0rc1
#if PY_VERSION_HEX < 0x030B00C1 && !defined(PYPY_VERSION)
static inline PyObject* PyCode_GetCellvars(PyCodeObject *code)
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
static inline PyObject* PyImport_AddModuleRef(const char *name)
{
    return Py_XNewRef(PyImport_AddModule(name));
}
#endif


// gh-105927 added PyWeakref_GetRef() to Python 3.13.0a1
#if PY_VERSION_HEX < 0x030D0000
static inline int PyWeakref_GetRef(PyObject *ref, PyObject **pobj)
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
    return 1;
}
#endif


// bpo-36974 added PY_VECTORCALL_ARGUMENTS_OFFSET to Python 3.8b1
#ifndef PY_VECTORCALL_ARGUMENTS_OFFSET
#  define PY_VECTORCALL_ARGUMENTS_OFFSET (_Py_CAST(size_t, 1) << (8 * sizeof(size_t) - 1))
#endif

// bpo-36974 added PyVectorcall_NARGS() to Python 3.8b1
#if PY_VERSION_HEX < 0x030800B1
static inline Py_ssize_t PyVectorcall_NARGS(size_t n)
{
    return n & ~PY_VECTORCALL_ARGUMENTS_OFFSET;
}
#endif


// gh-105922 added PyObject_Vectorcall() to Python 3.9.0a4
#if PY_VERSION_HEX < 0x030900A4
static inline PyObject*
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


// gh-106521 added PyObject_GetOptionalAttr() and
// PyObject_GetOptionalAttrString() to Python 3.13.0a1
#if PY_VERSION_HEX < 0x030D00A1
static inline int
PyObject_GetOptionalAttr(PyObject *obj, PyObject *attr_name, PyObject **result)
{
    // bpo-32571 added _PyObject_LookupAttr() to Python 3.7.0b1
#if PY_VERSION_HEX >= 0x030700B1 && !defined(PYPY_VERSION)
    return _PyObject_LookupAttr(obj, attr_name, result);
#else
    *result = PyObject_GetAttr(obj, attr_name);
    if (*result != NULL) {
        return 1;
    }
    if (!PyErr_Occurred()) {
        return 0;
    }
    if (PyErr_ExceptionMatches(PyExc_AttributeError)) {
        PyErr_Clear();
        return 0;
    }
    return -1;
#endif
}

static inline int
PyObject_GetOptionalAttrString(PyObject *obj, const char *attr_name, PyObject **result)
{
    PyObject *name_obj;
    int rc;
#if PY_VERSION_HEX >= 0x03000000
    name_obj = PyUnicode_FromString(attr_name);
#else
    name_obj = PyString_FromString(attr_name);
#endif
    if (name_obj == NULL) {
        *result = NULL;
        return -1;
    }
    rc = PyObject_GetOptionalAttr(obj, name_obj, result);
    Py_DECREF(name_obj);
    return rc;
}
#endif


// gh-106307 added PyObject_GetOptionalAttr() and
// PyMapping_GetOptionalItemString() to Python 3.13.0a1
#if PY_VERSION_HEX < 0x030D00A1
static inline int
PyMapping_GetOptionalItem(PyObject *obj, PyObject *key, PyObject **result)
{
    *result = PyObject_GetItem(obj, key);
    if (*result) {
        return 1;
    }
    if (!PyErr_ExceptionMatches(PyExc_KeyError)) {
        return -1;
    }
    PyErr_Clear();
    return 0;
}

static inline int
PyMapping_GetOptionalItemString(PyObject *obj, const char *key, PyObject **result)
{
    PyObject *key_obj;
    int rc;
#if PY_VERSION_HEX >= 0x03000000
    key_obj = PyUnicode_FromString(key);
#else
    key_obj = PyString_FromString(key);
#endif
    if (key_obj == NULL) {
        *result = NULL;
        return -1;
    }
    rc = PyMapping_GetOptionalItem(obj, key_obj, result);
    Py_DECREF(key_obj);
    return rc;
}
#endif

// gh-108511 added PyMapping_HasKeyWithError() and
// PyMapping_HasKeyStringWithError() to Python 3.13.0a1
#if PY_VERSION_HEX < 0x030D00A1
static inline int
PyMapping_HasKeyWithError(PyObject *obj, PyObject *key)
{
    PyObject *res;
    int rc = PyMapping_GetOptionalItem(obj, key, &res);
    Py_XDECREF(res);
    return rc;
}

static inline int
PyMapping_HasKeyStringWithError(PyObject *obj, const char *key)
{
    PyObject *res;
    int rc = PyMapping_GetOptionalItemString(obj, key, &res);
    Py_XDECREF(res);
    return rc;
}
#endif


// gh-108511 added PyObject_HasAttrWithError() and
// PyObject_HasAttrStringWithError() to Python 3.13.0a1
#if PY_VERSION_HEX < 0x030D00A1
static inline int
PyObject_HasAttrWithError(PyObject *obj, PyObject *attr)
{
    PyObject *res;
    int rc = PyObject_GetOptionalAttr(obj, attr, &res);
    Py_XDECREF(res);
    return rc;
}

static inline int
PyObject_HasAttrStringWithError(PyObject *obj, const char *attr)
{
    PyObject *res;
    int rc = PyObject_GetOptionalAttrString(obj, attr, &res);
    Py_XDECREF(res);
    return rc;
}
#endif


// gh-106004 added PyDict_GetItemRef() and PyDict_GetItemStringRef()
// to Python 3.13.0a1
#if PY_VERSION_HEX < 0x030D00A1
static inline int
PyDict_GetItemRef(PyObject *mp, PyObject *key, PyObject **result)
{
#if PY_VERSION_HEX >= 0x03000000
    PyObject *item = PyDict_GetItemWithError(mp, key);
#else
    PyObject *item = _PyDict_GetItemWithError(mp, key);
#endif
    if (item != NULL) {
        *result = Py_NewRef(item);
        return 1;  // found
    }
    if (!PyErr_Occurred()) {
        *result = NULL;
        return 0;  // not found
    }
    *result = NULL;
    return -1;
}

static inline int
PyDict_GetItemStringRef(PyObject *mp, const char *key, PyObject **result)
{
    int res;
#if PY_VERSION_HEX >= 0x03000000
    PyObject *key_obj = PyUnicode_FromString(key);
#else
    PyObject *key_obj = PyString_FromString(key);
#endif
    if (key_obj == NULL) {
        *result = NULL;
        return -1;
    }
    res = PyDict_GetItemRef(mp, key_obj, result);
    Py_DECREF(key_obj);
    return res;
}
#endif


// gh-106307 added PyModule_Add() to Python 3.13.0a1
#if PY_VERSION_HEX < 0x030D00A1
static inline int
PyModule_Add(PyObject *mod, const char *name, PyObject *value)
{
    int res = PyModule_AddObjectRef(mod, name, value);
    Py_XDECREF(value);
    return res;
}
#endif


// gh-108014 added Py_IsFinalizing() to Python 3.13.0a1
// bpo-1856 added _Py_Finalizing to Python 3.2.1b1.
// _Py_IsFinalizing() was added to PyPy 7.3.0.
#if (0x030201B1 <= PY_VERSION_HEX && PY_VERSION_HEX < 0x030D00A1) \
        && (!defined(PYPY_VERSION_NUM) || PYPY_VERSION_NUM >= 0x7030000)
static inline int Py_IsFinalizing(void)
{
#if PY_VERSION_HEX >= 0x030700A1
    // _Py_IsFinalizing() was added to Python 3.7.0a1.
    return _Py_IsFinalizing();
#else
    return (_Py_Finalizing != NULL);
#endif
}
#endif


// gh-108323 added PyDict_ContainsString() to Python 3.13.0a1
#if PY_VERSION_HEX < 0x030D00A1
static inline int PyDict_ContainsString(PyObject *op, const char *key)
{
    PyObject *key_obj = PyUnicode_FromString(key);
    if (key_obj == NULL) {
        return -1;
    }
    int res = PyDict_Contains(op, key_obj);
    Py_DECREF(key_obj);
    return res;
}
#endif


// gh-108445 added PyLong_AsInt() to Python 3.13.0a1
#if PY_VERSION_HEX < 0x030D00A1
static inline int PyLong_AsInt(PyObject *obj)
{
#ifdef PYPY_VERSION
    long value = PyLong_AsLong(obj);
    if (value == -1 && PyErr_Occurred()) {
        return -1;
    }
    if (value < (long)INT_MIN || (long)INT_MAX < value) {
        PyErr_SetString(PyExc_OverflowError,
                        "Python int too large to convert to C int");
        return -1;
    }
    return (int)value;
#else
    return _PyLong_AsInt(obj);
#endif
}
#endif


// gh-107073 added PyObject_VisitManagedDict() to Python 3.13.0a1
#if PY_VERSION_HEX < 0x030D00A1
static inline int
PyObject_VisitManagedDict(PyObject *obj, visitproc visit, void *arg)
{
    PyObject **dict = _PyObject_GetDictPtr(obj);
    if (dict == NULL || *dict == NULL) {
        return -1;
    }
    Py_VISIT(*dict);
    return 0;
}

static inline void
PyObject_ClearManagedDict(PyObject *obj)
{
    PyObject **dict = _PyObject_GetDictPtr(obj);
    if (dict == NULL || *dict == NULL) {
        return;
    }
    Py_CLEAR(*dict);
}
#endif

// gh-108867 added PyThreadState_GetUnchecked() to Python 3.13.0a1
// Python 3.5.2 added _PyThreadState_UncheckedGet().
#if PY_VERSION_HEX >= 0x03050200 && PY_VERSION_HEX < 0x030D00A1
static inline PyThreadState*
PyThreadState_GetUnchecked(void)
{
    return _PyThreadState_UncheckedGet();
}
#endif

// gh-110289 added PyUnicode_EqualToUTF8() and PyUnicode_EqualToUTF8AndSize()
// to Python 3.13.0a1
#if PY_VERSION_HEX < 0x030D00A1
static inline int
PyUnicode_EqualToUTF8AndSize(PyObject *unicode, const char *str, Py_ssize_t str_len)
{
    Py_ssize_t len;
    const void *utf8;
    PyObject *exc_type, *exc_value, *exc_tb;
    int res;

    // API cannot report errors so save/restore the exception
    PyErr_Fetch(&exc_type, &exc_value, &exc_tb);

    // Python 3.3.0a1 added PyUnicode_AsUTF8AndSize()
#if PY_VERSION_HEX >= 0x030300A1
    if (PyUnicode_IS_ASCII(unicode)) {
        utf8 = PyUnicode_DATA(unicode);
        len = PyUnicode_GET_LENGTH(unicode);
    }
    else {
        utf8 = PyUnicode_AsUTF8AndSize(unicode, &len);
        if (utf8 == NULL) {
            // Memory allocation failure. The API cannot report error,
            // so ignore the exception and return 0.
            res = 0;
            goto done;
        }
    }

    if (len != str_len) {
        res = 0;
        goto done;
    }
    res = (memcmp(utf8, str, (size_t)len) == 0);
#else
    PyObject *bytes = PyUnicode_AsUTF8String(unicode);
    if (bytes == NULL) {
        // Memory allocation failure. The API cannot report error,
        // so ignore the exception and return 0.
        res = 0;
        goto done;
    }

#if PY_VERSION_HEX >= 0x03000000
    len = PyBytes_GET_SIZE(bytes);
    utf8 = PyBytes_AS_STRING(bytes);
#else
    len = PyString_GET_SIZE(bytes);
    utf8 = PyString_AS_STRING(bytes);
#endif
    if (len != str_len) {
        Py_DECREF(bytes);
        res = 0;
        goto done;
    }

    res = (memcmp(utf8, str, (size_t)len) == 0);
    Py_DECREF(bytes);
#endif

done:
    PyErr_Restore(exc_type, exc_value, exc_tb);
    return res;
}

static inline int
PyUnicode_EqualToUTF8(PyObject *unicode, const char *str)
{
    return PyUnicode_EqualToUTF8AndSize(unicode, str, (Py_ssize_t)strlen(str));
}
#endif


// gh-111138 added PyList_Extend() and PyList_Clear() to Python 3.13.0a2
#if PY_VERSION_HEX < 0x030D00A2
static inline int
PyList_Extend(PyObject *list, PyObject *iterable)
{
    return PyList_SetSlice(list, PY_SSIZE_T_MAX, PY_SSIZE_T_MAX, iterable);
}

static inline int
PyList_Clear(PyObject *list)
{
    return PyList_SetSlice(list, 0, PY_SSIZE_T_MAX, NULL);
}
#endif

// gh-111262 added PyDict_Pop() and PyDict_PopString() to Python 3.13.0a2
#if PY_VERSION_HEX < 0x030D00A2
static inline int
PyDict_Pop(PyObject *dict, PyObject *key, PyObject **result)
{
    PyObject *value;

    if (!PyDict_Check(dict)) {
        PyErr_BadInternalCall();
        if (result) {
            *result = NULL;
        }
        return -1;
    }

    // bpo-16991 added _PyDict_Pop() to Python 3.5.0b2.
    // Python 3.6.0b3 changed _PyDict_Pop() first argument type to PyObject*.
    // Python 3.13.0a1 removed _PyDict_Pop().
#if defined(PYPY_VERSION) || PY_VERSION_HEX < 0x030500b2 || PY_VERSION_HEX >= 0x030D0000
    value = PyObject_CallMethod(dict, "pop", "O", key);
#elif PY_VERSION_HEX < 0x030600b3
    value = _PyDict_Pop(_Py_CAST(PyDictObject*, dict), key, NULL);
#else
    value = _PyDict_Pop(dict, key, NULL);
#endif
    if (value == NULL) {
        if (result) {
            *result = NULL;
        }
        if (PyErr_Occurred() && !PyErr_ExceptionMatches(PyExc_KeyError)) {
            return -1;
        }
        PyErr_Clear();
        return 0;
    }
    if (result) {
        *result = value;
    }
    else {
        Py_DECREF(value);
    }
    return 1;
}

static inline int
PyDict_PopString(PyObject *dict, const char *key, PyObject **result)
{
    PyObject *key_obj = PyUnicode_FromString(key);
    if (key_obj == NULL) {
        if (result != NULL) {
            *result = NULL;
        }
        return -1;
    }

    int res = PyDict_Pop(dict, key_obj, result);
    Py_DECREF(key_obj);
    return res;
}
#endif


#if PY_VERSION_HEX < 0x030200A4
// Python 3.2.0a4 added Py_hash_t type
typedef Py_ssize_t Py_hash_t;
#endif


// gh-111545 added Py_HashPointer() to Python 3.13.0a3
#if PY_VERSION_HEX < 0x030D00A3
static inline Py_hash_t Py_HashPointer(const void *ptr)
{
#if PY_VERSION_HEX >= 0x030900A4 && !defined(PYPY_VERSION)
    return _Py_HashPointer(ptr);
#else
    return _Py_HashPointer(_Py_CAST(void*, ptr));
#endif
}
#endif


// Python 3.13a4 added a PyTime API.
// Use the private API added to Python 3.5.
#if PY_VERSION_HEX < 0x030D00A4 && PY_VERSION_HEX  >= 0x03050000
typedef _PyTime_t PyTime_t;
#define PyTime_MIN _PyTime_MIN
#define PyTime_MAX _PyTime_MAX

static inline double PyTime_AsSecondsDouble(PyTime_t t)
{ return _PyTime_AsSecondsDouble(t); }

static inline int PyTime_Monotonic(PyTime_t *result)
{ return _PyTime_GetMonotonicClockWithInfo(result, NULL); }

static inline int PyTime_Time(PyTime_t *result)
{ return _PyTime_GetSystemClockWithInfo(result, NULL); }

static inline int PyTime_PerfCounter(PyTime_t *result)
{
#if PY_VERSION_HEX >= 0x03070000 && !defined(PYPY_VERSION)
    return _PyTime_GetPerfCounterWithInfo(result, NULL);
#elif PY_VERSION_HEX >= 0x03070000
    // Call time.perf_counter_ns() and convert Python int object to PyTime_t.
    // Cache time.perf_counter_ns() function for best performance.
    static PyObject *func = NULL;
    if (func == NULL) {
        PyObject *mod = PyImport_ImportModule("time");
        if (mod == NULL) {
            return -1;
        }

        func = PyObject_GetAttrString(mod, "perf_counter_ns");
        Py_DECREF(mod);
        if (func == NULL) {
            return -1;
        }
    }

    PyObject *res = PyObject_CallNoArgs(func);
    if (res == NULL) {
        return -1;
    }
    long long value = PyLong_AsLongLong(res);
    Py_DECREF(res);

    if (value == -1 && PyErr_Occurred()) {
        return -1;
    }

    Py_BUILD_ASSERT(sizeof(value) >= sizeof(PyTime_t));
    *result = (PyTime_t)value;
    return 0;
#else
    // Call time.perf_counter() and convert C double to PyTime_t.
    // Cache time.perf_counter() function for best performance.
    static PyObject *func = NULL;
    if (func == NULL) {
        PyObject *mod = PyImport_ImportModule("time");
        if (mod == NULL) {
            return -1;
        }

        func = PyObject_GetAttrString(mod, "perf_counter");
        Py_DECREF(mod);
        if (func == NULL) {
            return -1;
        }
    }

    PyObject *res = PyObject_CallNoArgs(func);
    if (res == NULL) {
        return -1;
    }
    double d = PyFloat_AsDouble(res);
    Py_DECREF(res);

    if (d == -1.0 && PyErr_Occurred()) {
        return -1;
    }

    // Avoid floor() to avoid having to link to libm
    *result = (PyTime_t)(d * 1e9);
    return 0;
#endif
}

#endif

// gh-111389 added hash constants to Python 3.13.0a5. These constants were
// added first as private macros to Python 3.4.0b1 and PyPy 7.3.8.
#if (!defined(PyHASH_BITS) \
     && ((!defined(PYPY_VERSION) && PY_VERSION_HEX >= 0x030400B1) \
         || (defined(PYPY_VERSION) && PY_VERSION_HEX >= 0x03070000 \
             && PYPY_VERSION_NUM >= 0x07030800)))
#  define PyHASH_BITS _PyHASH_BITS
#  define PyHASH_MODULUS _PyHASH_MODULUS
#  define PyHASH_INF _PyHASH_INF
#  define PyHASH_IMAG _PyHASH_IMAG
#endif


// gh-111545 added Py_GetConstant() and Py_GetConstantBorrowed()
// to Python 3.13.0a6
#if PY_VERSION_HEX < 0x030D00A6 && !defined(Py_CONSTANT_NONE)

#define Py_CONSTANT_NONE 0
#define Py_CONSTANT_FALSE 1
#define Py_CONSTANT_TRUE 2
#define Py_CONSTANT_ELLIPSIS 3
#define Py_CONSTANT_NOT_IMPLEMENTED 4
#define Py_CONSTANT_ZERO 5
#define Py_CONSTANT_ONE 6
#define Py_CONSTANT_EMPTY_STR 7
#define Py_CONSTANT_EMPTY_BYTES 8
#define Py_CONSTANT_EMPTY_TUPLE 9

static inline PyObject* Py_GetConstant(unsigned int constant_id)
{
    static PyObject* constants[Py_CONSTANT_EMPTY_TUPLE + 1] = {NULL};

    if (constants[Py_CONSTANT_NONE] == NULL) {
        constants[Py_CONSTANT_NONE] = Py_None;
        constants[Py_CONSTANT_FALSE] = Py_False;
        constants[Py_CONSTANT_TRUE] = Py_True;
        constants[Py_CONSTANT_ELLIPSIS] = Py_Ellipsis;
        constants[Py_CONSTANT_NOT_IMPLEMENTED] = Py_NotImplemented;

        constants[Py_CONSTANT_ZERO] = PyLong_FromLong(0);
        if (constants[Py_CONSTANT_ZERO] == NULL) {
            goto fatal_error;
        }

        constants[Py_CONSTANT_ONE] = PyLong_FromLong(1);
        if (constants[Py_CONSTANT_ONE] == NULL) {
            goto fatal_error;
        }

        constants[Py_CONSTANT_EMPTY_STR] = PyUnicode_FromStringAndSize("", 0);
        if (constants[Py_CONSTANT_EMPTY_STR] == NULL) {
            goto fatal_error;
        }

        constants[Py_CONSTANT_EMPTY_BYTES] = PyBytes_FromStringAndSize("", 0);
        if (constants[Py_CONSTANT_EMPTY_BYTES] == NULL) {
            goto fatal_error;
        }

        constants[Py_CONSTANT_EMPTY_TUPLE] = PyTuple_New(0);
        if (constants[Py_CONSTANT_EMPTY_TUPLE] == NULL) {
            goto fatal_error;
        }
        // goto dance to avoid compiler warnings about Py_FatalError()
        goto init_done;

fatal_error:
        // This case should never happen
        Py_FatalError("Py_GetConstant() failed to get constants");
    }

init_done:
    if (constant_id <= Py_CONSTANT_EMPTY_TUPLE) {
        return Py_NewRef(constants[constant_id]);
    }
    else {
        PyErr_BadInternalCall();
        return NULL;
    }
}

static inline PyObject* Py_GetConstantBorrowed(unsigned int constant_id)
{
    PyObject *obj = Py_GetConstant(constant_id);
    Py_XDECREF(obj);
    return obj;
}
#endif


// gh-114329 added PyList_GetItemRef() to Python 3.13.0a4
#if PY_VERSION_HEX < 0x030D00A4
static inline PyObject *
PyList_GetItemRef(PyObject *op, Py_ssize_t index)
{
    PyObject *item = PyList_GetItem(op, index);
    Py_XINCREF(item);
    return item;
}
#endif


// gh-114329 added PyList_GetItemRef() to Python 3.13.0a4
#if PY_VERSION_HEX < 0x030D00A4
static inline int
PyDict_SetDefaultRef(PyObject *d, PyObject *key, PyObject *default_value,
                     PyObject **result)
{
    PyObject *value;
    if (PyDict_GetItemRef(d, key, &value) < 0) {
        // get error
        if (result) {
            *result = NULL;
        }
        return -1;
    }
    if (value != NULL) {
        // present
        if (result) {
            *result = value;
        }
        else {
            Py_DECREF(value);
        }
        return 1;
    }

    // missing: set the item
    if (PyDict_SetItem(d, key, default_value) < 0) {
        // set error
        if (result) {
            *result = NULL;
        }
        return -1;
    }
    if (result) {
        *result = Py_NewRef(default_value);
    }
    return 0;
}
#endif

#if PY_VERSION_HEX < 0x030D00B3
#  define Py_BEGIN_CRITICAL_SECTION(op) {
#  define Py_END_CRITICAL_SECTION() }
#  define Py_BEGIN_CRITICAL_SECTION2(a, b) {
#  define Py_END_CRITICAL_SECTION2() }
#endif

#if PY_VERSION_HEX < 0x030E0000 && PY_VERSION_HEX >= 0x03060000 && !defined(PYPY_VERSION)
typedef struct PyUnicodeWriter PyUnicodeWriter;

static inline void PyUnicodeWriter_Discard(PyUnicodeWriter *writer)
{
    _PyUnicodeWriter_Dealloc((_PyUnicodeWriter*)writer);
    PyMem_Free(writer);
}

static inline PyUnicodeWriter* PyUnicodeWriter_Create(Py_ssize_t length)
{
    if (length < 0) {
        PyErr_SetString(PyExc_ValueError,
                        "length must be positive");
        return NULL;
    }

    const size_t size = sizeof(_PyUnicodeWriter);
    PyUnicodeWriter *pub_writer = (PyUnicodeWriter *)PyMem_Malloc(size);
    if (pub_writer == _Py_NULL) {
        PyErr_NoMemory();
        return _Py_NULL;
    }
    _PyUnicodeWriter *writer = (_PyUnicodeWriter *)pub_writer;

    _PyUnicodeWriter_Init(writer);
    if (_PyUnicodeWriter_Prepare(writer, length, 127) < 0) {
        PyUnicodeWriter_Discard(pub_writer);
        return NULL;
    }
    writer->overallocate = 1;
    return pub_writer;
}

static inline PyObject* PyUnicodeWriter_Finish(PyUnicodeWriter *writer)
{
    PyObject *str = _PyUnicodeWriter_Finish((_PyUnicodeWriter*)writer);
    assert(((_PyUnicodeWriter*)writer)->buffer == NULL);
    PyMem_Free(writer);
    return str;
}

static inline int
PyUnicodeWriter_WriteChar(PyUnicodeWriter *writer, Py_UCS4 ch)
{
    if (ch > 0x10ffff) {
        PyErr_SetString(PyExc_ValueError,
                        "character must be in range(0x110000)");
        return -1;
    }

    return _PyUnicodeWriter_WriteChar((_PyUnicodeWriter*)writer, ch);
}

static inline int
PyUnicodeWriter_WriteStr(PyUnicodeWriter *writer, PyObject *obj)
{
    PyObject *str = PyObject_Str(obj);
    if (str == NULL) {
        return -1;
    }

    int res = _PyUnicodeWriter_WriteStr((_PyUnicodeWriter*)writer, str);
    Py_DECREF(str);
    return res;
}

static inline int
PyUnicodeWriter_WriteRepr(PyUnicodeWriter *writer, PyObject *obj)
{
    PyObject *str = PyObject_Repr(obj);
    if (str == NULL) {
        return -1;
    }

    int res = _PyUnicodeWriter_WriteStr((_PyUnicodeWriter*)writer, str);
    Py_DECREF(str);
    return res;
}

static inline int
PyUnicodeWriter_WriteUTF8(PyUnicodeWriter *writer,
                          const char *str, Py_ssize_t size)
{
    if (size < 0) {
        size = (Py_ssize_t)strlen(str);
    }

    PyObject *str_obj = PyUnicode_FromStringAndSize(str, size);
    if (str_obj == _Py_NULL) {
        return -1;
    }

    int res = _PyUnicodeWriter_WriteStr((_PyUnicodeWriter*)writer, str_obj);
    Py_DECREF(str_obj);
    return res;
}

static inline int
PyUnicodeWriter_WriteASCII(PyUnicodeWriter *writer,
                           const char *str, Py_ssize_t size)
{
    if (size < 0) {
        size = (Py_ssize_t)strlen(str);
    }

    return _PyUnicodeWriter_WriteASCIIString((_PyUnicodeWriter*)writer,
                                             str, size);
}

static inline int
PyUnicodeWriter_WriteWideChar(PyUnicodeWriter *writer,
                              const wchar_t *str, Py_ssize_t size)
{
    if (size < 0) {
        size = (Py_ssize_t)wcslen(str);
    }

    PyObject *str_obj = PyUnicode_FromWideChar(str, size);
    if (str_obj == _Py_NULL) {
        return -1;
    }

    int res = _PyUnicodeWriter_WriteStr((_PyUnicodeWriter*)writer, str_obj);
    Py_DECREF(str_obj);
    return res;
}

static inline int
PyUnicodeWriter_WriteSubstring(PyUnicodeWriter *writer, PyObject *str,
                               Py_ssize_t start, Py_ssize_t end)
{
    if (!PyUnicode_Check(str)) {
        PyErr_Format(PyExc_TypeError, "expect str, not %s",
                     Py_TYPE(str)->tp_name);
        return -1;
    }
    if (start < 0 || start > end) {
        PyErr_Format(PyExc_ValueError, "invalid start argument");
        return -1;
    }
    if (end > PyUnicode_GET_LENGTH(str)) {
        PyErr_Format(PyExc_ValueError, "invalid end argument");
        return -1;
    }

    return _PyUnicodeWriter_WriteSubstring((_PyUnicodeWriter*)writer, str,
                                           start, end);
}

static inline int
PyUnicodeWriter_Format(PyUnicodeWriter *writer, const char *format, ...)
{
    va_list vargs;
    va_start(vargs, format);
    PyObject *str = PyUnicode_FromFormatV(format, vargs);
    va_end(vargs);
    if (str == _Py_NULL) {
        return -1;
    }

    int res = _PyUnicodeWriter_WriteStr((_PyUnicodeWriter*)writer, str);
    Py_DECREF(str);
    return res;
}
#endif  // PY_VERSION_HEX < 0x030E0000

// gh-116560 added PyLong_GetSign() to Python 3.14.0a0
#if PY_VERSION_HEX < 0x030E00A0
static inline int PyLong_GetSign(PyObject *obj, int *sign)
{
    if (!PyLong_Check(obj)) {
        PyErr_Format(PyExc_TypeError, "expect int, got %s", Py_TYPE(obj)->tp_name);
        return -1;
    }

    *sign = _PyLong_Sign(obj);
    return 0;
}
#endif

// gh-126061 added PyLong_IsPositive/Negative/Zero() to Python in 3.14.0a2
#if PY_VERSION_HEX < 0x030E00A2
static inline int PyLong_IsPositive(PyObject *obj)
{
    if (!PyLong_Check(obj)) {
        PyErr_Format(PyExc_TypeError, "expected int, got %s", Py_TYPE(obj)->tp_name);
        return -1;
    }
    return _PyLong_Sign(obj) == 1;
}

static inline int PyLong_IsNegative(PyObject *obj)
{
    if (!PyLong_Check(obj)) {
        PyErr_Format(PyExc_TypeError, "expected int, got %s", Py_TYPE(obj)->tp_name);
        return -1;
    }
    return _PyLong_Sign(obj) == -1;
}

static inline int PyLong_IsZero(PyObject *obj)
{
    if (!PyLong_Check(obj)) {
        PyErr_Format(PyExc_TypeError, "expected int, got %s", Py_TYPE(obj)->tp_name);
        return -1;
    }
    return _PyLong_Sign(obj) == 0;
}
#endif


// gh-124502 added PyUnicode_Equal() to Python 3.14.0a0
#if PY_VERSION_HEX < 0x030E00A0
static inline int PyUnicode_Equal(PyObject *str1, PyObject *str2)
{
    if (!PyUnicode_Check(str1)) {
        PyErr_Format(PyExc_TypeError, "first argument must be str, not %s",
                     Py_TYPE(str1)->tp_name);
        return -1;
    }
    if (!PyUnicode_Check(str2)) {
        PyErr_Format(PyExc_TypeError, "second argument must be str, not %s",
                     Py_TYPE(str2)->tp_name);
        return -1;
    }

#if PY_VERSION_HEX >= 0x030d0000 && !defined(PYPY_VERSION)
    PyAPI_FUNC(int) _PyUnicode_Equal(PyObject *str1, PyObject *str2);

    return _PyUnicode_Equal(str1, str2);
#elif PY_VERSION_HEX >= 0x03060000 && !defined(PYPY_VERSION)
    return _PyUnicode_EQ(str1, str2);
#elif PY_VERSION_HEX >= 0x03090000 && defined(PYPY_VERSION)
    return _PyUnicode_EQ(str1, str2);
#else
    return (PyUnicode_Compare(str1, str2) == 0);
#endif
}
#endif


// gh-121645 added PyBytes_Join() to Python 3.14.0a0
#if PY_VERSION_HEX < 0x030E00A0
static inline PyObject* PyBytes_Join(PyObject *sep, PyObject *iterable)
{
    return _PyBytes_Join(sep, iterable);
}
#endif


#if PY_VERSION_HEX < 0x030E00A0
static inline Py_hash_t Py_HashBuffer(const void *ptr, Py_ssize_t len)
{
#if PY_VERSION_HEX >= 0x03000000 && !defined(PYPY_VERSION)
    PyAPI_FUNC(Py_hash_t) _Py_HashBytes(const void *src, Py_ssize_t len);

    return _Py_HashBytes(ptr, len);
#else
    Py_hash_t hash;
    PyObject *bytes = PyBytes_FromStringAndSize((const char*)ptr, len);
    if (bytes == NULL) {
        return -1;
    }
    hash = PyObject_Hash(bytes);
    Py_DECREF(bytes);
    return hash;
#endif
}
#endif


#if PY_VERSION_HEX < 0x030E00A0
static inline int PyIter_NextItem(PyObject *iter, PyObject **item)
{
    iternextfunc tp_iternext;

    assert(iter != NULL);
    assert(item != NULL);

    tp_iternext = Py_TYPE(iter)->tp_iternext;
    if (tp_iternext == NULL) {
        *item = NULL;
        PyErr_Format(PyExc_TypeError, "expected an iterator, got '%s'",
                     Py_TYPE(iter)->tp_name);
        return -1;
    }

    if ((*item = tp_iternext(iter))) {
        return 1;
    }
    if (!PyErr_Occurred()) {
        return 0;
    }
    if (PyErr_ExceptionMatches(PyExc_StopIteration)) {
        PyErr_Clear();
        return 0;
    }
    return -1;
}
#endif


#if PY_VERSION_HEX < 0x030E00A0
static inline PyObject* PyLong_FromInt32(int32_t value)
{
    Py_BUILD_ASSERT(sizeof(long) >= 4);
    return PyLong_FromLong(value);
}

static inline PyObject* PyLong_FromInt64(int64_t value)
{
    Py_BUILD_ASSERT(sizeof(long long) >= 8);
    return PyLong_FromLongLong(value);
}

static inline PyObject* PyLong_FromUInt32(uint32_t value)
{
    Py_BUILD_ASSERT(sizeof(unsigned long) >= 4);
    return PyLong_FromUnsignedLong(value);
}

static inline PyObject* PyLong_FromUInt64(uint64_t value)
{
    Py_BUILD_ASSERT(sizeof(unsigned long long) >= 8);
    return PyLong_FromUnsignedLongLong(value);
}

static inline int PyLong_AsInt32(PyObject *obj, int32_t *pvalue)
{
    Py_BUILD_ASSERT(sizeof(int) == 4);
    int value = PyLong_AsInt(obj);
    if (value == -1 && PyErr_Occurred()) {
        return -1;
    }
    *pvalue = (int32_t)value;
    return 0;
}

static inline int PyLong_AsInt64(PyObject *obj, int64_t *pvalue)
{
    Py_BUILD_ASSERT(sizeof(long long) == 8);
    long long value = PyLong_AsLongLong(obj);
    if (value == -1 && PyErr_Occurred()) {
        return -1;
    }
    *pvalue = (int64_t)value;
    return 0;
}

static inline int PyLong_AsUInt32(PyObject *obj, uint32_t *pvalue)
{
    Py_BUILD_ASSERT(sizeof(long) >= 4);
    unsigned long value = PyLong_AsUnsignedLong(obj);
    if (value == (unsigned long)-1 && PyErr_Occurred()) {
        return -1;
    }
#if SIZEOF_LONG > 4
    if ((unsigned long)UINT32_MAX < value) {
        PyErr_SetString(PyExc_OverflowError,
                        "Python int too large to convert to C uint32_t");
        return -1;
    }
#endif
    *pvalue = (uint32_t)value;
    return 0;
}

static inline int PyLong_AsUInt64(PyObject *obj, uint64_t *pvalue)
{
    Py_BUILD_ASSERT(sizeof(long long) == 8);
    unsigned long long value = PyLong_AsUnsignedLongLong(obj);
    if (value == (unsigned long long)-1 && PyErr_Occurred()) {
        return -1;
    }
    *pvalue = (uint64_t)value;
    return 0;
}
#endif


// gh-102471 added import and export API for integers to 3.14.0a2.
#if PY_VERSION_HEX < 0x030E00A2 && PY_VERSION_HEX >= 0x03000000 && !defined(PYPY_VERSION)
// Helpers to access PyLongObject internals.
static inline void
_PyLong_SetSignAndDigitCount(PyLongObject *op, int sign, Py_ssize_t size)
{
#if PY_VERSION_HEX >= 0x030C0000
    op->long_value.lv_tag = (uintptr_t)(1 - sign) | ((uintptr_t)(size) << 3);
#elif PY_VERSION_HEX >= 0x030900A4
    Py_SET_SIZE(op, sign * size);
#else
    Py_SIZE(op) = sign * size;
#endif
}

static inline Py_ssize_t
_PyLong_DigitCount(const PyLongObject *op)
{
#if PY_VERSION_HEX >= 0x030C0000
    return (Py_ssize_t)(op->long_value.lv_tag >> 3);
#else
    return _PyLong_Sign((PyObject*)op) < 0 ? -Py_SIZE(op) : Py_SIZE(op);
#endif
}

static inline digit*
_PyLong_GetDigits(const PyLongObject *op)
{
#if PY_VERSION_HEX >= 0x030C0000
    return (digit*)(op->long_value.ob_digit);
#else
    return (digit*)(op->ob_digit);
#endif
}

typedef struct PyLongLayout {
    uint8_t bits_per_digit;
    uint8_t digit_size;
    int8_t digits_order;
    int8_t digit_endianness;
} PyLongLayout;

typedef struct PyLongExport {
    int64_t value;
    uint8_t negative;
    Py_ssize_t ndigits;
    const void *digits;
    Py_uintptr_t _reserved;
} PyLongExport;

typedef struct PyLongWriter PyLongWriter;

static inline const PyLongLayout*
PyLong_GetNativeLayout(void)
{
    static const PyLongLayout PyLong_LAYOUT = {
        PyLong_SHIFT,
        sizeof(digit),
        -1,  // least significant first
        PY_LITTLE_ENDIAN ? -1 : 1,
    };

    return &PyLong_LAYOUT;
}

static inline int
PyLong_Export(PyObject *obj, PyLongExport *export_long)
{
    if (!PyLong_Check(obj)) {
        memset(export_long, 0, sizeof(*export_long));
        PyErr_Format(PyExc_TypeError, "expected int, got %s",
                     Py_TYPE(obj)->tp_name);
        return -1;
    }

    // Fast-path: try to convert to a int64_t
    PyLongObject *self = (PyLongObject*)obj;
    int overflow;
#if SIZEOF_LONG == 8
    long value = PyLong_AsLongAndOverflow(obj, &overflow);
#else
    // Windows has 32-bit long, so use 64-bit long long instead
    long long value = PyLong_AsLongLongAndOverflow(obj, &overflow);
#endif
    Py_BUILD_ASSERT(sizeof(value) == sizeof(int64_t));
    // the function cannot fail since obj is a PyLongObject
    assert(!(value == -1 && PyErr_Occurred()));

    if (!overflow) {
        export_long->value = value;
        export_long->negative = 0;
        export_long->ndigits = 0;
        export_long->digits = 0;
        export_long->_reserved = 0;
    }
    else {
        export_long->value = 0;
        export_long->negative = _PyLong_Sign(obj) < 0;
        export_long->ndigits = _PyLong_DigitCount(self);
        if (export_long->ndigits == 0) {
            export_long->ndigits = 1;
        }
        export_long->digits = _PyLong_GetDigits(self);
        export_long->_reserved = (Py_uintptr_t)Py_NewRef(obj);
    }
    return 0;
}

static inline void
PyLong_FreeExport(PyLongExport *export_long)
{
    PyObject *obj = (PyObject*)export_long->_reserved;

    if (obj) {
        export_long->_reserved = 0;
        Py_DECREF(obj);
    }
}

static inline PyLongWriter*
PyLongWriter_Create(int negative, Py_ssize_t ndigits, void **digits)
{
    if (ndigits <= 0) {
        PyErr_SetString(PyExc_ValueError, "ndigits must be positive");
        return NULL;
    }
    assert(digits != NULL);

    PyLongObject *obj = _PyLong_New(ndigits);
    if (obj == NULL) {
        return NULL;
    }
    _PyLong_SetSignAndDigitCount(obj, negative?-1:1, ndigits);

    *digits = _PyLong_GetDigits(obj);
    return (PyLongWriter*)obj;
}

static inline void
PyLongWriter_Discard(PyLongWriter *writer)
{
    PyLongObject *obj = (PyLongObject *)writer;

    assert(Py_REFCNT(obj) == 1);
    Py_DECREF(obj);
}

static inline PyObject*
PyLongWriter_Finish(PyLongWriter *writer)
{
    PyObject *obj = (PyObject *)writer;
    PyLongObject *self = (PyLongObject*)obj;
    Py_ssize_t j = _PyLong_DigitCount(self);
    Py_ssize_t i = j;
    int sign = _PyLong_Sign(obj);

    assert(Py_REFCNT(obj) == 1);

    // Normalize and get singleton if possible
    while (i > 0 && _PyLong_GetDigits(self)[i-1] == 0) {
        --i;
    }
    if (i != j) {
        if (i == 0) {
            sign = 0;
        }
        _PyLong_SetSignAndDigitCount(self, sign, i);
    }
    if (i <= 1) {
        long val = sign * (long)(_PyLong_GetDigits(self)[0]);
        Py_DECREF(obj);
        return PyLong_FromLong(val);
    }

    return obj;
}
#endif


#if PY_VERSION_HEX < 0x030C00A3
#  define Py_T_SHORT      0
#  define Py_T_INT        1
#  define Py_T_LONG       2
#  define Py_T_FLOAT      3
#  define Py_T_DOUBLE     4
#  define Py_T_STRING     5
#  define _Py_T_OBJECT    6
#  define Py_T_CHAR       7
#  define Py_T_BYTE       8
#  define Py_T_UBYTE      9
#  define Py_T_USHORT     10
#  define Py_T_UINT       11
#  define Py_T_ULONG      12
#  define Py_T_STRING_INPLACE  13
#  define Py_T_BOOL       14
#  define Py_T_OBJECT_EX  16
#  define Py_T_LONGLONG   17
#  define Py_T_ULONGLONG  18
#  define Py_T_PYSSIZET   19

#  if PY_VERSION_HEX >= 0x03000000 && !defined(PYPY_VERSION)
#    define _Py_T_NONE      20
#  endif

#  define Py_READONLY            1
#  define Py_AUDIT_READ          2
#  define _Py_WRITE_RESTRICTED   4
#endif


// gh-127350 added Py_fopen() and Py_fclose() to Python 3.14a4
#if PY_VERSION_HEX < 0x030E00A4
static inline FILE* Py_fopen(PyObject *path, const char *mode)
{
#if 0x030400A2 <= PY_VERSION_HEX && !defined(PYPY_VERSION)
    PyAPI_FUNC(FILE*) _Py_fopen_obj(PyObject *path, const char *mode);

    return _Py_fopen_obj(path, mode);
#else
    FILE *f;
    PyObject *bytes;
#if PY_VERSION_HEX >= 0x03000000
    if (!PyUnicode_FSConverter(path, &bytes)) {
        return NULL;
    }
#else
    if (!PyString_Check(path)) {
        PyErr_SetString(PyExc_TypeError, "except str");
        return NULL;
    }
    bytes = Py_NewRef(path);
#endif
    const char *path_bytes = PyBytes_AS_STRING(bytes);

    f = fopen(path_bytes, mode);
    Py_DECREF(bytes);

    if (f == NULL) {
        PyErr_SetFromErrnoWithFilenameObject(PyExc_OSError, path);
        return NULL;
    }
    return f;
#endif
}

static inline int Py_fclose(FILE *file)
{
    return fclose(file);
}
#endif


#if 0x03080000 <= PY_VERSION_HEX && PY_VERSION_HEX < 0x030E0000 && !defined(PYPY_VERSION)
PyAPI_FUNC(const PyConfig*) _Py_GetConfig(void);

static inline PyObject*
PyConfig_Get(const char *name)
{
    typedef enum {
        _PyConfig_MEMBER_INT,
        _PyConfig_MEMBER_UINT,
        _PyConfig_MEMBER_ULONG,
        _PyConfig_MEMBER_BOOL,
        _PyConfig_MEMBER_WSTR,
        _PyConfig_MEMBER_WSTR_OPT,
        _PyConfig_MEMBER_WSTR_LIST,
    } PyConfigMemberType;

    typedef struct {
        const char *name;
        size_t offset;
        PyConfigMemberType type;
        const char *sys_attr;
    } PyConfigSpec;

#define PYTHONCAPI_COMPAT_SPEC(MEMBER, TYPE, sys_attr) \
    {#MEMBER, offsetof(PyConfig, MEMBER), \
     _PyConfig_MEMBER_##TYPE, sys_attr}

    static const PyConfigSpec config_spec[] = {
        PYTHONCAPI_COMPAT_SPEC(argv, WSTR_LIST, "argv"),
        PYTHONCAPI_COMPAT_SPEC(base_exec_prefix, WSTR_OPT, "base_exec_prefix"),
        PYTHONCAPI_COMPAT_SPEC(base_executable, WSTR_OPT, "_base_executable"),
        PYTHONCAPI_COMPAT_SPEC(base_prefix, WSTR_OPT, "base_prefix"),
        PYTHONCAPI_COMPAT_SPEC(bytes_warning, UINT, _Py_NULL),
        PYTHONCAPI_COMPAT_SPEC(exec_prefix, WSTR_OPT, "exec_prefix"),
        PYTHONCAPI_COMPAT_SPEC(executable, WSTR_OPT, "executable"),
        PYTHONCAPI_COMPAT_SPEC(inspect, BOOL, _Py_NULL),
#if 0x030C0000 <= PY_VERSION_HEX
        PYTHONCAPI_COMPAT_SPEC(int_max_str_digits, UINT, _Py_NULL),
#endif
        PYTHONCAPI_COMPAT_SPEC(interactive, BOOL, _Py_NULL),
        PYTHONCAPI_COMPAT_SPEC(module_search_paths, WSTR_LIST, "path"),
        PYTHONCAPI_COMPAT_SPEC(optimization_level, UINT, _Py_NULL),
        PYTHONCAPI_COMPAT_SPEC(parser_debug, BOOL, _Py_NULL),
#if 0x03090000 <= PY_VERSION_HEX
        PYTHONCAPI_COMPAT_SPEC(platlibdir, WSTR, "platlibdir"),
#endif
        PYTHONCAPI_COMPAT_SPEC(prefix, WSTR_OPT, "prefix"),
        PYTHONCAPI_COMPAT_SPEC(pycache_prefix, WSTR_OPT, "pycache_prefix"),
        PYTHONCAPI_COMPAT_SPEC(quiet, BOOL, _Py_NULL),
#if 0x030B0000 <= PY_VERSION_HEX
        PYTHONCAPI_COMPAT_SPEC(stdlib_dir, WSTR_OPT, "_stdlib_dir"),
#endif
        PYTHONCAPI_COMPAT_SPEC(use_environment, BOOL, _Py_NULL),
        PYTHONCAPI_COMPAT_SPEC(verbose, UINT, _Py_NULL),
        PYTHONCAPI_COMPAT_SPEC(warnoptions, WSTR_LIST, "warnoptions"),
        PYTHONCAPI_COMPAT_SPEC(write_bytecode, BOOL, _Py_NULL),
        PYTHONCAPI_COMPAT_SPEC(xoptions, WSTR_LIST, "_xoptions"),
        PYTHONCAPI_COMPAT_SPEC(buffered_stdio, BOOL, _Py_NULL),
        PYTHONCAPI_COMPAT_SPEC(check_hash_pycs_mode, WSTR, _Py_NULL),
#if 0x030B0000 <= PY_VERSION_HEX
        PYTHONCAPI_COMPAT_SPEC(code_debug_ranges, BOOL, _Py_NULL),
#endif
        PYTHONCAPI_COMPAT_SPEC(configure_c_stdio, BOOL, _Py_NULL),
#if 0x030D0000 <= PY_VERSION_HEX
        PYTHONCAPI_COMPAT_SPEC(cpu_count, INT, _Py_NULL),
#endif
        PYTHONCAPI_COMPAT_SPEC(dev_mode, BOOL, _Py_NULL),
        PYTHONCAPI_COMPAT_SPEC(dump_refs, BOOL, _Py_NULL),
#if 0x030B0000 <= PY_VERSION_HEX
        PYTHONCAPI_COMPAT_SPEC(dump_refs_file, WSTR_OPT, _Py_NULL),
#endif
#ifdef Py_GIL_DISABLED
        PYTHONCAPI_COMPAT_SPEC(enable_gil, INT, _Py_NULL),
#endif
        PYTHONCAPI_COMPAT_SPEC(faulthandler, BOOL, _Py_NULL),
        PYTHONCAPI_COMPAT_SPEC(filesystem_encoding, WSTR, _Py_NULL),
        PYTHONCAPI_COMPAT_SPEC(filesystem_errors, WSTR, _Py_NULL),
        PYTHONCAPI_COMPAT_SPEC(hash_seed, ULONG, _Py_NULL),
        PYTHONCAPI_COMPAT_SPEC(home, WSTR_OPT, _Py_NULL),
        PYTHONCAPI_COMPAT_SPEC(import_time, BOOL, _Py_NULL),
        PYTHONCAPI_COMPAT_SPEC(install_signal_handlers, BOOL, _Py_NULL),
        PYTHONCAPI_COMPAT_SPEC(isolated, BOOL, _Py_NULL),
#ifdef MS_WINDOWS
        PYTHONCAPI_COMPAT_SPEC(legacy_windows_stdio, BOOL, _Py_NULL),
#endif
        PYTHONCAPI_COMPAT_SPEC(malloc_stats, BOOL, _Py_NULL),
#if 0x030A0000 <= PY_VERSION_HEX
        PYTHONCAPI_COMPAT_SPEC(orig_argv, WSTR_LIST, "orig_argv"),
#endif
        PYTHONCAPI_COMPAT_SPEC(parse_argv, BOOL, _Py_NULL),
        PYTHONCAPI_COMPAT_SPEC(pathconfig_warnings, BOOL, _Py_NULL),
#if 0x030C0000 <= PY_VERSION_HEX
        PYTHONCAPI_COMPAT_SPEC(perf_profiling, UINT, _Py_NULL),
#endif
        PYTHONCAPI_COMPAT_SPEC(program_name, WSTR, _Py_NULL),
        PYTHONCAPI_COMPAT_SPEC(run_command, WSTR_OPT, _Py_NULL),
        PYTHONCAPI_COMPAT_SPEC(run_filename, WSTR_OPT, _Py_NULL),
        PYTHONCAPI_COMPAT_SPEC(run_module, WSTR_OPT, _Py_NULL),
#if 0x030B0000 <= PY_VERSION_HEX
        PYTHONCAPI_COMPAT_SPEC(safe_path, BOOL, _Py_NULL),
#endif
        PYTHONCAPI_COMPAT_SPEC(show_ref_count, BOOL, _Py_NULL),
        PYTHONCAPI_COMPAT_SPEC(site_import, BOOL, _Py_NULL),
        PYTHONCAPI_COMPAT_SPEC(skip_source_first_line, BOOL, _Py_NULL),
        PYTHONCAPI_COMPAT_SPEC(stdio_encoding, WSTR, _Py_NULL),
        PYTHONCAPI_COMPAT_SPEC(stdio_errors, WSTR, _Py_NULL),
        PYTHONCAPI_COMPAT_SPEC(tracemalloc, UINT, _Py_NULL),
#if 0x030B0000 <= PY_VERSION_HEX
        PYTHONCAPI_COMPAT_SPEC(use_frozen_modules, BOOL, _Py_NULL),
#endif
        PYTHONCAPI_COMPAT_SPEC(use_hash_seed, BOOL, _Py_NULL),
        PYTHONCAPI_COMPAT_SPEC(user_site_directory, BOOL, _Py_NULL),
#if 0x030A0000 <= PY_VERSION_HEX
        PYTHONCAPI_COMPAT_SPEC(warn_default_encoding, BOOL, _Py_NULL),
#endif
    };

#undef PYTHONCAPI_COMPAT_SPEC

    const PyConfigSpec *spec;
    int found = 0;
    for (size_t i=0; i < sizeof(config_spec) / sizeof(config_spec[0]); i++) {
        spec = &config_spec[i];
        if (strcmp(spec->name, name) == 0) {
            found = 1;
            break;
        }
    }
    if (found) {
        if (spec->sys_attr != NULL) {
            PyObject *value = PySys_GetObject(spec->sys_attr);
            if (value == NULL) {
                PyErr_Format(PyExc_RuntimeError, "lost sys.%s", spec->sys_attr);
                return NULL;
            }
            return Py_NewRef(value);
        }

        const PyConfig *config = _Py_GetConfig();
        void *member = (char *)config + spec->offset;
        switch (spec->type) {
        case _PyConfig_MEMBER_INT:
        case _PyConfig_MEMBER_UINT:
        {
            int value = *(int *)member;
            return PyLong_FromLong(value);
        }
        case _PyConfig_MEMBER_BOOL:
        {
            int value = *(int *)member;
            return PyBool_FromLong(value != 0);
        }
        case _PyConfig_MEMBER_ULONG:
        {
            unsigned long value = *(unsigned long *)member;
            return PyLong_FromUnsignedLong(value);
        }
        case _PyConfig_MEMBER_WSTR:
        case _PyConfig_MEMBER_WSTR_OPT:
        {
            wchar_t *wstr = *(wchar_t **)member;
            if (wstr != NULL) {
                return PyUnicode_FromWideChar(wstr, -1);
            }
            else {
                return Py_NewRef(Py_None);
            }
        }
        case _PyConfig_MEMBER_WSTR_LIST:
        {
            const PyWideStringList *list = (const PyWideStringList *)member;
            PyObject *tuple = PyTuple_New(list->length);
            if (tuple == NULL) {
                return NULL;
            }

            for (Py_ssize_t i = 0; i < list->length; i++) {
                PyObject *item = PyUnicode_FromWideChar(list->items[i], -1);
                if (item == NULL) {
                    Py_DECREF(tuple);
                    return NULL;
                }
                PyTuple_SET_ITEM(tuple, i, item);
            }
            return tuple;
        }
        default:
            Py_UNREACHABLE();
        }
    }

    PyErr_Format(PyExc_ValueError, "unknown config option name: %s", name);
    return NULL;
}

static inline int
PyConfig_GetInt(const char *name, int *value)
{
    PyObject *obj = PyConfig_Get(name);
    if (obj == NULL) {
        return -1;
    }

    if (!PyLong_Check(obj)) {
        Py_DECREF(obj);
        PyErr_Format(PyExc_TypeError, "config option %s is not an int", name);
        return -1;
    }

    int as_int = PyLong_AsInt(obj);
    Py_DECREF(obj);
    if (as_int == -1 && PyErr_Occurred()) {
        PyErr_Format(PyExc_OverflowError,
                     "config option %s value does not fit into a C int", name);
        return -1;
    }

    *value = as_int;
    return 0;
}
#endif  // PY_VERSION_HEX > 0x03090000 && !defined(PYPY_VERSION)

// gh-133144 added PyUnstable_Object_IsUniquelyReferenced() to Python 3.14.0b1.
// Adapted from  _PyObject_IsUniquelyReferenced() implementation.
#if PY_VERSION_HEX < 0x030E00B0
static inline int PyUnstable_Object_IsUniquelyReferenced(PyObject *obj)
{
#if !defined(Py_GIL_DISABLED)
    return Py_REFCNT(obj) == 1;
#else
    // NOTE: the entire ob_ref_shared field must be zero, including flags, to
    // ensure that other threads cannot concurrently create new references to
    // this object.
    return (_Py_IsOwnedByCurrentThread(obj) &&
            _Py_atomic_load_uint32_relaxed(&obj->ob_ref_local) == 1 &&
            _Py_atomic_load_ssize_relaxed(&obj->ob_ref_shared) == 0);
#endif
}
#endif

// gh-128926 added PyUnstable_TryIncRef() and PyUnstable_EnableTryIncRef() to
// Python 3.14.0a5. Adapted from _Py_TryIncref() and _PyObject_SetMaybeWeakref().
#if PY_VERSION_HEX < 0x030E00A5
static inline int PyUnstable_TryIncRef(PyObject *op)
{
#ifndef Py_GIL_DISABLED
    if (Py_REFCNT(op) > 0) {
        Py_INCREF(op);
        return 1;
    }
    return 0;
#else
    // _Py_TryIncrefFast()
    uint32_t local = _Py_atomic_load_uint32_relaxed(&op->ob_ref_local);
    local += 1;
    if (local == 0) {
        // immortal
        return 1;
    }
    if (_Py_IsOwnedByCurrentThread(op)) {
        _Py_INCREF_STAT_INC();
        _Py_atomic_store_uint32_relaxed(&op->ob_ref_local, local);
#ifdef Py_REF_DEBUG
        _Py_INCREF_IncRefTotal();
#endif
        return 1;
    }

    // _Py_TryIncRefShared()
    Py_ssize_t shared = _Py_atomic_load_ssize_relaxed(&op->ob_ref_shared);
    for (;;) {
        // If the shared refcount is zero and the object is either merged
        // or may not have weak references, then we cannot incref it.
        if (shared == 0 || shared == _Py_REF_MERGED) {
            return 0;
        }

        if (_Py_atomic_compare_exchange_ssize(
                &op->ob_ref_shared,
                &shared,
                shared + (1 << _Py_REF_SHARED_SHIFT))) {
#ifdef Py_REF_DEBUG
            _Py_INCREF_IncRefTotal();
#endif
            _Py_INCREF_STAT_INC();
            return 1;
        }
    }
#endif
}

static inline void PyUnstable_EnableTryIncRef(PyObject *op)
{
#ifdef Py_GIL_DISABLED
    // _PyObject_SetMaybeWeakref()
    if (_Py_IsImmortal(op)) {
        return;
    }
    for (;;) {
        Py_ssize_t shared = _Py_atomic_load_ssize_relaxed(&op->ob_ref_shared);
        if ((shared & _Py_REF_SHARED_FLAG_MASK) != 0) {
            // Nothing to do if it's in WEAKREFS, QUEUED, or MERGED states.
            return;
        }
        if (_Py_atomic_compare_exchange_ssize(
                &op->ob_ref_shared, &shared, shared | _Py_REF_MAYBE_WEAKREF)) {
            return;
        }
    }
#else
    (void)op;  // unused argument
#endif
}
#endif


#if PY_VERSION_HEX < 0x030F0000
static inline PyObject*
PySys_GetAttrString(const char *name)
{
#if PY_VERSION_HEX >= 0x03000000
    PyObject *value = Py_XNewRef(PySys_GetObject(name));
#else
    PyObject *value = Py_XNewRef(PySys_GetObject((char*)name));
#endif
    if (value != NULL) {
        return value;
    }
    if (!PyErr_Occurred()) {
        PyErr_Format(PyExc_RuntimeError, "lost sys.%s", name);
    }
    return NULL;
}

static inline PyObject*
PySys_GetAttr(PyObject *name)
{
#if PY_VERSION_HEX >= 0x03000000
    const char *name_str = PyUnicode_AsUTF8(name);
#else
    const char *name_str = PyString_AsString(name);
#endif
    if (name_str == NULL) {
        return NULL;
    }

    return PySys_GetAttrString(name_str);
}

static inline int
PySys_GetOptionalAttrString(const char *name, PyObject **value)
{
#if PY_VERSION_HEX >= 0x03000000
    *value = Py_XNewRef(PySys_GetObject(name));
#else
    *value = Py_XNewRef(PySys_GetObject((char*)name));
#endif
    if (*value != NULL) {
        return 1;
    }
    return 0;
}

static inline int
PySys_GetOptionalAttr(PyObject *name, PyObject **value)
{
#if PY_VERSION_HEX >= 0x03000000
    const char *name_str = PyUnicode_AsUTF8(name);
#else
    const char *name_str = PyString_AsString(name);
#endif
    if (name_str == NULL) {
        *value = NULL;
        return -1;
    }

    return PySys_GetOptionalAttrString(name_str, value);
}
#endif  // PY_VERSION_HEX < 0x030F00A1


#if PY_VERSION_HEX < 0x030F00A1
typedef struct PyBytesWriter {
    char small_buffer[256];
    PyObject *obj;
    Py_ssize_t size;
} PyBytesWriter;

static inline Py_ssize_t
_PyBytesWriter_GetAllocated(PyBytesWriter *writer)
{
    if (writer->obj == NULL) {
        return sizeof(writer->small_buffer);
    }
    else {
        return PyBytes_GET_SIZE(writer->obj);
    }
}


static inline int
_PyBytesWriter_Resize_impl(PyBytesWriter *writer, Py_ssize_t size,
                           int resize)
{
    int overallocate = resize;
    assert(size >= 0);

    if (size <= _PyBytesWriter_GetAllocated(writer)) {
        return 0;
    }

    if (overallocate) {
#ifdef MS_WINDOWS
        /* On Windows, overallocate by 50% is the best factor */
        if (size <= (PY_SSIZE_T_MAX - size / 2)) {
            size += size / 2;
        }
#else
        /* On Linux, overallocate by 25% is the best factor */
        if (size <= (PY_SSIZE_T_MAX - size / 4)) {
            size += size / 4;
        }
#endif
    }

    if (writer->obj != NULL) {
        if (_PyBytes_Resize(&writer->obj, size)) {
            return -1;
        }
        assert(writer->obj != NULL);
    }
    else {
        writer->obj = PyBytes_FromStringAndSize(NULL, size);
        if (writer->obj == NULL) {
            return -1;
        }

        if (resize) {
            assert((size_t)size > sizeof(writer->small_buffer));
            memcpy(PyBytes_AS_STRING(writer->obj),
                   writer->small_buffer,
                   sizeof(writer->small_buffer));
        }
    }
    return 0;
}

static inline void*
PyBytesWriter_GetData(PyBytesWriter *writer)
{
    if (writer->obj == NULL) {
        return writer->small_buffer;
    }
    else {
        return PyBytes_AS_STRING(writer->obj);
    }
}

static inline Py_ssize_t
PyBytesWriter_GetSize(PyBytesWriter *writer)
{
    return writer->size;
}

static inline void
PyBytesWriter_Discard(PyBytesWriter *writer)
{
    if (writer == NULL) {
        return;
    }

    Py_XDECREF(writer->obj);
    PyMem_Free(writer);
}

static inline PyBytesWriter*
PyBytesWriter_Create(Py_ssize_t size)
{
    if (size < 0) {
        PyErr_SetString(PyExc_ValueError, "size must be >= 0");
        return NULL;
    }

    PyBytesWriter *writer = (PyBytesWriter*)PyMem_Malloc(sizeof(PyBytesWriter));
    if (writer == NULL) {
        PyErr_NoMemory();
        return NULL;
    }

    writer->obj = NULL;
    writer->size = 0;

    if (size >= 1) {
        if (_PyBytesWriter_Resize_impl(writer, size, 0) < 0) {
            PyBytesWriter_Discard(writer);
            return NULL;
        }
        writer->size = size;
    }
    return writer;
}

static inline PyObject*
PyBytesWriter_FinishWithSize(PyBytesWriter *writer, Py_ssize_t size)
{
    PyObject *result;
    if (size == 0) {
        result = PyBytes_FromStringAndSize("", 0);
    }
    else if (writer->obj != NULL) {
        if (size != PyBytes_GET_SIZE(writer->obj)) {
            if (_PyBytes_Resize(&writer->obj, size)) {
                goto error;
            }
        }
        result = writer->obj;
        writer->obj = NULL;
    }
    else {
        result = PyBytes_FromStringAndSize(writer->small_buffer, size);
    }
    PyBytesWriter_Discard(writer);
    return result;

error:
    PyBytesWriter_Discard(writer);
    return NULL;
}

static inline PyObject*
PyBytesWriter_Finish(PyBytesWriter *writer)
{
    return PyBytesWriter_FinishWithSize(writer, writer->size);
}

static inline PyObject*
PyBytesWriter_FinishWithPointer(PyBytesWriter *writer, void *buf)
{
    Py_ssize_t size = (char*)buf - (char*)PyBytesWriter_GetData(writer);
    if (size < 0 || size > _PyBytesWriter_GetAllocated(writer)) {
        PyBytesWriter_Discard(writer);
        PyErr_SetString(PyExc_ValueError, "invalid end pointer");
        return NULL;
    }

    return PyBytesWriter_FinishWithSize(writer, size);
}

static inline int
PyBytesWriter_Resize(PyBytesWriter *writer, Py_ssize_t size)
{
    if (size < 0) {
        PyErr_SetString(PyExc_ValueError, "size must be >= 0");
        return -1;
    }
    if (_PyBytesWriter_Resize_impl(writer, size, 1) < 0) {
        return -1;
    }
    writer->size = size;
    return 0;
}

static inline int
PyBytesWriter_Grow(PyBytesWriter *writer, Py_ssize_t size)
{
    if (size < 0 && writer->size + size < 0) {
        PyErr_SetString(PyExc_ValueError, "invalid size");
        return -1;
    }
    if (size > PY_SSIZE_T_MAX - writer->size) {
        PyErr_NoMemory();
        return -1;
    }
    size = writer->size + size;

    if (_PyBytesWriter_Resize_impl(writer, size, 1) < 0) {
        return -1;
    }
    writer->size = size;
    return 0;
}

static inline void*
PyBytesWriter_GrowAndUpdatePointer(PyBytesWriter *writer,
                                   Py_ssize_t size, void *buf)
{
    Py_ssize_t pos = (char*)buf - (char*)PyBytesWriter_GetData(writer);
    if (PyBytesWriter_Grow(writer, size) < 0) {
        return NULL;
    }
    return (char*)PyBytesWriter_GetData(writer) + pos;
}

static inline int
PyBytesWriter_WriteBytes(PyBytesWriter *writer,
                         const void *bytes, Py_ssize_t size)
{
    if (size < 0) {
        size_t len = strlen((const char*)bytes);
        if (len > (size_t)PY_SSIZE_T_MAX) {
            PyErr_NoMemory();
            return -1;
        }
        size = (Py_ssize_t)len;
    }

    Py_ssize_t pos = writer->size;
    if (PyBytesWriter_Grow(writer, size) < 0) {
        return -1;
    }
    char *buf = (char*)PyBytesWriter_GetData(writer);
    memcpy(buf + pos, bytes, (size_t)size);
    return 0;
}

static inline int
PyBytesWriter_Format(PyBytesWriter *writer, const char *format, ...)
                     Py_GCC_ATTRIBUTE((format(printf, 2, 3)));

static inline int
PyBytesWriter_Format(PyBytesWriter *writer, const char *format, ...)
{
    va_list vargs;
    va_start(vargs, format);
    PyObject *str = PyBytes_FromFormatV(format, vargs);
    va_end(vargs);

    if (str == NULL) {
        return -1;
    }
    int res = PyBytesWriter_WriteBytes(writer,
                                       PyBytes_AS_STRING(str),
                                       PyBytes_GET_SIZE(str));
    Py_DECREF(str);
    return res;
}
#endif  // PY_VERSION_HEX < 0x030F00A1


#if PY_VERSION_HEX < 0x030F00A1
static inline PyObject*
PyTuple_FromArray(PyObject *const *array, Py_ssize_t size)
{
    PyObject *tuple = PyTuple_New(size);
    if (tuple == NULL) {
        return NULL;
    }
    for (Py_ssize_t i=0; i < size; i++) {
        PyObject *item = array[i];
        PyTuple_SET_ITEM(tuple, i, Py_NewRef(item));
    }
    return tuple;
}
#endif


#if PY_VERSION_HEX < 0x030F00A1
static inline Py_hash_t
PyUnstable_Unicode_GET_CACHED_HASH(PyObject *op)
{
#ifdef PYPY_VERSION
    (void)op;  // unused argument
    return -1;
#elif PY_VERSION_HEX >= 0x03000000
    return ((PyASCIIObject*)op)->hash;
#else
    return ((PyUnicodeObject*)op)->hash;
#endif
}
#endif


#ifdef __cplusplus
}
#endif
#endif  // PYTHONCAPI_COMPAT
