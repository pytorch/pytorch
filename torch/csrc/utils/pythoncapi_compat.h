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
#if (defined (__STDC_VERSION__) && __STDC_VERSION__ > 201710L) \
        || (defined(__cplusplus) && __cplusplus >= 201103)
#  define _Py_NULL nullptr
#else
#  define _Py_NULL NULL
#endif

// Cast argument to PyObject* type.
#ifndef _PyObject_CAST
#  define _PyObject_CAST(op) _Py_CAST(PyObject*, op)
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

static inline PyCodeObject* _PyFrame_GetCodeBorrow(PyFrameObject *frame)
{
    PyCodeObject *code = PyFrame_GetCode(frame);
    Py_DECREF(code);
    return code;
}


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


#if !defined(PYPY_VERSION)
static inline PyFrameObject*
_PyThreadState_GetFrameBorrow(PyThreadState *tstate)
{
    PyFrameObject *frame = PyThreadState_GetFrame(tstate);
    Py_XDECREF(frame);
    return frame;
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
    return (*pobj != NULL);
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
    if (*dict == NULL) {
        return -1;
    }
    Py_VISIT(*dict);
    return 0;
}

static inline void
PyObject_ClearManagedDict(PyObject *obj)
{
    PyObject **dict = _PyObject_GetDictPtr(obj);
    if (*dict == NULL) {
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
// added first as private macros to Python 3.4.0b1 and PyPy 7.3.9.
#if (!defined(PyHASH_BITS) \
     && ((!defined(PYPY_VERSION) && PY_VERSION_HEX >= 0x030400B1) \
         || (defined(PYPY_VERSION) && PY_VERSION_HEX >= 0x03070000 \
             && PYPY_VERSION_NUM >= 0x07090000)))
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
        PyErr_Format(PyExc_TypeError, "expect str, not %T", str);
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


#ifdef __cplusplus
}
#endif
#endif  // PYTHONCAPI_COMPAT
