// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <utility>
#include <ostream>
#include <memory>

#define PY_BEGIN try {
#define PY_END(v) } catch(mpy::exception_set & err) { return (v); }

#if PY_VERSION_HEX < 0x03080000
    #define PY_VECTORCALL _PyObject_FastCallKeywords
#else
    #define PY_VECTORCALL _PyObject_Vectorcall
#endif

struct irange {
 public:
    irange(int64_t end)
    : irange(0, end, 1) {}
    irange(int64_t begin, int64_t end, int64_t step = 1)
    : begin_(begin), end_(end), step_(step) {}
    int64_t operator*() const {
        return begin_;
    }
    irange& operator++() {
        begin_ += step_;
        return *this;
    }
    bool operator!=(const irange& other) {
        return begin_ != other.begin_;
    }
    irange begin() {
        return *this;
    }
    irange end() {
        return irange {end_, end_, step_};
    }
 private:
    int64_t begin_;
    int64_t end_;
    int64_t step_;
};

namespace mpy {

struct exception_set {
};

struct object;
struct vector_args;

struct handle {
    handle(PyObject* ptr)
    : ptr_(ptr) {}
    handle() = default;


    PyObject* ptr() const {
        return ptr_;
    }
    object attr(const char* key);
    bool hasattr(const char* key);
    handle type() const {
        return (PyObject*) Py_TYPE(ptr());
    }

    template<typename... Args>
    object call(Args&&... args);
    object call_object(mpy::handle args);
    object call_object(mpy::handle args, mpy::handle kwargs);
    object call_vector(mpy::handle* begin, Py_ssize_t nargs, mpy::handle kwnames);
    object call_vector(vector_args args);
    bool operator==(handle rhs) {
        return ptr_ == rhs.ptr_;
    }

    static handle checked(PyObject* ptr) {
        if (!ptr) {
            throw exception_set();
        }
        return ptr;
    }

protected:
    PyObject* ptr_ = nullptr;
};


template<typename T>
struct obj;

template<typename T>
struct hdl : public handle {
    T* ptr() {
        return  (T*) handle::ptr();
    }
    T* operator->() {
        return ptr();
    }
    hdl(T* ptr)
    : hdl((PyObject*) ptr) {}
    hdl(const obj<T>& o)
    : hdl(o.ptr()) {}
private:
    hdl(handle h) : handle(h) {}
};

struct object : public handle {
    object() = default;
    object(const object& other)
    : handle(other.ptr_) {
        Py_XINCREF(ptr_);
    }
    object(object&& other) noexcept
    : handle(other.ptr_) {
        other.ptr_ = nullptr;
    }
    object& operator=(const object& other) {
        return *this = object(other);
    }
    object& operator=(object&& other) noexcept {
        PyObject* tmp = ptr_;
        ptr_ = other.ptr_;
        other.ptr_ = tmp;
        return *this;
    }
    ~object() {
        Py_XDECREF(ptr_);
    }
    static object steal(handle o) {
        return object(o.ptr());
    }
    static object checked_steal(handle o) {
        if (!o.ptr()) {
            throw exception_set();
        }
        return steal(o);
    }
    static object borrow(handle o) {
        Py_XINCREF(o.ptr());
        return steal(o);
    }
    PyObject* release() {
        auto tmp = ptr_;
        ptr_ = nullptr;
        return tmp;
    }
protected:
    explicit object(PyObject* ptr)
    : handle(ptr) {}
};

template<typename T>
struct obj : public object {
    obj() = default;
    obj(const obj& other)
    : object(other.ptr_) {
        Py_XINCREF(ptr_);
    }
    obj(obj&& other) noexcept
    : object(other.ptr_) {
        other.ptr_ = nullptr;
    }
    obj& operator=(const obj& other) {
        return *this = obj(other);
    }
    obj& operator=(obj&& other) noexcept {
        PyObject* tmp = ptr_;
        ptr_ = other.ptr_;
        other.ptr_ = tmp;
        return *this;
    }
    static obj steal(hdl<T> o) {
        return obj(o.ptr());
    }
    static obj checked_steal(hdl<T> o) {
        if (!o.ptr()) {
            throw exception_set();
        }
        return steal(o);
    }
    static obj borrow(hdl<T> o) {
        Py_XINCREF(o.ptr());
        return steal(o);
    }
    T* ptr() const {
        return (T*) object::ptr();
    }
    T* operator->() {
        return ptr();
    }
protected:
    explicit obj(T* ptr)
    : object((PyObject*)ptr) {}
};


static bool isinstance(handle h, handle c) {
    return PyObject_IsInstance(h.ptr(), c.ptr());
}

[[ noreturn ]] inline void raise_error(handle exception, const char *format, ...) {
    va_list args;
    va_start(args, format);
    PyErr_FormatV(exception.ptr(), format, args);
    va_end(args);
    throw exception_set();
}

template<typename T>
struct base {
    PyObject_HEAD
    PyObject* ptr() const {
        return (PyObject*) this;
    }
    static obj<T> alloc(PyTypeObject* type = nullptr) {
        if (!type) {
            type = &T::Type;
        }
        auto self = (T*) type->tp_alloc(type, 0);
        if (!self) {
            throw mpy::exception_set();
        }
        new (self) T;
        return obj<T>::steal(self);
    }
    template<typename ... Args>
    static obj<T> create(Args ... args) {
        auto self = alloc();
        self->init(std::forward<Args>(args)...);
        return self;
    }
    static bool check(handle v) {
        return isinstance(v, (PyObject*)&T::Type);
    }

    static hdl<T> unchecked_wrap(handle self_) {
        return hdl<T>((T*)self_.ptr());
    }
    static hdl<T> wrap(handle self_) {
        if (!check(self_)) {
            raise_error(PyExc_ValueError, "not an instance of %S", &T::Type);
        }
        return unchecked_wrap(self_);
    }

    static obj<T> unchecked_wrap(object self_) {
        return obj<T>::steal(unchecked_wrap(self_.release()));
    }
    static obj<T> wrap(object self_) {
        return obj<T>::steal(wrap(self_.release()));
    }

    static PyObject* new_stub(PyTypeObject *type, PyObject *args, PyObject *kwds) {
        PY_BEGIN
        return (PyObject*) alloc(type).release();
        PY_END(nullptr)
    }
    static void dealloc_stub(PyObject *self) {
        ((T*)self)->~T();
        Py_TYPE(self)->tp_free(self);
    }
    static void ready(mpy::handle mod, const char* name) {
        if (PyType_Ready(&T::Type)) {
            throw exception_set();
        }
        if(PyModule_AddObject(mod.ptr(), name, (PyObject*) &T::Type) < 0) {
            throw exception_set();
        }
    }
};

inline object handle::attr(const char* key) {
    return object::checked_steal(PyObject_GetAttrString(ptr(), key));
}

inline bool handle::hasattr(const char* key) {
    return PyObject_HasAttrString(ptr(), key);
}

inline object import(const char* module) {
    return object::checked_steal(PyImport_ImportModule(module));
}

template<typename... Args>
inline object handle::call(Args&&... args) {
    return object::checked_steal(PyObject_CallFunctionObjArgs(ptr_, args.ptr()..., nullptr));
}

inline object handle::call_object(mpy::handle args) {
    return object::checked_steal(PyObject_CallObject(ptr(), args.ptr()));
}


inline object handle::call_object(mpy::handle args, mpy::handle kwargs) {
    return object::checked_steal(PyObject_Call(ptr(), args.ptr(), kwargs.ptr()));
}

inline object handle::call_vector(mpy::handle* begin, Py_ssize_t nargs, mpy::handle kwnames) {
    return object::checked_steal(PY_VECTORCALL(ptr(), (PyObject*const*) begin, nargs, kwnames.ptr()));
}

struct tuple : public object {
    void set(int i, object v) {
        PyTuple_SET_ITEM(ptr_, i, v.release());
    }
    tuple(int size)
    : object(checked_steal(PyTuple_New(size))) {}
};

struct list : public object {
    void set(int i, object v) {
        PyList_SET_ITEM(ptr_, i, v.release());
    }
    list(int size)
    : object(checked_steal(PyList_New(size))) {}
};

namespace{
mpy::object unicode_from_format(const char* format, ...) {
    va_list args;
    va_start(args, format);
    auto r = PyUnicode_FromFormatV(format, args);
    va_end(args);
    return mpy::object::checked_steal(r);
}
mpy::object unicode_from_string(const char * str) {
    return mpy::object::checked_steal(PyUnicode_FromString(str));
}

mpy::object from_int(Py_ssize_t s) {
    return mpy::object::checked_steal(PyLong_FromSsize_t(s));
}
mpy::object from_bool(bool b) {
    return mpy::object::borrow(b ? Py_True : Py_False);
}

bool is_sequence(handle h) {
    return PySequence_Check(h.ptr());
}
}

struct sequence_view : public handle {
    sequence_view(handle h)
    : handle(h) {}
    Py_ssize_t size() const {
        auto r = PySequence_Size(ptr());
        if (r == -1 && PyErr_Occurred()) {
            throw mpy::exception_set();
        }
        return r;
    }
    irange enumerate() const {
        return irange(size());
    }
    static sequence_view wrap(handle h) {
        if (!is_sequence(h)) {
            raise_error(PyExc_ValueError, "expected a sequence");
        }
        return sequence_view(h);
    }
    mpy::object operator[](Py_ssize_t i) const {
        return mpy::object::checked_steal(PySequence_GetItem(ptr(), i));
    }
};

namespace {
mpy::object repr(handle h) {
    return mpy::object::checked_steal(PyObject_Repr(h.ptr()));
}

mpy::object str(handle h) {
    return mpy::object::checked_steal(PyObject_Str(h.ptr()));
}


bool is_int(handle h) {
    return PyLong_Check(h.ptr());
}

bool is_float(handle h) {
    return PyFloat_Check(h.ptr());
}

bool is_none(handle h) {
    return h.ptr() == Py_None;
}

bool is_bool(handle h) {
    return PyBool_Check(h.ptr());
}

Py_ssize_t to_int(handle h) {
    Py_ssize_t r = PyLong_AsSsize_t(h.ptr());
    if (r == -1 && PyErr_Occurred()) {
        throw mpy::exception_set();
    }
    return r;
}

double to_float(handle h) {
    double r = PyFloat_AsDouble(h.ptr());
    if (PyErr_Occurred()) {
        throw mpy::exception_set();
    }
    return r;
}

bool to_bool_unsafe(handle h) {
    return h.ptr() == Py_True;
}

bool to_bool(handle h) {
    return PyObject_IsTrue(h.ptr()) != 0;
}
}

struct slice_view {
    slice_view(handle h, Py_ssize_t size)  {
        if(PySlice_Unpack(h.ptr(), &start, &stop, &step) == -1) {
            throw mpy::exception_set();
        }
        slicelength = PySlice_AdjustIndices(size, &start, &stop, step);
    }
    Py_ssize_t start, stop, step, slicelength;
};

static bool is_slice(handle h) {
    return PySlice_Check(h.ptr());
}

inline std::ostream& operator<<(std::ostream& ss, handle h) {
    ss << PyUnicode_AsUTF8(str(h).ptr());
    return ss;
}

struct tuple_view : public handle {
    tuple_view() = default;
    tuple_view(handle h) : handle(h) {}

    Py_ssize_t size() const {
        return PyTuple_GET_SIZE(ptr());
    }

    irange enumerate() const {
        return irange(size());
    }

    handle operator[](Py_ssize_t i) {
        return PyTuple_GET_ITEM(ptr(), i);
    }

    static bool check(handle h) {
        return PyTuple_Check(h.ptr());
    }
};

struct list_view : public handle {
    list_view() = default;
    list_view(handle h) : handle(h) {}
    Py_ssize_t size() const {
        return PyList_GET_SIZE(ptr());
    }

    irange enumerate() const {
        return irange(size());
    }

    handle operator[](Py_ssize_t i) {
        return PyList_GET_ITEM(ptr(), i);
    }

    static bool check(handle h) {
        return PyList_Check(h.ptr());
    }
};

struct dict_view : public handle {
    dict_view() = default;
    dict_view(handle h) : handle(h) {}
    object keys() const {
        return mpy::object::checked_steal(PyDict_Keys(ptr()));
    }
    object values() const {
        return mpy::object::checked_steal(PyDict_Values(ptr()));
    }
    object items() const {
        return mpy::object::checked_steal(PyDict_Items(ptr()));
    }
    bool contains(handle k) const {
        return PyDict_Contains(ptr(), k.ptr());
    }
    handle operator[](handle k) {
        return mpy::handle::checked(PyDict_GetItem(ptr(), k.ptr()));
    }
    static bool check(handle h) {
        return PyDict_Check(h.ptr());
    }
    bool next(Py_ssize_t* pos, mpy::handle* key, mpy::handle* value) {
        PyObject *k = nullptr, *v = nullptr;
        auto r = PyDict_Next(ptr(), pos, &k, &v);
        *key = k;
        *value = v;
        return r;
    }
    void set(handle k, handle v) {
        if (-1 == PyDict_SetItem(ptr(), k.ptr(), v.ptr())) {
            throw exception_set();
        }
    }
};


struct kwnames_view : public handle {
    kwnames_view() = default;
    kwnames_view(handle h) : handle(h) {}

    Py_ssize_t size() const {
        return PyTuple_GET_SIZE(ptr());
    }

    irange enumerate() const {
        return irange(size());
    }

    const char* operator[](Py_ssize_t i) const {
        PyObject* obj = PyTuple_GET_ITEM(ptr(), i);
        return PyUnicode_AsUTF8(obj);
    }

    static bool check(handle h) {
        return PyTuple_Check(h.ptr());
    }
};

inline mpy::object funcname(mpy::handle func) {
    if (func.hasattr("__name__")) {
        return func.attr("__name__");
    } else {
        return mpy::str(func);
    }
}

struct vector_args {
    vector_args(PyObject *const *a,
                      Py_ssize_t n,
                      PyObject *k)
    : vector_args((mpy::handle*)a, n, k) {}
    vector_args(mpy::handle* a,
                    Py_ssize_t n,
                    mpy::handle k)
    : args((mpy::handle*)a), nargs(n), kwnames(k) {}
    mpy::handle* args;
    Py_ssize_t nargs;
    kwnames_view kwnames;

    mpy::handle* begin() {
        return args;
    }
    mpy::handle* end() {
        return args + size();
    }

    mpy::handle operator[](int64_t i) const {
        return args[i];
    }
    bool has_keywords() const {
        return kwnames.ptr();
    }
    irange enumerate_positional() {
        return irange(nargs);
    }
    irange enumerate_all() {
        return irange(size());
    }
    int64_t size() const {
        return nargs + (has_keywords() ? kwnames.size() : 0);
    }

    // bind a test function so this can be tested, first two args for required/kwonly, then return what was parsed...

    // provide write kwarg
    // don't provide a required arg
    // don't provide an optional arg
    // provide a kwarg that is the name of already provided positional
    // provide a kwonly argument positionally
    // provide keyword arguments in the wrong order
    // provide only keyword arguments
    void parse(const char * fname_cstr, std::initializer_list<const char*> names, std::initializer_list<mpy::handle*> values, int required, int kwonly=0) {
        auto error = [&]() {
            // rather than try to match the slower infrastructure with error messages exactly, once we have detected an error, just use that
            // infrastructure to format it and throw it

            // have to leak this, because python expects these to last
            const char** names_buf = new const char*[names.size() + 1];
            std::copy(names.begin(), names.end(), &names_buf[0]);
            names_buf[names.size()] = nullptr;

#if PY_VERSION_HEX < 0x03080000
            char* format_str = new char[names.size() + 3];
            int i = 0;
            char* format_it = format_str;
            for (auto it = names.begin(); it != names.end(); ++it, ++i) {
                if (i == required) {
                    *format_it++ = '|';
                }
                if (i == (int)names.size() - kwonly) {
                    *format_it++ = '$';
                }
                *format_it++ = 'O';
            }
            *format_it++ = '\0';
            _PyArg_Parser* _parser = new _PyArg_Parser{format_str, &names_buf[0], fname_cstr, 0};
            PyObject *dummy = NULL;
            _PyArg_ParseStackAndKeywords((PyObject*const*)args, nargs, kwnames.ptr(), _parser, &dummy, &dummy, &dummy, &dummy, &dummy);
#else
            _PyArg_Parser* _parser = new _PyArg_Parser{NULL, &names_buf[0], fname_cstr, 0};
            std::unique_ptr<PyObject*[]> buf(new PyObject*[names.size()]);
            _PyArg_UnpackKeywords((PyObject*const*)args, nargs, NULL, kwnames.ptr(), _parser, required, (Py_ssize_t)values.size() - kwonly, 0, &buf[0]);
#endif
            throw exception_set();
        };

        auto values_it = values.begin();
        auto names_it = names.begin();
        auto npositional = values.size() - kwonly;

        if (nargs > (Py_ssize_t)npositional) {
            // TOO MANY ARGUMENTS
            error();
        }
        for (auto i : irange(nargs)) {
            *(*values_it++) = args[i];
            ++names_it;
        }

        if (!kwnames.ptr()) {
            if (nargs < required) {
                // not enough positional arguments
                error();
            }
        } else {
            int consumed = 0;
            for (auto i : irange(nargs, values.size())) {
                bool success = i >= required;
                const char* target_name = *(names_it++);
                for (auto j : kwnames.enumerate()) {
                    if (!strcmp(target_name,kwnames[j])) {
                        *(*values_it) = args[nargs + j];
                        ++consumed;
                        success = true;
                        break;
                    }
                }
                ++values_it;
                if (!success) {
                    // REQUIRED ARGUMENT NOT SPECIFIED
                    error();
                }
            }
            if (consumed != kwnames.size()) {
                // NOT ALL KWNAMES ARGUMENTS WERE USED
                error();
            }
        }
    }
    int index(const char* name, int pos) {
        if (pos < nargs) {
            return pos;
        }
        if (kwnames.ptr()) {
            for (auto j : kwnames.enumerate()) {
                if (!strcmp(name, kwnames[j])) {
                    return nargs + j;
                }
            }
        }
        return -1;
    }
};

inline object handle::call_vector(vector_args args) {
    return object::checked_steal(PY_VECTORCALL(ptr(), (PyObject*const*) args.args, args.nargs, args.kwnames.ptr()));
}


}

#define MPY_ARGS_NAME(typ, name) #name ,
#define MPY_ARGS_DECLARE(typ, name) typ name;
#define MPY_ARGS_POINTER(typ, name) &name ,
#define MPY_PARSE_ARGS_KWARGS(fmt, FORALL_ARGS) \
    static char* kwlist[] = { FORALL_ARGS(MPY_ARGS_NAME) nullptr}; \
    FORALL_ARGS(MPY_ARGS_DECLARE) \
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, fmt, kwlist, FORALL_ARGS(MPY_ARGS_POINTER) nullptr)) { \
        throw mpy::exception_set(); \
    }

#define MPY_PARSE_ARGS_KWNAMES(fmt, FORALL_ARGS) \
    static const char * const kwlist[] = { FORALL_ARGS(MPY_ARGS_NAME) nullptr}; \
    FORALL_ARGS(MPY_ARGS_DECLARE) \
    static _PyArg_Parser parser = {fmt, kwlist, 0}; \
    if (!_PyArg_ParseStackAndKeywords(args, nargs, kwnames, &parser, FORALL_ARGS(MPY_ARGS_POINTER) nullptr)) { \
        throw mpy::exception_set(); \
    }
