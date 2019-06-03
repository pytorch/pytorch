/*
    pybind11/pytypes.h: Convenience wrapper classes for basic Python types

    Copyright (c) 2016 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE file.
*/

#pragma once

#include "detail/common.h"
#include "buffer_info.h"
#include <utility>
#include <type_traits>

NAMESPACE_BEGIN(PYBIND11_NAMESPACE)

/* A few forward declarations */
class handle; class object;
class str; class iterator;
struct arg; struct arg_v;

NAMESPACE_BEGIN(detail)
class args_proxy;
inline bool isinstance_generic(handle obj, const std::type_info &tp);

// Accessor forward declarations
template <typename Policy> class accessor;
namespace accessor_policies {
    struct obj_attr;
    struct str_attr;
    struct generic_item;
    struct sequence_item;
    struct list_item;
    struct tuple_item;
}
using obj_attr_accessor = accessor<accessor_policies::obj_attr>;
using str_attr_accessor = accessor<accessor_policies::str_attr>;
using item_accessor = accessor<accessor_policies::generic_item>;
using sequence_accessor = accessor<accessor_policies::sequence_item>;
using list_accessor = accessor<accessor_policies::list_item>;
using tuple_accessor = accessor<accessor_policies::tuple_item>;

/// Tag and check to identify a class which implements the Python object API
class pyobject_tag { };
template <typename T> using is_pyobject = std::is_base_of<pyobject_tag, remove_reference_t<T>>;

/** \rst
    A mixin class which adds common functions to `handle`, `object` and various accessors.
    The only requirement for `Derived` is to implement ``PyObject *Derived::ptr() const``.
\endrst */
template <typename Derived>
class object_api : public pyobject_tag {
    const Derived &derived() const { return static_cast<const Derived &>(*this); }

public:
    /** \rst
        Return an iterator equivalent to calling ``iter()`` in Python. The object
        must be a collection which supports the iteration protocol.
    \endrst */
    iterator begin() const;
    /// Return a sentinel which ends iteration.
    iterator end() const;

    /** \rst
        Return an internal functor to invoke the object's sequence protocol. Casting
        the returned ``detail::item_accessor`` instance to a `handle` or `object`
        subclass causes a corresponding call to ``__getitem__``. Assigning a `handle`
        or `object` subclass causes a call to ``__setitem__``.
    \endrst */
    item_accessor operator[](handle key) const;
    /// See above (the only difference is that they key is provided as a string literal)
    item_accessor operator[](const char *key) const;

    /** \rst
        Return an internal functor to access the object's attributes. Casting the
        returned ``detail::obj_attr_accessor`` instance to a `handle` or `object`
        subclass causes a corresponding call to ``getattr``. Assigning a `handle`
        or `object` subclass causes a call to ``setattr``.
    \endrst */
    obj_attr_accessor attr(handle key) const;
    /// See above (the only difference is that they key is provided as a string literal)
    str_attr_accessor attr(const char *key) const;

    /** \rst
        Matches * unpacking in Python, e.g. to unpack arguments out of a ``tuple``
        or ``list`` for a function call. Applying another * to the result yields
        ** unpacking, e.g. to unpack a dict as function keyword arguments.
        See :ref:`calling_python_functions`.
    \endrst */
    args_proxy operator*() const;

    /// Check if the given item is contained within this object, i.e. ``item in obj``.
    template <typename T> bool contains(T &&item) const;

    /** \rst
        Assuming the Python object is a function or implements the ``__call__``
        protocol, ``operator()`` invokes the underlying function, passing an
        arbitrary set of parameters. The result is returned as a `object` and
        may need to be converted back into a Python object using `handle::cast()`.

        When some of the arguments cannot be converted to Python objects, the
        function will throw a `cast_error` exception. When the Python function
        call fails, a `error_already_set` exception is thrown.
    \endrst */
    template <return_value_policy policy = return_value_policy::automatic_reference, typename... Args>
    object operator()(Args &&...args) const;
    template <return_value_policy policy = return_value_policy::automatic_reference, typename... Args>
    PYBIND11_DEPRECATED("call(...) was deprecated in favor of operator()(...)")
        object call(Args&&... args) const;

    /// Equivalent to ``obj is other`` in Python.
    bool is(object_api const& other) const { return derived().ptr() == other.derived().ptr(); }
    /// Equivalent to ``obj is None`` in Python.
    bool is_none() const { return derived().ptr() == Py_None; }
    /// Equivalent to obj == other in Python
    bool equal(object_api const &other) const      { return rich_compare(other, Py_EQ); }
    bool not_equal(object_api const &other) const  { return rich_compare(other, Py_NE); }
    bool operator<(object_api const &other) const  { return rich_compare(other, Py_LT); }
    bool operator<=(object_api const &other) const { return rich_compare(other, Py_LE); }
    bool operator>(object_api const &other) const  { return rich_compare(other, Py_GT); }
    bool operator>=(object_api const &other) const { return rich_compare(other, Py_GE); }

    object operator-() const;
    object operator~() const;
    object operator+(object_api const &other) const;
    object operator+=(object_api const &other) const;
    object operator-(object_api const &other) const;
    object operator-=(object_api const &other) const;
    object operator*(object_api const &other) const;
    object operator*=(object_api const &other) const;
    object operator/(object_api const &other) const;
    object operator/=(object_api const &other) const;
    object operator|(object_api const &other) const;
    object operator|=(object_api const &other) const;
    object operator&(object_api const &other) const;
    object operator&=(object_api const &other) const;
    object operator^(object_api const &other) const;
    object operator^=(object_api const &other) const;
    object operator<<(object_api const &other) const;
    object operator<<=(object_api const &other) const;
    object operator>>(object_api const &other) const;
    object operator>>=(object_api const &other) const;

    PYBIND11_DEPRECATED("Use py::str(obj) instead")
    pybind11::str str() const;

    /// Get or set the object's docstring, i.e. ``obj.__doc__``.
    str_attr_accessor doc() const;

    /// Return the object's current reference count
    int ref_count() const { return static_cast<int>(Py_REFCNT(derived().ptr())); }
    /// Return a handle to the Python type object underlying the instance
    handle get_type() const;

private:
    bool rich_compare(object_api const &other, int value) const;
};

NAMESPACE_END(detail)

/** \rst
    Holds a reference to a Python object (no reference counting)

    The `handle` class is a thin wrapper around an arbitrary Python object (i.e. a
    ``PyObject *`` in Python's C API). It does not perform any automatic reference
    counting and merely provides a basic C++ interface to various Python API functions.

    .. seealso::
        The `object` class inherits from `handle` and adds automatic reference
        counting features.
\endrst */
class handle : public detail::object_api<handle> {
public:
    /// The default constructor creates a handle with a ``nullptr``-valued pointer
    handle() = default;
    /// Creates a ``handle`` from the given raw Python object pointer
    handle(PyObject *ptr) : m_ptr(ptr) { } // Allow implicit conversion from PyObject*

    /// Return the underlying ``PyObject *`` pointer
    PyObject *ptr() const { return m_ptr; }
    PyObject *&ptr() { return m_ptr; }

    /** \rst
        Manually increase the reference count of the Python object. Usually, it is
        preferable to use the `object` class which derives from `handle` and calls
        this function automatically. Returns a reference to itself.
    \endrst */
    const handle& inc_ref() const & { Py_XINCREF(m_ptr); return *this; }

    /** \rst
        Manually decrease the reference count of the Python object. Usually, it is
        preferable to use the `object` class which derives from `handle` and calls
        this function automatically. Returns a reference to itself.
    \endrst */
    const handle& dec_ref() const & { Py_XDECREF(m_ptr); return *this; }

    /** \rst
        Attempt to cast the Python object into the given C++ type. A `cast_error`
        will be throw upon failure.
    \endrst */
    template <typename T> T cast() const;
    /// Return ``true`` when the `handle` wraps a valid Python object
    explicit operator bool() const { return m_ptr != nullptr; }
    /** \rst
        Deprecated: Check that the underlying pointers are the same.
        Equivalent to ``obj1 is obj2`` in Python.
    \endrst */
    PYBIND11_DEPRECATED("Use obj1.is(obj2) instead")
    bool operator==(const handle &h) const { return m_ptr == h.m_ptr; }
    PYBIND11_DEPRECATED("Use !obj1.is(obj2) instead")
    bool operator!=(const handle &h) const { return m_ptr != h.m_ptr; }
    PYBIND11_DEPRECATED("Use handle::operator bool() instead")
    bool check() const { return m_ptr != nullptr; }
protected:
    PyObject *m_ptr = nullptr;
};

/** \rst
    Holds a reference to a Python object (with reference counting)

    Like `handle`, the `object` class is a thin wrapper around an arbitrary Python
    object (i.e. a ``PyObject *`` in Python's C API). In contrast to `handle`, it
    optionally increases the object's reference count upon construction, and it
    *always* decreases the reference count when the `object` instance goes out of
    scope and is destructed. When using `object` instances consistently, it is much
    easier to get reference counting right at the first attempt.
\endrst */
class object : public handle {
public:
    object() = default;
    PYBIND11_DEPRECATED("Use reinterpret_borrow<object>() or reinterpret_steal<object>()")
    object(handle h, bool is_borrowed) : handle(h) { if (is_borrowed) inc_ref(); }
    /// Copy constructor; always increases the reference count
    object(const object &o) : handle(o) { inc_ref(); }
    /// Move constructor; steals the object from ``other`` and preserves its reference count
    object(object &&other) noexcept { m_ptr = other.m_ptr; other.m_ptr = nullptr; }
    /// Destructor; automatically calls `handle::dec_ref()`
    ~object() { dec_ref(); }

    /** \rst
        Resets the internal pointer to ``nullptr`` without without decreasing the
        object's reference count. The function returns a raw handle to the original
        Python object.
    \endrst */
    handle release() {
      PyObject *tmp = m_ptr;
      m_ptr = nullptr;
      return handle(tmp);
    }

    object& operator=(const object &other) {
        other.inc_ref();
        dec_ref();
        m_ptr = other.m_ptr;
        return *this;
    }

    object& operator=(object &&other) noexcept {
        if (this != &other) {
            handle temp(m_ptr);
            m_ptr = other.m_ptr;
            other.m_ptr = nullptr;
            temp.dec_ref();
        }
        return *this;
    }

    // Calling cast() on an object lvalue just copies (via handle::cast)
    template <typename T> T cast() const &;
    // Calling on an object rvalue does a move, if needed and/or possible
    template <typename T> T cast() &&;

protected:
    // Tags for choosing constructors from raw PyObject *
    struct borrowed_t { };
    struct stolen_t { };

    template <typename T> friend T reinterpret_borrow(handle);
    template <typename T> friend T reinterpret_steal(handle);

public:
    // Only accessible from derived classes and the reinterpret_* functions
    object(handle h, borrowed_t) : handle(h) { inc_ref(); }
    object(handle h, stolen_t) : handle(h) { }
};

/** \rst
    Declare that a `handle` or ``PyObject *`` is a certain type and borrow the reference.
    The target type ``T`` must be `object` or one of its derived classes. The function
    doesn't do any conversions or checks. It's up to the user to make sure that the
    target type is correct.

    .. code-block:: cpp

        PyObject *p = PyList_GetItem(obj, index);
        py::object o = reinterpret_borrow<py::object>(p);
        // or
        py::tuple t = reinterpret_borrow<py::tuple>(p); // <-- `p` must be already be a `tuple`
\endrst */
template <typename T> T reinterpret_borrow(handle h) { return {h, object::borrowed_t{}}; }

/** \rst
    Like `reinterpret_borrow`, but steals the reference.

     .. code-block:: cpp

        PyObject *p = PyObject_Str(obj);
        py::str s = reinterpret_steal<py::str>(p); // <-- `p` must be already be a `str`
\endrst */
template <typename T> T reinterpret_steal(handle h) { return {h, object::stolen_t{}}; }

NAMESPACE_BEGIN(detail)
inline std::string error_string();
NAMESPACE_END(detail)

/// Fetch and hold an error which was already set in Python.  An instance of this is typically
/// thrown to propagate python-side errors back through C++ which can either be caught manually or
/// else falls back to the function dispatcher (which then raises the captured error back to
/// python).
class error_already_set : public std::runtime_error {
public:
    /// Constructs a new exception from the current Python error indicator, if any.  The current
    /// Python error indicator will be cleared.
    error_already_set() : std::runtime_error(detail::error_string()) {
        PyErr_Fetch(&type.ptr(), &value.ptr(), &trace.ptr());
    }

    error_already_set(const error_already_set &) = default;
    error_already_set(error_already_set &&) = default;

    inline ~error_already_set();

    /// Give the currently-held error back to Python, if any.  If there is currently a Python error
    /// already set it is cleared first.  After this call, the current object no longer stores the
    /// error variables (but the `.what()` string is still available).
    void restore() { PyErr_Restore(type.release().ptr(), value.release().ptr(), trace.release().ptr()); }

    // Does nothing; provided for backwards compatibility.
    PYBIND11_DEPRECATED("Use of error_already_set.clear() is deprecated")
    void clear() {}

    /// Check if the currently trapped error type matches the given Python exception class (or a
    /// subclass thereof).  May also be passed a tuple to search for any exception class matches in
    /// the given tuple.
    bool matches(handle ex) const { return PyErr_GivenExceptionMatches(ex.ptr(), type.ptr()); }

private:
    object type, value, trace;
};

/** \defgroup python_builtins _
    Unless stated otherwise, the following C++ functions behave the same
    as their Python counterparts.
 */

/** \ingroup python_builtins
    \rst
    Return true if ``obj`` is an instance of ``T``. Type ``T`` must be a subclass of
    `object` or a class which was exposed to Python as ``py::class_<T>``.
\endrst */
template <typename T, detail::enable_if_t<std::is_base_of<object, T>::value, int> = 0>
bool isinstance(handle obj) { return T::check_(obj); }

template <typename T, detail::enable_if_t<!std::is_base_of<object, T>::value, int> = 0>
bool isinstance(handle obj) { return detail::isinstance_generic(obj, typeid(T)); }

template <> inline bool isinstance<handle>(handle obj) = delete;
template <> inline bool isinstance<object>(handle obj) { return obj.ptr() != nullptr; }

/// \ingroup python_builtins
/// Return true if ``obj`` is an instance of the ``type``.
inline bool isinstance(handle obj, handle type) {
    const auto result = PyObject_IsInstance(obj.ptr(), type.ptr());
    if (result == -1)
        throw error_already_set();
    return result != 0;
}

/// \addtogroup python_builtins
/// @{
inline bool hasattr(handle obj, handle name) {
    return PyObject_HasAttr(obj.ptr(), name.ptr()) == 1;
}

inline bool hasattr(handle obj, const char *name) {
    return PyObject_HasAttrString(obj.ptr(), name) == 1;
}

inline void delattr(handle obj, handle name) {
    if (PyObject_DelAttr(obj.ptr(), name.ptr()) != 0) { throw error_already_set(); }
}

inline void delattr(handle obj, const char *name) {
    if (PyObject_DelAttrString(obj.ptr(), name) != 0) { throw error_already_set(); }
}

inline object getattr(handle obj, handle name) {
    PyObject *result = PyObject_GetAttr(obj.ptr(), name.ptr());
    if (!result) { throw error_already_set(); }
    return reinterpret_steal<object>(result);
}

inline object getattr(handle obj, const char *name) {
    PyObject *result = PyObject_GetAttrString(obj.ptr(), name);
    if (!result) { throw error_already_set(); }
    return reinterpret_steal<object>(result);
}

inline object getattr(handle obj, handle name, handle default_) {
    if (PyObject *result = PyObject_GetAttr(obj.ptr(), name.ptr())) {
        return reinterpret_steal<object>(result);
    } else {
        PyErr_Clear();
        return reinterpret_borrow<object>(default_);
    }
}

inline object getattr(handle obj, const char *name, handle default_) {
    if (PyObject *result = PyObject_GetAttrString(obj.ptr(), name)) {
        return reinterpret_steal<object>(result);
    } else {
        PyErr_Clear();
        return reinterpret_borrow<object>(default_);
    }
}

inline void setattr(handle obj, handle name, handle value) {
    if (PyObject_SetAttr(obj.ptr(), name.ptr(), value.ptr()) != 0) { throw error_already_set(); }
}

inline void setattr(handle obj, const char *name, handle value) {
    if (PyObject_SetAttrString(obj.ptr(), name, value.ptr()) != 0) { throw error_already_set(); }
}

inline ssize_t hash(handle obj) {
    auto h = PyObject_Hash(obj.ptr());
    if (h == -1) { throw error_already_set(); }
    return h;
}

/// @} python_builtins

NAMESPACE_BEGIN(detail)
inline handle get_function(handle value) {
    if (value) {
#if PY_MAJOR_VERSION >= 3
        if (PyInstanceMethod_Check(value.ptr()))
            value = PyInstanceMethod_GET_FUNCTION(value.ptr());
        else
#endif
        if (PyMethod_Check(value.ptr()))
            value = PyMethod_GET_FUNCTION(value.ptr());
    }
    return value;
}

// Helper aliases/functions to support implicit casting of values given to python accessors/methods.
// When given a pyobject, this simply returns the pyobject as-is; for other C++ type, the value goes
// through pybind11::cast(obj) to convert it to an `object`.
template <typename T, enable_if_t<is_pyobject<T>::value, int> = 0>
auto object_or_cast(T &&o) -> decltype(std::forward<T>(o)) { return std::forward<T>(o); }
// The following casting version is implemented in cast.h:
template <typename T, enable_if_t<!is_pyobject<T>::value, int> = 0>
object object_or_cast(T &&o);
// Match a PyObject*, which we want to convert directly to handle via its converting constructor
inline handle object_or_cast(PyObject *ptr) { return ptr; }

template <typename Policy>
class accessor : public object_api<accessor<Policy>> {
    using key_type = typename Policy::key_type;

public:
    accessor(handle obj, key_type key) : obj(obj), key(std::move(key)) { }
    accessor(const accessor &) = default;
    accessor(accessor &&) = default;

    // accessor overload required to override default assignment operator (templates are not allowed
    // to replace default compiler-generated assignments).
    void operator=(const accessor &a) && { std::move(*this).operator=(handle(a)); }
    void operator=(const accessor &a) & { operator=(handle(a)); }

    template <typename T> void operator=(T &&value) && {
        Policy::set(obj, key, object_or_cast(std::forward<T>(value)));
    }
    template <typename T> void operator=(T &&value) & {
        get_cache() = reinterpret_borrow<object>(object_or_cast(std::forward<T>(value)));
    }

    template <typename T = Policy>
    PYBIND11_DEPRECATED("Use of obj.attr(...) as bool is deprecated in favor of pybind11::hasattr(obj, ...)")
    explicit operator enable_if_t<std::is_same<T, accessor_policies::str_attr>::value ||
            std::is_same<T, accessor_policies::obj_attr>::value, bool>() const {
        return hasattr(obj, key);
    }
    template <typename T = Policy>
    PYBIND11_DEPRECATED("Use of obj[key] as bool is deprecated in favor of obj.contains(key)")
    explicit operator enable_if_t<std::is_same<T, accessor_policies::generic_item>::value, bool>() const {
        return obj.contains(key);
    }

    operator object() const { return get_cache(); }
    PyObject *ptr() const { return get_cache().ptr(); }
    template <typename T> T cast() const { return get_cache().template cast<T>(); }

private:
    object &get_cache() const {
        if (!cache) { cache = Policy::get(obj, key); }
        return cache;
    }

private:
    handle obj;
    key_type key;
    mutable object cache;
};

NAMESPACE_BEGIN(accessor_policies)
struct obj_attr {
    using key_type = object;
    static object get(handle obj, handle key) { return getattr(obj, key); }
    static void set(handle obj, handle key, handle val) { setattr(obj, key, val); }
};

struct str_attr {
    using key_type = const char *;
    static object get(handle obj, const char *key) { return getattr(obj, key); }
    static void set(handle obj, const char *key, handle val) { setattr(obj, key, val); }
};

struct generic_item {
    using key_type = object;

    static object get(handle obj, handle key) {
        PyObject *result = PyObject_GetItem(obj.ptr(), key.ptr());
        if (!result) { throw error_already_set(); }
        return reinterpret_steal<object>(result);
    }

    static void set(handle obj, handle key, handle val) {
        if (PyObject_SetItem(obj.ptr(), key.ptr(), val.ptr()) != 0) { throw error_already_set(); }
    }
};

struct sequence_item {
    using key_type = size_t;

    static object get(handle obj, size_t index) {
        PyObject *result = PySequence_GetItem(obj.ptr(), static_cast<ssize_t>(index));
        if (!result) { throw error_already_set(); }
        return reinterpret_steal<object>(result);
    }

    static void set(handle obj, size_t index, handle val) {
        // PySequence_SetItem does not steal a reference to 'val'
        if (PySequence_SetItem(obj.ptr(), static_cast<ssize_t>(index), val.ptr()) != 0) {
            throw error_already_set();
        }
    }
};

struct list_item {
    using key_type = size_t;

    static object get(handle obj, size_t index) {
        PyObject *result = PyList_GetItem(obj.ptr(), static_cast<ssize_t>(index));
        if (!result) { throw error_already_set(); }
        return reinterpret_borrow<object>(result);
    }

    static void set(handle obj, size_t index, handle val) {
        // PyList_SetItem steals a reference to 'val'
        if (PyList_SetItem(obj.ptr(), static_cast<ssize_t>(index), val.inc_ref().ptr()) != 0) {
            throw error_already_set();
        }
    }
};

struct tuple_item {
    using key_type = size_t;

    static object get(handle obj, size_t index) {
        PyObject *result = PyTuple_GetItem(obj.ptr(), static_cast<ssize_t>(index));
        if (!result) { throw error_already_set(); }
        return reinterpret_borrow<object>(result);
    }

    static void set(handle obj, size_t index, handle val) {
        // PyTuple_SetItem steals a reference to 'val'
        if (PyTuple_SetItem(obj.ptr(), static_cast<ssize_t>(index), val.inc_ref().ptr()) != 0) {
            throw error_already_set();
        }
    }
};
NAMESPACE_END(accessor_policies)

/// STL iterator template used for tuple, list, sequence and dict
template <typename Policy>
class generic_iterator : public Policy {
    using It = generic_iterator;

public:
    using difference_type = ssize_t;
    using iterator_category = typename Policy::iterator_category;
    using value_type = typename Policy::value_type;
    using reference = typename Policy::reference;
    using pointer = typename Policy::pointer;

    generic_iterator() = default;
    generic_iterator(handle seq, ssize_t index) : Policy(seq, index) { }

    reference operator*() const { return Policy::dereference(); }
    reference operator[](difference_type n) const { return *(*this + n); }
    pointer operator->() const { return **this; }

    It &operator++() { Policy::increment(); return *this; }
    It operator++(int) { auto copy = *this; Policy::increment(); return copy; }
    It &operator--() { Policy::decrement(); return *this; }
    It operator--(int) { auto copy = *this; Policy::decrement(); return copy; }
    It &operator+=(difference_type n) { Policy::advance(n); return *this; }
    It &operator-=(difference_type n) { Policy::advance(-n); return *this; }

    friend It operator+(const It &a, difference_type n) { auto copy = a; return copy += n; }
    friend It operator+(difference_type n, const It &b) { return b + n; }
    friend It operator-(const It &a, difference_type n) { auto copy = a; return copy -= n; }
    friend difference_type operator-(const It &a, const It &b) { return a.distance_to(b); }

    friend bool operator==(const It &a, const It &b) { return a.equal(b); }
    friend bool operator!=(const It &a, const It &b) { return !(a == b); }
    friend bool operator< (const It &a, const It &b) { return b - a > 0; }
    friend bool operator> (const It &a, const It &b) { return b < a; }
    friend bool operator>=(const It &a, const It &b) { return !(a < b); }
    friend bool operator<=(const It &a, const It &b) { return !(a > b); }
};

NAMESPACE_BEGIN(iterator_policies)
/// Quick proxy class needed to implement ``operator->`` for iterators which can't return pointers
template <typename T>
struct arrow_proxy {
    T value;

    arrow_proxy(T &&value) : value(std::move(value)) { }
    T *operator->() const { return &value; }
};

/// Lightweight iterator policy using just a simple pointer: see ``PySequence_Fast_ITEMS``
class sequence_fast_readonly {
protected:
    using iterator_category = std::random_access_iterator_tag;
    using value_type = handle;
    using reference = const handle;
    using pointer = arrow_proxy<const handle>;

    sequence_fast_readonly(handle obj, ssize_t n) : ptr(PySequence_Fast_ITEMS(obj.ptr()) + n) { }

    reference dereference() const { return *ptr; }
    void increment() { ++ptr; }
    void decrement() { --ptr; }
    void advance(ssize_t n) { ptr += n; }
    bool equal(const sequence_fast_readonly &b) const { return ptr == b.ptr; }
    ssize_t distance_to(const sequence_fast_readonly &b) const { return ptr - b.ptr; }

private:
    PyObject **ptr;
};

/// Full read and write access using the sequence protocol: see ``detail::sequence_accessor``
class sequence_slow_readwrite {
protected:
    using iterator_category = std::random_access_iterator_tag;
    using value_type = object;
    using reference = sequence_accessor;
    using pointer = arrow_proxy<const sequence_accessor>;

    sequence_slow_readwrite(handle obj, ssize_t index) : obj(obj), index(index) { }

    reference dereference() const { return {obj, static_cast<size_t>(index)}; }
    void increment() { ++index; }
    void decrement() { --index; }
    void advance(ssize_t n) { index += n; }
    bool equal(const sequence_slow_readwrite &b) const { return index == b.index; }
    ssize_t distance_to(const sequence_slow_readwrite &b) const { return index - b.index; }

private:
    handle obj;
    ssize_t index;
};

/// Python's dictionary protocol permits this to be a forward iterator
class dict_readonly {
protected:
    using iterator_category = std::forward_iterator_tag;
    using value_type = std::pair<handle, handle>;
    using reference = const value_type;
    using pointer = arrow_proxy<const value_type>;

    dict_readonly() = default;
    dict_readonly(handle obj, ssize_t pos) : obj(obj), pos(pos) { increment(); }

    reference dereference() const { return {key, value}; }
    void increment() { if (!PyDict_Next(obj.ptr(), &pos, &key, &value)) { pos = -1; } }
    bool equal(const dict_readonly &b) const { return pos == b.pos; }

private:
    handle obj;
    PyObject *key, *value;
    ssize_t pos = -1;
};
NAMESPACE_END(iterator_policies)

#if !defined(PYPY_VERSION)
using tuple_iterator = generic_iterator<iterator_policies::sequence_fast_readonly>;
using list_iterator = generic_iterator<iterator_policies::sequence_fast_readonly>;
#else
using tuple_iterator = generic_iterator<iterator_policies::sequence_slow_readwrite>;
using list_iterator = generic_iterator<iterator_policies::sequence_slow_readwrite>;
#endif

using sequence_iterator = generic_iterator<iterator_policies::sequence_slow_readwrite>;
using dict_iterator = generic_iterator<iterator_policies::dict_readonly>;

inline bool PyIterable_Check(PyObject *obj) {
    PyObject *iter = PyObject_GetIter(obj);
    if (iter) {
        Py_DECREF(iter);
        return true;
    } else {
        PyErr_Clear();
        return false;
    }
}

inline bool PyNone_Check(PyObject *o) { return o == Py_None; }
#if PY_MAJOR_VERSION >= 3
inline bool PyEllipsis_Check(PyObject *o) { return o == Py_Ellipsis; }
#endif

inline bool PyUnicode_Check_Permissive(PyObject *o) { return PyUnicode_Check(o) || PYBIND11_BYTES_CHECK(o); }

class kwargs_proxy : public handle {
public:
    explicit kwargs_proxy(handle h) : handle(h) { }
};

class args_proxy : public handle {
public:
    explicit args_proxy(handle h) : handle(h) { }
    kwargs_proxy operator*() const { return kwargs_proxy(*this); }
};

/// Python argument categories (using PEP 448 terms)
template <typename T> using is_keyword = std::is_base_of<arg, T>;
template <typename T> using is_s_unpacking = std::is_same<args_proxy, T>; // * unpacking
template <typename T> using is_ds_unpacking = std::is_same<kwargs_proxy, T>; // ** unpacking
template <typename T> using is_positional = satisfies_none_of<T,
    is_keyword, is_s_unpacking, is_ds_unpacking
>;
template <typename T> using is_keyword_or_ds = satisfies_any_of<T, is_keyword, is_ds_unpacking>;

// Call argument collector forward declarations
template <return_value_policy policy = return_value_policy::automatic_reference>
class simple_collector;
template <return_value_policy policy = return_value_policy::automatic_reference>
class unpacking_collector;

NAMESPACE_END(detail)

// TODO: After the deprecated constructors are removed, this macro can be simplified by
//       inheriting ctors: `using Parent::Parent`. It's not an option right now because
//       the `using` statement triggers the parent deprecation warning even if the ctor
//       isn't even used.
#define PYBIND11_OBJECT_COMMON(Name, Parent, CheckFun) \
    public: \
        PYBIND11_DEPRECATED("Use reinterpret_borrow<"#Name">() or reinterpret_steal<"#Name">()") \
        Name(handle h, bool is_borrowed) : Parent(is_borrowed ? Parent(h, borrowed_t{}) : Parent(h, stolen_t{})) { } \
        Name(handle h, borrowed_t) : Parent(h, borrowed_t{}) { } \
        Name(handle h, stolen_t) : Parent(h, stolen_t{}) { } \
        PYBIND11_DEPRECATED("Use py::isinstance<py::python_type>(obj) instead") \
        bool check() const { return m_ptr != nullptr && (bool) CheckFun(m_ptr); } \
        static bool check_(handle h) { return h.ptr() != nullptr && CheckFun(h.ptr()); }

#define PYBIND11_OBJECT_CVT(Name, Parent, CheckFun, ConvertFun) \
    PYBIND11_OBJECT_COMMON(Name, Parent, CheckFun) \
    /* This is deliberately not 'explicit' to allow implicit conversion from object: */ \
    Name(const object &o) \
    : Parent(check_(o) ? o.inc_ref().ptr() : ConvertFun(o.ptr()), stolen_t{}) \
    { if (!m_ptr) throw error_already_set(); } \
    Name(object &&o) \
    : Parent(check_(o) ? o.release().ptr() : ConvertFun(o.ptr()), stolen_t{}) \
    { if (!m_ptr) throw error_already_set(); } \
    template <typename Policy_> \
    Name(const ::pybind11::detail::accessor<Policy_> &a) : Name(object(a)) { }

#define PYBIND11_OBJECT(Name, Parent, CheckFun) \
    PYBIND11_OBJECT_COMMON(Name, Parent, CheckFun) \
    /* This is deliberately not 'explicit' to allow implicit conversion from object: */ \
    Name(const object &o) : Parent(o) { } \
    Name(object &&o) : Parent(std::move(o)) { }

#define PYBIND11_OBJECT_DEFAULT(Name, Parent, CheckFun) \
    PYBIND11_OBJECT(Name, Parent, CheckFun) \
    Name() : Parent() { }

/// \addtogroup pytypes
/// @{

/** \rst
    Wraps a Python iterator so that it can also be used as a C++ input iterator

    Caveat: copying an iterator does not (and cannot) clone the internal
    state of the Python iterable. This also applies to the post-increment
    operator. This iterator should only be used to retrieve the current
    value using ``operator*()``.
\endrst */
class iterator : public object {
public:
    using iterator_category = std::input_iterator_tag;
    using difference_type = ssize_t;
    using value_type = handle;
    using reference = const handle;
    using pointer = const handle *;

    PYBIND11_OBJECT_DEFAULT(iterator, object, PyIter_Check)

    iterator& operator++() {
        advance();
        return *this;
    }

    iterator operator++(int) {
        auto rv = *this;
        advance();
        return rv;
    }

    reference operator*() const {
        if (m_ptr && !value.ptr()) {
            auto& self = const_cast<iterator &>(*this);
            self.advance();
        }
        return value;
    }

    pointer operator->() const { operator*(); return &value; }

    /** \rst
         The value which marks the end of the iteration. ``it == iterator::sentinel()``
         is equivalent to catching ``StopIteration`` in Python.

         .. code-block:: cpp

             void foo(py::iterator it) {
                 while (it != py::iterator::sentinel()) {
                    // use `*it`
                    ++it;
                 }
             }
    \endrst */
    static iterator sentinel() { return {}; }

    friend bool operator==(const iterator &a, const iterator &b) { return a->ptr() == b->ptr(); }
    friend bool operator!=(const iterator &a, const iterator &b) { return a->ptr() != b->ptr(); }

private:
    void advance() {
        value = reinterpret_steal<object>(PyIter_Next(m_ptr));
        if (PyErr_Occurred()) { throw error_already_set(); }
    }

private:
    object value = {};
};

class iterable : public object {
public:
    PYBIND11_OBJECT_DEFAULT(iterable, object, detail::PyIterable_Check)
};

class bytes;

class str : public object {
public:
    PYBIND11_OBJECT_CVT(str, object, detail::PyUnicode_Check_Permissive, raw_str)

    str(const char *c, size_t n)
        : object(PyUnicode_FromStringAndSize(c, (ssize_t) n), stolen_t{}) {
        if (!m_ptr) pybind11_fail("Could not allocate string object!");
    }

    // 'explicit' is explicitly omitted from the following constructors to allow implicit conversion to py::str from C++ string-like objects
    str(const char *c = "")
        : object(PyUnicode_FromString(c), stolen_t{}) {
        if (!m_ptr) pybind11_fail("Could not allocate string object!");
    }

    str(const std::string &s) : str(s.data(), s.size()) { }

    explicit str(const bytes &b);

    /** \rst
        Return a string representation of the object. This is analogous to
        the ``str()`` function in Python.
    \endrst */
    explicit str(handle h) : object(raw_str(h.ptr()), stolen_t{}) { }

    operator std::string() const {
        object temp = *this;
        if (PyUnicode_Check(m_ptr)) {
            temp = reinterpret_steal<object>(PyUnicode_AsUTF8String(m_ptr));
            if (!temp)
                pybind11_fail("Unable to extract string contents! (encoding issue)");
        }
        char *buffer;
        ssize_t length;
        if (PYBIND11_BYTES_AS_STRING_AND_SIZE(temp.ptr(), &buffer, &length))
            pybind11_fail("Unable to extract string contents! (invalid type)");
        return std::string(buffer, (size_t) length);
    }

    template <typename... Args>
    str format(Args &&...args) const {
        return attr("format")(std::forward<Args>(args)...);
    }

private:
    /// Return string representation -- always returns a new reference, even if already a str
    static PyObject *raw_str(PyObject *op) {
        PyObject *str_value = PyObject_Str(op);
#if PY_MAJOR_VERSION < 3
        if (!str_value) throw error_already_set();
        PyObject *unicode = PyUnicode_FromEncodedObject(str_value, "utf-8", nullptr);
        Py_XDECREF(str_value); str_value = unicode;
#endif
        return str_value;
    }
};
/// @} pytypes

inline namespace literals {
/** \rst
    String literal version of `str`
 \endrst */
inline str operator"" _s(const char *s, size_t size) { return {s, size}; }
}

/// \addtogroup pytypes
/// @{
class bytes : public object {
public:
    PYBIND11_OBJECT(bytes, object, PYBIND11_BYTES_CHECK)

    // Allow implicit conversion:
    bytes(const char *c = "")
        : object(PYBIND11_BYTES_FROM_STRING(c), stolen_t{}) {
        if (!m_ptr) pybind11_fail("Could not allocate bytes object!");
    }

    bytes(const char *c, size_t n)
        : object(PYBIND11_BYTES_FROM_STRING_AND_SIZE(c, (ssize_t) n), stolen_t{}) {
        if (!m_ptr) pybind11_fail("Could not allocate bytes object!");
    }

    // Allow implicit conversion:
    bytes(const std::string &s) : bytes(s.data(), s.size()) { }

    explicit bytes(const pybind11::str &s);

    operator std::string() const {
        char *buffer;
        ssize_t length;
        if (PYBIND11_BYTES_AS_STRING_AND_SIZE(m_ptr, &buffer, &length))
            pybind11_fail("Unable to extract bytes contents!");
        return std::string(buffer, (size_t) length);
    }
};

inline bytes::bytes(const pybind11::str &s) {
    object temp = s;
    if (PyUnicode_Check(s.ptr())) {
        temp = reinterpret_steal<object>(PyUnicode_AsUTF8String(s.ptr()));
        if (!temp)
            pybind11_fail("Unable to extract string contents! (encoding issue)");
    }
    char *buffer;
    ssize_t length;
    if (PYBIND11_BYTES_AS_STRING_AND_SIZE(temp.ptr(), &buffer, &length))
        pybind11_fail("Unable to extract string contents! (invalid type)");
    auto obj = reinterpret_steal<object>(PYBIND11_BYTES_FROM_STRING_AND_SIZE(buffer, length));
    if (!obj)
        pybind11_fail("Could not allocate bytes object!");
    m_ptr = obj.release().ptr();
}

inline str::str(const bytes& b) {
    char *buffer;
    ssize_t length;
    if (PYBIND11_BYTES_AS_STRING_AND_SIZE(b.ptr(), &buffer, &length))
        pybind11_fail("Unable to extract bytes contents!");
    auto obj = reinterpret_steal<object>(PyUnicode_FromStringAndSize(buffer, (ssize_t) length));
    if (!obj)
        pybind11_fail("Could not allocate string object!");
    m_ptr = obj.release().ptr();
}

class none : public object {
public:
    PYBIND11_OBJECT(none, object, detail::PyNone_Check)
    none() : object(Py_None, borrowed_t{}) { }
};

#if PY_MAJOR_VERSION >= 3
class ellipsis : public object {
public:
    PYBIND11_OBJECT(ellipsis, object, detail::PyEllipsis_Check)
    ellipsis() : object(Py_Ellipsis, borrowed_t{}) { }
};
#endif

class bool_ : public object {
public:
    PYBIND11_OBJECT_CVT(bool_, object, PyBool_Check, raw_bool)
    bool_() : object(Py_False, borrowed_t{}) { }
    // Allow implicit conversion from and to `bool`:
    bool_(bool value) : object(value ? Py_True : Py_False, borrowed_t{}) { }
    operator bool() const { return m_ptr && PyLong_AsLong(m_ptr) != 0; }

private:
    /// Return the truth value of an object -- always returns a new reference
    static PyObject *raw_bool(PyObject *op) {
        const auto value = PyObject_IsTrue(op);
        if (value == -1) return nullptr;
        return handle(value ? Py_True : Py_False).inc_ref().ptr();
    }
};

NAMESPACE_BEGIN(detail)
// Converts a value to the given unsigned type.  If an error occurs, you get back (Unsigned) -1;
// otherwise you get back the unsigned long or unsigned long long value cast to (Unsigned).
// (The distinction is critically important when casting a returned -1 error value to some other
// unsigned type: (A)-1 != (B)-1 when A and B are unsigned types of different sizes).
template <typename Unsigned>
Unsigned as_unsigned(PyObject *o) {
    if (sizeof(Unsigned) <= sizeof(unsigned long)
#if PY_VERSION_HEX < 0x03000000
            || PyInt_Check(o)
#endif
    ) {
        unsigned long v = PyLong_AsUnsignedLong(o);
        return v == (unsigned long) -1 && PyErr_Occurred() ? (Unsigned) -1 : (Unsigned) v;
    }
    else {
        unsigned long long v = PyLong_AsUnsignedLongLong(o);
        return v == (unsigned long long) -1 && PyErr_Occurred() ? (Unsigned) -1 : (Unsigned) v;
    }
}
NAMESPACE_END(detail)

class int_ : public object {
public:
    PYBIND11_OBJECT_CVT(int_, object, PYBIND11_LONG_CHECK, PyNumber_Long)
    int_() : object(PyLong_FromLong(0), stolen_t{}) { }
    // Allow implicit conversion from C++ integral types:
    template <typename T,
              detail::enable_if_t<std::is_integral<T>::value, int> = 0>
    int_(T value) {
        if (sizeof(T) <= sizeof(long)) {
            if (std::is_signed<T>::value)
                m_ptr = PyLong_FromLong((long) value);
            else
                m_ptr = PyLong_FromUnsignedLong((unsigned long) value);
        } else {
            if (std::is_signed<T>::value)
                m_ptr = PyLong_FromLongLong((long long) value);
            else
                m_ptr = PyLong_FromUnsignedLongLong((unsigned long long) value);
        }
        if (!m_ptr) pybind11_fail("Could not allocate int object!");
    }

    template <typename T,
              detail::enable_if_t<std::is_integral<T>::value, int> = 0>
    operator T() const {
        return std::is_unsigned<T>::value
            ? detail::as_unsigned<T>(m_ptr)
            : sizeof(T) <= sizeof(long)
              ? (T) PyLong_AsLong(m_ptr)
              : (T) PYBIND11_LONG_AS_LONGLONG(m_ptr);
    }
};

class float_ : public object {
public:
    PYBIND11_OBJECT_CVT(float_, object, PyFloat_Check, PyNumber_Float)
    // Allow implicit conversion from float/double:
    float_(float value) : object(PyFloat_FromDouble((double) value), stolen_t{}) {
        if (!m_ptr) pybind11_fail("Could not allocate float object!");
    }
    float_(double value = .0) : object(PyFloat_FromDouble((double) value), stolen_t{}) {
        if (!m_ptr) pybind11_fail("Could not allocate float object!");
    }
    operator float() const { return (float) PyFloat_AsDouble(m_ptr); }
    operator double() const { return (double) PyFloat_AsDouble(m_ptr); }
};

class weakref : public object {
public:
    PYBIND11_OBJECT_DEFAULT(weakref, object, PyWeakref_Check)
    explicit weakref(handle obj, handle callback = {})
        : object(PyWeakref_NewRef(obj.ptr(), callback.ptr()), stolen_t{}) {
        if (!m_ptr) pybind11_fail("Could not allocate weak reference!");
    }
};

class slice : public object {
public:
    PYBIND11_OBJECT_DEFAULT(slice, object, PySlice_Check)
    slice(ssize_t start_, ssize_t stop_, ssize_t step_) {
        int_ start(start_), stop(stop_), step(step_);
        m_ptr = PySlice_New(start.ptr(), stop.ptr(), step.ptr());
        if (!m_ptr) pybind11_fail("Could not allocate slice object!");
    }
    bool compute(size_t length, size_t *start, size_t *stop, size_t *step,
                 size_t *slicelength) const {
        return PySlice_GetIndicesEx((PYBIND11_SLICE_OBJECT *) m_ptr,
                                    (ssize_t) length, (ssize_t *) start,
                                    (ssize_t *) stop, (ssize_t *) step,
                                    (ssize_t *) slicelength) == 0;
    }
};

class capsule : public object {
public:
    PYBIND11_OBJECT_DEFAULT(capsule, object, PyCapsule_CheckExact)
    PYBIND11_DEPRECATED("Use reinterpret_borrow<capsule>() or reinterpret_steal<capsule>()")
    capsule(PyObject *ptr, bool is_borrowed) : object(is_borrowed ? object(ptr, borrowed_t{}) : object(ptr, stolen_t{})) { }

    explicit capsule(const void *value, const char *name = nullptr, void (*destructor)(PyObject *) = nullptr)
        : object(PyCapsule_New(const_cast<void *>(value), name, destructor), stolen_t{}) {
        if (!m_ptr)
            pybind11_fail("Could not allocate capsule object!");
    }

    PYBIND11_DEPRECATED("Please pass a destructor that takes a void pointer as input")
    capsule(const void *value, void (*destruct)(PyObject *))
        : object(PyCapsule_New(const_cast<void*>(value), nullptr, destruct), stolen_t{}) {
        if (!m_ptr)
            pybind11_fail("Could not allocate capsule object!");
    }

    capsule(const void *value, void (*destructor)(void *)) {
        m_ptr = PyCapsule_New(const_cast<void *>(value), nullptr, [](PyObject *o) {
            auto destructor = reinterpret_cast<void (*)(void *)>(PyCapsule_GetContext(o));
            void *ptr = PyCapsule_GetPointer(o, nullptr);
            destructor(ptr);
        });

        if (!m_ptr)
            pybind11_fail("Could not allocate capsule object!");

        if (PyCapsule_SetContext(m_ptr, (void *) destructor) != 0)
            pybind11_fail("Could not set capsule context!");
    }

    capsule(void (*destructor)()) {
        m_ptr = PyCapsule_New(reinterpret_cast<void *>(destructor), nullptr, [](PyObject *o) {
            auto destructor = reinterpret_cast<void (*)()>(PyCapsule_GetPointer(o, nullptr));
            destructor();
        });

        if (!m_ptr)
            pybind11_fail("Could not allocate capsule object!");
    }

    template <typename T> operator T *() const {
        auto name = this->name();
        T * result = static_cast<T *>(PyCapsule_GetPointer(m_ptr, name));
        if (!result) pybind11_fail("Unable to extract capsule contents!");
        return result;
    }

    const char *name() const { return PyCapsule_GetName(m_ptr); }
};

class tuple : public object {
public:
    PYBIND11_OBJECT_CVT(tuple, object, PyTuple_Check, PySequence_Tuple)
    explicit tuple(size_t size = 0) : object(PyTuple_New((ssize_t) size), stolen_t{}) {
        if (!m_ptr) pybind11_fail("Could not allocate tuple object!");
    }
    size_t size() const { return (size_t) PyTuple_Size(m_ptr); }
    detail::tuple_accessor operator[](size_t index) const { return {*this, index}; }
    detail::item_accessor operator[](handle h) const { return object::operator[](h); }
    detail::tuple_iterator begin() const { return {*this, 0}; }
    detail::tuple_iterator end() const { return {*this, PyTuple_GET_SIZE(m_ptr)}; }
};

class dict : public object {
public:
    PYBIND11_OBJECT_CVT(dict, object, PyDict_Check, raw_dict)
    dict() : object(PyDict_New(), stolen_t{}) {
        if (!m_ptr) pybind11_fail("Could not allocate dict object!");
    }
    template <typename... Args,
              typename = detail::enable_if_t<detail::all_of<detail::is_keyword_or_ds<Args>...>::value>,
              // MSVC workaround: it can't compile an out-of-line definition, so defer the collector
              typename collector = detail::deferred_t<detail::unpacking_collector<>, Args...>>
    explicit dict(Args &&...args) : dict(collector(std::forward<Args>(args)...).kwargs()) { }

    size_t size() const { return (size_t) PyDict_Size(m_ptr); }
    detail::dict_iterator begin() const { return {*this, 0}; }
    detail::dict_iterator end() const { return {}; }
    void clear() const { PyDict_Clear(ptr()); }
    bool contains(handle key) const { return PyDict_Contains(ptr(), key.ptr()) == 1; }
    bool contains(const char *key) const { return PyDict_Contains(ptr(), pybind11::str(key).ptr()) == 1; }

private:
    /// Call the `dict` Python type -- always returns a new reference
    static PyObject *raw_dict(PyObject *op) {
        if (PyDict_Check(op))
            return handle(op).inc_ref().ptr();
        return PyObject_CallFunctionObjArgs((PyObject *) &PyDict_Type, op, nullptr);
    }
};

class sequence : public object {
public:
    PYBIND11_OBJECT_DEFAULT(sequence, object, PySequence_Check)
    size_t size() const { return (size_t) PySequence_Size(m_ptr); }
    detail::sequence_accessor operator[](size_t index) const { return {*this, index}; }
    detail::item_accessor operator[](handle h) const { return object::operator[](h); }
    detail::sequence_iterator begin() const { return {*this, 0}; }
    detail::sequence_iterator end() const { return {*this, PySequence_Size(m_ptr)}; }
};

class list : public object {
public:
    PYBIND11_OBJECT_CVT(list, object, PyList_Check, PySequence_List)
    explicit list(size_t size = 0) : object(PyList_New((ssize_t) size), stolen_t{}) {
        if (!m_ptr) pybind11_fail("Could not allocate list object!");
    }
    size_t size() const { return (size_t) PyList_Size(m_ptr); }
    detail::list_accessor operator[](size_t index) const { return {*this, index}; }
    detail::item_accessor operator[](handle h) const { return object::operator[](h); }
    detail::list_iterator begin() const { return {*this, 0}; }
    detail::list_iterator end() const { return {*this, PyList_GET_SIZE(m_ptr)}; }
    template <typename T> void append(T &&val) const {
        PyList_Append(m_ptr, detail::object_or_cast(std::forward<T>(val)).ptr());
    }
};

class args : public tuple { PYBIND11_OBJECT_DEFAULT(args, tuple, PyTuple_Check) };
class kwargs : public dict { PYBIND11_OBJECT_DEFAULT(kwargs, dict, PyDict_Check)  };

class set : public object {
public:
    PYBIND11_OBJECT_CVT(set, object, PySet_Check, PySet_New)
    set() : object(PySet_New(nullptr), stolen_t{}) {
        if (!m_ptr) pybind11_fail("Could not allocate set object!");
    }
    size_t size() const { return (size_t) PySet_Size(m_ptr); }
    template <typename T> bool add(T &&val) const {
        return PySet_Add(m_ptr, detail::object_or_cast(std::forward<T>(val)).ptr()) == 0;
    }
    void clear() const { PySet_Clear(m_ptr); }
};

class function : public object {
public:
    PYBIND11_OBJECT_DEFAULT(function, object, PyCallable_Check)
    handle cpp_function() const {
        handle fun = detail::get_function(m_ptr);
        if (fun && PyCFunction_Check(fun.ptr()))
            return fun;
        return handle();
    }
    bool is_cpp_function() const { return (bool) cpp_function(); }
};

class buffer : public object {
public:
    PYBIND11_OBJECT_DEFAULT(buffer, object, PyObject_CheckBuffer)

    buffer_info request(bool writable = false) {
        int flags = PyBUF_STRIDES | PyBUF_FORMAT;
        if (writable) flags |= PyBUF_WRITABLE;
        Py_buffer *view = new Py_buffer();
        if (PyObject_GetBuffer(m_ptr, view, flags) != 0) {
            delete view;
            throw error_already_set();
        }
        return buffer_info(view);
    }
};

class memoryview : public object {
public:
    explicit memoryview(const buffer_info& info) {
        static Py_buffer buf { };
        // Py_buffer uses signed sizes, strides and shape!..
        static std::vector<Py_ssize_t> py_strides { };
        static std::vector<Py_ssize_t> py_shape { };
        buf.buf = info.ptr;
        buf.itemsize = info.itemsize;
        buf.format = const_cast<char *>(info.format.c_str());
        buf.ndim = (int) info.ndim;
        buf.len = info.size;
        py_strides.clear();
        py_shape.clear();
        for (size_t i = 0; i < (size_t) info.ndim; ++i) {
            py_strides.push_back(info.strides[i]);
            py_shape.push_back(info.shape[i]);
        }
        buf.strides = py_strides.data();
        buf.shape = py_shape.data();
        buf.suboffsets = nullptr;
        buf.readonly = false;
        buf.internal = nullptr;

        m_ptr = PyMemoryView_FromBuffer(&buf);
        if (!m_ptr)
            pybind11_fail("Unable to create memoryview from buffer descriptor");
    }

    PYBIND11_OBJECT_CVT(memoryview, object, PyMemoryView_Check, PyMemoryView_FromObject)
};
/// @} pytypes

/// \addtogroup python_builtins
/// @{
inline size_t len(handle h) {
    ssize_t result = PyObject_Length(h.ptr());
    if (result < 0)
        pybind11_fail("Unable to compute length of object");
    return (size_t) result;
}

inline str repr(handle h) {
    PyObject *str_value = PyObject_Repr(h.ptr());
    if (!str_value) throw error_already_set();
#if PY_MAJOR_VERSION < 3
    PyObject *unicode = PyUnicode_FromEncodedObject(str_value, "utf-8", nullptr);
    Py_XDECREF(str_value); str_value = unicode;
    if (!str_value) throw error_already_set();
#endif
    return reinterpret_steal<str>(str_value);
}

inline iterator iter(handle obj) {
    PyObject *result = PyObject_GetIter(obj.ptr());
    if (!result) { throw error_already_set(); }
    return reinterpret_steal<iterator>(result);
}
/// @} python_builtins

NAMESPACE_BEGIN(detail)
template <typename D> iterator object_api<D>::begin() const { return iter(derived()); }
template <typename D> iterator object_api<D>::end() const { return iterator::sentinel(); }
template <typename D> item_accessor object_api<D>::operator[](handle key) const {
    return {derived(), reinterpret_borrow<object>(key)};
}
template <typename D> item_accessor object_api<D>::operator[](const char *key) const {
    return {derived(), pybind11::str(key)};
}
template <typename D> obj_attr_accessor object_api<D>::attr(handle key) const {
    return {derived(), reinterpret_borrow<object>(key)};
}
template <typename D> str_attr_accessor object_api<D>::attr(const char *key) const {
    return {derived(), key};
}
template <typename D> args_proxy object_api<D>::operator*() const {
    return args_proxy(derived().ptr());
}
template <typename D> template <typename T> bool object_api<D>::contains(T &&item) const {
    return attr("__contains__")(std::forward<T>(item)).template cast<bool>();
}

template <typename D>
pybind11::str object_api<D>::str() const { return pybind11::str(derived()); }

template <typename D>
str_attr_accessor object_api<D>::doc() const { return attr("__doc__"); }

template <typename D>
handle object_api<D>::get_type() const { return (PyObject *) Py_TYPE(derived().ptr()); }

template <typename D>
bool object_api<D>::rich_compare(object_api const &other, int value) const {
    int rv = PyObject_RichCompareBool(derived().ptr(), other.derived().ptr(), value);
    if (rv == -1)
        throw error_already_set();
    return rv == 1;
}

#define PYBIND11_MATH_OPERATOR_UNARY(op, fn)                                   \
    template <typename D> object object_api<D>::op() const {                   \
        object result = reinterpret_steal<object>(fn(derived().ptr()));        \
        if (!result.ptr())                                                     \
            throw error_already_set();                                         \
        return result;                                                         \
    }

#define PYBIND11_MATH_OPERATOR_BINARY(op, fn)                                  \
    template <typename D>                                                      \
    object object_api<D>::op(object_api const &other) const {                  \
        object result = reinterpret_steal<object>(                             \
            fn(derived().ptr(), other.derived().ptr()));                       \
        if (!result.ptr())                                                     \
            throw error_already_set();                                         \
        return result;                                                         \
    }

PYBIND11_MATH_OPERATOR_UNARY (operator~,   PyNumber_Invert)
PYBIND11_MATH_OPERATOR_UNARY (operator-,   PyNumber_Negative)
PYBIND11_MATH_OPERATOR_BINARY(operator+,   PyNumber_Add)
PYBIND11_MATH_OPERATOR_BINARY(operator+=,  PyNumber_InPlaceAdd)
PYBIND11_MATH_OPERATOR_BINARY(operator-,   PyNumber_Subtract)
PYBIND11_MATH_OPERATOR_BINARY(operator-=,  PyNumber_InPlaceSubtract)
PYBIND11_MATH_OPERATOR_BINARY(operator*,   PyNumber_Multiply)
PYBIND11_MATH_OPERATOR_BINARY(operator*=,  PyNumber_InPlaceMultiply)
PYBIND11_MATH_OPERATOR_BINARY(operator/,   PyNumber_TrueDivide)
PYBIND11_MATH_OPERATOR_BINARY(operator/=,  PyNumber_InPlaceTrueDivide)
PYBIND11_MATH_OPERATOR_BINARY(operator|,   PyNumber_Or)
PYBIND11_MATH_OPERATOR_BINARY(operator|=,  PyNumber_InPlaceOr)
PYBIND11_MATH_OPERATOR_BINARY(operator&,   PyNumber_And)
PYBIND11_MATH_OPERATOR_BINARY(operator&=,  PyNumber_InPlaceAnd)
PYBIND11_MATH_OPERATOR_BINARY(operator^,   PyNumber_Xor)
PYBIND11_MATH_OPERATOR_BINARY(operator^=,  PyNumber_InPlaceXor)
PYBIND11_MATH_OPERATOR_BINARY(operator<<,  PyNumber_Lshift)
PYBIND11_MATH_OPERATOR_BINARY(operator<<=, PyNumber_InPlaceLshift)
PYBIND11_MATH_OPERATOR_BINARY(operator>>,  PyNumber_Rshift)
PYBIND11_MATH_OPERATOR_BINARY(operator>>=, PyNumber_InPlaceRshift)

#undef PYBIND11_MATH_OPERATOR_UNARY
#undef PYBIND11_MATH_OPERATOR_BINARY

NAMESPACE_END(detail)
NAMESPACE_END(PYBIND11_NAMESPACE)
