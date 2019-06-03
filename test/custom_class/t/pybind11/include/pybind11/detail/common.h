/*
    pybind11/detail/common.h -- Basic macros

    Copyright (c) 2016 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE file.
*/

#pragma once

#if !defined(NAMESPACE_BEGIN)
#  define NAMESPACE_BEGIN(name) namespace name {
#endif
#if !defined(NAMESPACE_END)
#  define NAMESPACE_END(name) }
#endif

// Robust support for some features and loading modules compiled against different pybind versions
// requires forcing hidden visibility on pybind code, so we enforce this by setting the attribute on
// the main `pybind11` namespace.
#if !defined(PYBIND11_NAMESPACE)
#  ifdef __GNUG__
#    define PYBIND11_NAMESPACE pybind11 __attribute__((visibility("hidden")))
#  else
#    define PYBIND11_NAMESPACE pybind11
#  endif
#endif

#if !(defined(_MSC_VER) && __cplusplus == 199711L) && !defined(__INTEL_COMPILER)
#  if __cplusplus >= 201402L
#    define PYBIND11_CPP14
#    if __cplusplus >= 201703L
#      define PYBIND11_CPP17
#    endif
#  endif
#elif defined(_MSC_VER) && __cplusplus == 199711L
// MSVC sets _MSVC_LANG rather than __cplusplus (supposedly until the standard is fully implemented)
// Unless you use the /Zc:__cplusplus flag on Visual Studio 2017 15.7 Preview 3 or newer
#  if _MSVC_LANG >= 201402L
#    define PYBIND11_CPP14
#    if _MSVC_LANG > 201402L && _MSC_VER >= 1910
#      define PYBIND11_CPP17
#    endif
#  endif
#endif

// Compiler version assertions
#if defined(__INTEL_COMPILER)
#  if __INTEL_COMPILER < 1700
#    error pybind11 requires Intel C++ compiler v17 or newer
#  endif
#elif defined(__clang__) && !defined(__apple_build_version__)
#  if __clang_major__ < 3 || (__clang_major__ == 3 && __clang_minor__ < 3)
#    error pybind11 requires clang 3.3 or newer
#  endif
#elif defined(__clang__)
// Apple changes clang version macros to its Xcode version; the first Xcode release based on
// (upstream) clang 3.3 was Xcode 5:
#  if __clang_major__ < 5
#    error pybind11 requires Xcode/clang 5.0 or newer
#  endif
#elif defined(__GNUG__)
#  if __GNUC__ < 4 || (__GNUC__ == 4 && __GNUC_MINOR__ < 8)
#    error pybind11 requires gcc 4.8 or newer
#  endif
#elif defined(_MSC_VER)
// Pybind hits various compiler bugs in 2015u2 and earlier, and also makes use of some stl features
// (e.g. std::negation) added in 2015u3:
#  if _MSC_FULL_VER < 190024210
#    error pybind11 requires MSVC 2015 update 3 or newer
#  endif
#endif

#if !defined(PYBIND11_EXPORT)
#  if defined(WIN32) || defined(_WIN32)
#    define PYBIND11_EXPORT __declspec(dllexport)
#  else
#    define PYBIND11_EXPORT __attribute__ ((visibility("default")))
#  endif
#endif

#if defined(_MSC_VER)
#  define PYBIND11_NOINLINE __declspec(noinline)
#else
#  define PYBIND11_NOINLINE __attribute__ ((noinline))
#endif

#if defined(PYBIND11_CPP14)
#  define PYBIND11_DEPRECATED(reason) [[deprecated(reason)]]
#else
#  define PYBIND11_DEPRECATED(reason) __attribute__((deprecated(reason)))
#endif

#define PYBIND11_VERSION_MAJOR 2
#define PYBIND11_VERSION_MINOR 3
#define PYBIND11_VERSION_PATCH dev0

/// Include Python header, disable linking to pythonX_d.lib on Windows in debug mode
#if defined(_MSC_VER)
#  if (PY_MAJOR_VERSION == 3 && PY_MINOR_VERSION < 4)
#    define HAVE_ROUND 1
#  endif
#  pragma warning(push)
#  pragma warning(disable: 4510 4610 4512 4005)
#  if defined(_DEBUG)
#    define PYBIND11_DEBUG_MARKER
#    undef _DEBUG
#  endif
#endif

#include <Python.h>
#include <frameobject.h>
#include <pythread.h>

#if defined(_WIN32) && (defined(min) || defined(max))
#  error Macro clash with min and max -- define NOMINMAX when compiling your program on Windows
#endif

#if defined(isalnum)
#  undef isalnum
#  undef isalpha
#  undef islower
#  undef isspace
#  undef isupper
#  undef tolower
#  undef toupper
#endif

#if defined(_MSC_VER)
#  if defined(PYBIND11_DEBUG_MARKER)
#    define _DEBUG
#    undef PYBIND11_DEBUG_MARKER
#  endif
#  pragma warning(pop)
#endif

#include <cstddef>
#include <cstring>
#include <forward_list>
#include <vector>
#include <string>
#include <stdexcept>
#include <unordered_set>
#include <unordered_map>
#include <memory>
#include <typeindex>
#include <type_traits>

#if PY_MAJOR_VERSION >= 3 /// Compatibility macros for various Python versions
#define PYBIND11_INSTANCE_METHOD_NEW(ptr, class_) PyInstanceMethod_New(ptr)
#define PYBIND11_INSTANCE_METHOD_CHECK PyInstanceMethod_Check
#define PYBIND11_INSTANCE_METHOD_GET_FUNCTION PyInstanceMethod_GET_FUNCTION
#define PYBIND11_BYTES_CHECK PyBytes_Check
#define PYBIND11_BYTES_FROM_STRING PyBytes_FromString
#define PYBIND11_BYTES_FROM_STRING_AND_SIZE PyBytes_FromStringAndSize
#define PYBIND11_BYTES_AS_STRING_AND_SIZE PyBytes_AsStringAndSize
#define PYBIND11_BYTES_AS_STRING PyBytes_AsString
#define PYBIND11_BYTES_SIZE PyBytes_Size
#define PYBIND11_LONG_CHECK(o) PyLong_Check(o)
#define PYBIND11_LONG_AS_LONGLONG(o) PyLong_AsLongLong(o)
#define PYBIND11_LONG_FROM_SIGNED(o) PyLong_FromSsize_t((ssize_t) o)
#define PYBIND11_LONG_FROM_UNSIGNED(o) PyLong_FromSize_t((size_t) o)
#define PYBIND11_BYTES_NAME "bytes"
#define PYBIND11_STRING_NAME "str"
#define PYBIND11_SLICE_OBJECT PyObject
#define PYBIND11_FROM_STRING PyUnicode_FromString
#define PYBIND11_STR_TYPE ::pybind11::str
#define PYBIND11_BOOL_ATTR "__bool__"
#define PYBIND11_NB_BOOL(ptr) ((ptr)->nb_bool)
#define PYBIND11_PLUGIN_IMPL(name) \
    extern "C" PYBIND11_EXPORT PyObject *PyInit_##name()

#else
#define PYBIND11_INSTANCE_METHOD_NEW(ptr, class_) PyMethod_New(ptr, nullptr, class_)
#define PYBIND11_INSTANCE_METHOD_CHECK PyMethod_Check
#define PYBIND11_INSTANCE_METHOD_GET_FUNCTION PyMethod_GET_FUNCTION
#define PYBIND11_BYTES_CHECK PyString_Check
#define PYBIND11_BYTES_FROM_STRING PyString_FromString
#define PYBIND11_BYTES_FROM_STRING_AND_SIZE PyString_FromStringAndSize
#define PYBIND11_BYTES_AS_STRING_AND_SIZE PyString_AsStringAndSize
#define PYBIND11_BYTES_AS_STRING PyString_AsString
#define PYBIND11_BYTES_SIZE PyString_Size
#define PYBIND11_LONG_CHECK(o) (PyInt_Check(o) || PyLong_Check(o))
#define PYBIND11_LONG_AS_LONGLONG(o) (PyInt_Check(o) ? (long long) PyLong_AsLong(o) : PyLong_AsLongLong(o))
#define PYBIND11_LONG_FROM_SIGNED(o) PyInt_FromSsize_t((ssize_t) o) // Returns long if needed.
#define PYBIND11_LONG_FROM_UNSIGNED(o) PyInt_FromSize_t((size_t) o) // Returns long if needed.
#define PYBIND11_BYTES_NAME "str"
#define PYBIND11_STRING_NAME "unicode"
#define PYBIND11_SLICE_OBJECT PySliceObject
#define PYBIND11_FROM_STRING PyString_FromString
#define PYBIND11_STR_TYPE ::pybind11::bytes
#define PYBIND11_BOOL_ATTR "__nonzero__"
#define PYBIND11_NB_BOOL(ptr) ((ptr)->nb_nonzero)
#define PYBIND11_PLUGIN_IMPL(name) \
    static PyObject *pybind11_init_wrapper();               \
    extern "C" PYBIND11_EXPORT void init##name() {          \
        (void)pybind11_init_wrapper();                      \
    }                                                       \
    PyObject *pybind11_init_wrapper()
#endif

#if PY_VERSION_HEX >= 0x03050000 && PY_VERSION_HEX < 0x03050200
extern "C" {
    struct _Py_atomic_address { void *value; };
    PyAPI_DATA(_Py_atomic_address) _PyThreadState_Current;
}
#endif

#define PYBIND11_TRY_NEXT_OVERLOAD ((PyObject *) 1) // special failure return code
#define PYBIND11_STRINGIFY(x) #x
#define PYBIND11_TOSTRING(x) PYBIND11_STRINGIFY(x)
#define PYBIND11_CONCAT(first, second) first##second

#define PYBIND11_CHECK_PYTHON_VERSION \
    {                                                                          \
        const char *compiled_ver = PYBIND11_TOSTRING(PY_MAJOR_VERSION)         \
            "." PYBIND11_TOSTRING(PY_MINOR_VERSION);                           \
        const char *runtime_ver = Py_GetVersion();                             \
        size_t len = std::strlen(compiled_ver);                                \
        if (std::strncmp(runtime_ver, compiled_ver, len) != 0                  \
                || (runtime_ver[len] >= '0' && runtime_ver[len] <= '9')) {     \
            PyErr_Format(PyExc_ImportError,                                    \
                "Python version mismatch: module was compiled for Python %s, " \
                "but the interpreter version is incompatible: %s.",            \
                compiled_ver, runtime_ver);                                    \
            return nullptr;                                                    \
        }                                                                      \
    }

#define PYBIND11_CATCH_INIT_EXCEPTIONS \
        catch (pybind11::error_already_set &e) {                               \
            PyErr_SetString(PyExc_ImportError, e.what());                      \
            return nullptr;                                                    \
        } catch (const std::exception &e) {                                    \
            PyErr_SetString(PyExc_ImportError, e.what());                      \
            return nullptr;                                                    \
        }                                                                      \

/** \rst
    ***Deprecated in favor of PYBIND11_MODULE***

    This macro creates the entry point that will be invoked when the Python interpreter
    imports a plugin library. Please create a `module` in the function body and return
    the pointer to its underlying Python object at the end.

    .. code-block:: cpp

        PYBIND11_PLUGIN(example) {
            pybind11::module m("example", "pybind11 example plugin");
            /// Set up bindings here
            return m.ptr();
        }
\endrst */
#define PYBIND11_PLUGIN(name)                                                  \
    PYBIND11_DEPRECATED("PYBIND11_PLUGIN is deprecated, use PYBIND11_MODULE")  \
    static PyObject *pybind11_init();                                          \
    PYBIND11_PLUGIN_IMPL(name) {                                               \
        PYBIND11_CHECK_PYTHON_VERSION                                          \
        try {                                                                  \
            return pybind11_init();                                            \
        } PYBIND11_CATCH_INIT_EXCEPTIONS                                       \
    }                                                                          \
    PyObject *pybind11_init()

/** \rst
    This macro creates the entry point that will be invoked when the Python interpreter
    imports an extension module. The module name is given as the fist argument and it
    should not be in quotes. The second macro argument defines a variable of type
    `py::module` which can be used to initialize the module.

    .. code-block:: cpp

        PYBIND11_MODULE(example, m) {
            m.doc() = "pybind11 example module";

            // Add bindings here
            m.def("foo", []() {
                return "Hello, World!";
            });
        }
\endrst */
#define PYBIND11_MODULE(name, variable)                                        \
    static void PYBIND11_CONCAT(pybind11_init_, name)(pybind11::module &);     \
    PYBIND11_PLUGIN_IMPL(name) {                                               \
        PYBIND11_CHECK_PYTHON_VERSION                                          \
        auto m = pybind11::module(PYBIND11_TOSTRING(name));                    \
        try {                                                                  \
            PYBIND11_CONCAT(pybind11_init_, name)(m);                          \
            return m.ptr();                                                    \
        } PYBIND11_CATCH_INIT_EXCEPTIONS                                       \
    }                                                                          \
    void PYBIND11_CONCAT(pybind11_init_, name)(pybind11::module &variable)


NAMESPACE_BEGIN(PYBIND11_NAMESPACE)

using ssize_t = Py_ssize_t;
using size_t  = std::size_t;

/// Approach used to cast a previously unknown C++ instance into a Python object
enum class return_value_policy : uint8_t {
    /** This is the default return value policy, which falls back to the policy
        return_value_policy::take_ownership when the return value is a pointer.
        Otherwise, it uses return_value::move or return_value::copy for rvalue
        and lvalue references, respectively. See below for a description of what
        all of these different policies do. */
    automatic = 0,

    /** As above, but use policy return_value_policy::reference when the return
        value is a pointer. This is the default conversion policy for function
        arguments when calling Python functions manually from C++ code (i.e. via
        handle::operator()). You probably won't need to use this. */
    automatic_reference,

    /** Reference an existing object (i.e. do not create a new copy) and take
        ownership. Python will call the destructor and delete operator when the
        object’s reference count reaches zero. Undefined behavior ensues when
        the C++ side does the same.. */
    take_ownership,

    /** Create a new copy of the returned object, which will be owned by
        Python. This policy is comparably safe because the lifetimes of the two
        instances are decoupled. */
    copy,

    /** Use std::move to move the return value contents into a new instance
        that will be owned by Python. This policy is comparably safe because the
        lifetimes of the two instances (move source and destination) are
        decoupled. */
    move,

    /** Reference an existing object, but do not take ownership. The C++ side
        is responsible for managing the object’s lifetime and deallocating it
        when it is no longer used. Warning: undefined behavior will ensue when
        the C++ side deletes an object that is still referenced and used by
        Python. */
    reference,

    /** This policy only applies to methods and properties. It references the
        object without taking ownership similar to the above
        return_value_policy::reference policy. In contrast to that policy, the
        function or property’s implicit this argument (called the parent) is
        considered to be the the owner of the return value (the child).
        pybind11 then couples the lifetime of the parent to the child via a
        reference relationship that ensures that the parent cannot be garbage
        collected while Python is still using the child. More advanced
        variations of this scheme are also possible using combinations of
        return_value_policy::reference and the keep_alive call policy */
    reference_internal
};

NAMESPACE_BEGIN(detail)

inline static constexpr int log2(size_t n, int k = 0) { return (n <= 1) ? k : log2(n >> 1, k + 1); }

// Returns the size as a multiple of sizeof(void *), rounded up.
inline static constexpr size_t size_in_ptrs(size_t s) { return 1 + ((s - 1) >> log2(sizeof(void *))); }

/**
 * The space to allocate for simple layout instance holders (see below) in multiple of the size of
 * a pointer (e.g.  2 means 16 bytes on 64-bit architectures).  The default is the minimum required
 * to holder either a std::unique_ptr or std::shared_ptr (which is almost always
 * sizeof(std::shared_ptr<T>)).
 */
constexpr size_t instance_simple_holder_in_ptrs() {
    static_assert(sizeof(std::shared_ptr<int>) >= sizeof(std::unique_ptr<int>),
            "pybind assumes std::shared_ptrs are at least as big as std::unique_ptrs");
    return size_in_ptrs(sizeof(std::shared_ptr<int>));
}

// Forward declarations
struct type_info;
struct value_and_holder;

struct nonsimple_values_and_holders {
    void **values_and_holders;
    uint8_t *status;
};

/// The 'instance' type which needs to be standard layout (need to be able to use 'offsetof')
struct instance {
    PyObject_HEAD
    /// Storage for pointers and holder; see simple_layout, below, for a description
    union {
        void *simple_value_holder[1 + instance_simple_holder_in_ptrs()];
        nonsimple_values_and_holders nonsimple;
    };
    /// Weak references
    PyObject *weakrefs;
    /// If true, the pointer is owned which means we're free to manage it with a holder.
    bool owned : 1;
    /**
     * An instance has two possible value/holder layouts.
     *
     * Simple layout (when this flag is true), means the `simple_value_holder` is set with a pointer
     * and the holder object governing that pointer, i.e. [val1*][holder].  This layout is applied
     * whenever there is no python-side multiple inheritance of bound C++ types *and* the type's
     * holder will fit in the default space (which is large enough to hold either a std::unique_ptr
     * or std::shared_ptr).
     *
     * Non-simple layout applies when using custom holders that require more space than `shared_ptr`
     * (which is typically the size of two pointers), or when multiple inheritance is used on the
     * python side.  Non-simple layout allocates the required amount of memory to have multiple
     * bound C++ classes as parents.  Under this layout, `nonsimple.values_and_holders` is set to a
     * pointer to allocated space of the required space to hold a sequence of value pointers and
     * holders followed `status`, a set of bit flags (1 byte each), i.e.
     * [val1*][holder1][val2*][holder2]...[bb...]  where each [block] is rounded up to a multiple of
     * `sizeof(void *)`.  `nonsimple.status` is, for convenience, a pointer to the
     * beginning of the [bb...] block (but not independently allocated).
     *
     * Status bits indicate whether the associated holder is constructed (&
     * status_holder_constructed) and whether the value pointer is registered (&
     * status_instance_registered) in `registered_instances`.
     */
    bool simple_layout : 1;
    /// For simple layout, tracks whether the holder has been constructed
    bool simple_holder_constructed : 1;
    /// For simple layout, tracks whether the instance is registered in `registered_instances`
    bool simple_instance_registered : 1;
    /// If true, get_internals().patients has an entry for this object
    bool has_patients : 1;

    /// Initializes all of the above type/values/holders data (but not the instance values themselves)
    void allocate_layout();

    /// Destroys/deallocates all of the above
    void deallocate_layout();

    /// Returns the value_and_holder wrapper for the given type (or the first, if `find_type`
    /// omitted).  Returns a default-constructed (with `.inst = nullptr`) object on failure if
    /// `throw_if_missing` is false.
    value_and_holder get_value_and_holder(const type_info *find_type = nullptr, bool throw_if_missing = true);

    /// Bit values for the non-simple status flags
    static constexpr uint8_t status_holder_constructed  = 1;
    static constexpr uint8_t status_instance_registered = 2;
};

static_assert(std::is_standard_layout<instance>::value, "Internal error: `pybind11::detail::instance` is not standard layout!");

/// from __cpp_future__ import (convenient aliases from C++14/17)
#if defined(PYBIND11_CPP14) && (!defined(_MSC_VER) || _MSC_VER >= 1910)
using std::enable_if_t;
using std::conditional_t;
using std::remove_cv_t;
using std::remove_reference_t;
#else
template <bool B, typename T = void> using enable_if_t = typename std::enable_if<B, T>::type;
template <bool B, typename T, typename F> using conditional_t = typename std::conditional<B, T, F>::type;
template <typename T> using remove_cv_t = typename std::remove_cv<T>::type;
template <typename T> using remove_reference_t = typename std::remove_reference<T>::type;
#endif

/// Index sequences
#if defined(PYBIND11_CPP14)
using std::index_sequence;
using std::make_index_sequence;
#else
template<size_t ...> struct index_sequence  { };
template<size_t N, size_t ...S> struct make_index_sequence_impl : make_index_sequence_impl <N - 1, N - 1, S...> { };
template<size_t ...S> struct make_index_sequence_impl <0, S...> { typedef index_sequence<S...> type; };
template<size_t N> using make_index_sequence = typename make_index_sequence_impl<N>::type;
#endif

/// Make an index sequence of the indices of true arguments
template <typename ISeq, size_t, bool...> struct select_indices_impl { using type = ISeq; };
template <size_t... IPrev, size_t I, bool B, bool... Bs> struct select_indices_impl<index_sequence<IPrev...>, I, B, Bs...>
    : select_indices_impl<conditional_t<B, index_sequence<IPrev..., I>, index_sequence<IPrev...>>, I + 1, Bs...> {};
template <bool... Bs> using select_indices = typename select_indices_impl<index_sequence<>, 0, Bs...>::type;

/// Backports of std::bool_constant and std::negation to accommodate older compilers
template <bool B> using bool_constant = std::integral_constant<bool, B>;
template <typename T> struct negation : bool_constant<!T::value> { };

template <typename...> struct void_t_impl { using type = void; };
template <typename... Ts> using void_t = typename void_t_impl<Ts...>::type;

/// Compile-time all/any/none of that check the boolean value of all template types
#if defined(__cpp_fold_expressions) && !(defined(_MSC_VER) && (_MSC_VER < 1916))
template <class... Ts> using all_of = bool_constant<(Ts::value && ...)>;
template <class... Ts> using any_of = bool_constant<(Ts::value || ...)>;
#elif !defined(_MSC_VER)
template <bool...> struct bools {};
template <class... Ts> using all_of = std::is_same<
    bools<Ts::value..., true>,
    bools<true, Ts::value...>>;
template <class... Ts> using any_of = negation<all_of<negation<Ts>...>>;
#else
// MSVC has trouble with the above, but supports std::conjunction, which we can use instead (albeit
// at a slight loss of compilation efficiency).
template <class... Ts> using all_of = std::conjunction<Ts...>;
template <class... Ts> using any_of = std::disjunction<Ts...>;
#endif
template <class... Ts> using none_of = negation<any_of<Ts...>>;

template <class T, template<class> class... Predicates> using satisfies_all_of = all_of<Predicates<T>...>;
template <class T, template<class> class... Predicates> using satisfies_any_of = any_of<Predicates<T>...>;
template <class T, template<class> class... Predicates> using satisfies_none_of = none_of<Predicates<T>...>;

/// Strip the class from a method type
template <typename T> struct remove_class { };
template <typename C, typename R, typename... A> struct remove_class<R (C::*)(A...)> { typedef R type(A...); };
template <typename C, typename R, typename... A> struct remove_class<R (C::*)(A...) const> { typedef R type(A...); };

/// Helper template to strip away type modifiers
template <typename T> struct intrinsic_type                       { typedef T type; };
template <typename T> struct intrinsic_type<const T>              { typedef typename intrinsic_type<T>::type type; };
template <typename T> struct intrinsic_type<T*>                   { typedef typename intrinsic_type<T>::type type; };
template <typename T> struct intrinsic_type<T&>                   { typedef typename intrinsic_type<T>::type type; };
template <typename T> struct intrinsic_type<T&&>                  { typedef typename intrinsic_type<T>::type type; };
template <typename T, size_t N> struct intrinsic_type<const T[N]> { typedef typename intrinsic_type<T>::type type; };
template <typename T, size_t N> struct intrinsic_type<T[N]>       { typedef typename intrinsic_type<T>::type type; };
template <typename T> using intrinsic_t = typename intrinsic_type<T>::type;

/// Helper type to replace 'void' in some expressions
struct void_type { };

/// Helper template which holds a list of types
template <typename...> struct type_list { };

/// Compile-time integer sum
#ifdef __cpp_fold_expressions
template <typename... Ts> constexpr size_t constexpr_sum(Ts... ns) { return (0 + ... + size_t{ns}); }
#else
constexpr size_t constexpr_sum() { return 0; }
template <typename T, typename... Ts>
constexpr size_t constexpr_sum(T n, Ts... ns) { return size_t{n} + constexpr_sum(ns...); }
#endif

NAMESPACE_BEGIN(constexpr_impl)
/// Implementation details for constexpr functions
constexpr int first(int i) { return i; }
template <typename T, typename... Ts>
constexpr int first(int i, T v, Ts... vs) { return v ? i : first(i + 1, vs...); }

constexpr int last(int /*i*/, int result) { return result; }
template <typename T, typename... Ts>
constexpr int last(int i, int result, T v, Ts... vs) { return last(i + 1, v ? i : result, vs...); }
NAMESPACE_END(constexpr_impl)

/// Return the index of the first type in Ts which satisfies Predicate<T>.  Returns sizeof...(Ts) if
/// none match.
template <template<typename> class Predicate, typename... Ts>
constexpr int constexpr_first() { return constexpr_impl::first(0, Predicate<Ts>::value...); }

/// Return the index of the last type in Ts which satisfies Predicate<T>, or -1 if none match.
template <template<typename> class Predicate, typename... Ts>
constexpr int constexpr_last() { return constexpr_impl::last(0, -1, Predicate<Ts>::value...); }

/// Return the Nth element from the parameter pack
template <size_t N, typename T, typename... Ts>
struct pack_element { using type = typename pack_element<N - 1, Ts...>::type; };
template <typename T, typename... Ts>
struct pack_element<0, T, Ts...> { using type = T; };

/// Return the one and only type which matches the predicate, or Default if none match.
/// If more than one type matches the predicate, fail at compile-time.
template <template<typename> class Predicate, typename Default, typename... Ts>
struct exactly_one {
    static constexpr auto found = constexpr_sum(Predicate<Ts>::value...);
    static_assert(found <= 1, "Found more than one type matching the predicate");

    static constexpr auto index = found ? constexpr_first<Predicate, Ts...>() : 0;
    using type = conditional_t<found, typename pack_element<index, Ts...>::type, Default>;
};
template <template<typename> class P, typename Default>
struct exactly_one<P, Default> { using type = Default; };

template <template<typename> class Predicate, typename Default, typename... Ts>
using exactly_one_t = typename exactly_one<Predicate, Default, Ts...>::type;

/// Defer the evaluation of type T until types Us are instantiated
template <typename T, typename... /*Us*/> struct deferred_type { using type = T; };
template <typename T, typename... Us> using deferred_t = typename deferred_type<T, Us...>::type;

/// Like is_base_of, but requires a strict base (i.e. `is_strict_base_of<T, T>::value == false`,
/// unlike `std::is_base_of`)
template <typename Base, typename Derived> using is_strict_base_of = bool_constant<
    std::is_base_of<Base, Derived>::value && !std::is_same<Base, Derived>::value>;

/// Like is_base_of, but also requires that the base type is accessible (i.e. that a Derived pointer
/// can be converted to a Base pointer)
template <typename Base, typename Derived> using is_accessible_base_of = bool_constant<
    std::is_base_of<Base, Derived>::value && std::is_convertible<Derived *, Base *>::value>;

template <template<typename...> class Base>
struct is_template_base_of_impl {
    template <typename... Us> static std::true_type check(Base<Us...> *);
    static std::false_type check(...);
};

/// Check if a template is the base of a type. For example:
/// `is_template_base_of<Base, T>` is true if `struct T : Base<U> {}` where U can be anything
template <template<typename...> class Base, typename T>
#if !defined(_MSC_VER)
using is_template_base_of = decltype(is_template_base_of_impl<Base>::check((intrinsic_t<T>*)nullptr));
#else // MSVC2015 has trouble with decltype in template aliases
struct is_template_base_of : decltype(is_template_base_of_impl<Base>::check((intrinsic_t<T>*)nullptr)) { };
#endif

/// Check if T is an instantiation of the template `Class`. For example:
/// `is_instantiation<shared_ptr, T>` is true if `T == shared_ptr<U>` where U can be anything.
template <template<typename...> class Class, typename T>
struct is_instantiation : std::false_type { };
template <template<typename...> class Class, typename... Us>
struct is_instantiation<Class, Class<Us...>> : std::true_type { };

/// Check if T is std::shared_ptr<U> where U can be anything
template <typename T> using is_shared_ptr = is_instantiation<std::shared_ptr, T>;

/// Check if T looks like an input iterator
template <typename T, typename = void> struct is_input_iterator : std::false_type {};
template <typename T>
struct is_input_iterator<T, void_t<decltype(*std::declval<T &>()), decltype(++std::declval<T &>())>>
    : std::true_type {};

template <typename T> using is_function_pointer = bool_constant<
    std::is_pointer<T>::value && std::is_function<typename std::remove_pointer<T>::type>::value>;

template <typename F> struct strip_function_object {
    using type = typename remove_class<decltype(&F::operator())>::type;
};

// Extracts the function signature from a function, function pointer or lambda.
template <typename Function, typename F = remove_reference_t<Function>>
using function_signature_t = conditional_t<
    std::is_function<F>::value,
    F,
    typename conditional_t<
        std::is_pointer<F>::value || std::is_member_pointer<F>::value,
        std::remove_pointer<F>,
        strip_function_object<F>
    >::type
>;

/// Returns true if the type looks like a lambda: that is, isn't a function, pointer or member
/// pointer.  Note that this can catch all sorts of other things, too; this is intended to be used
/// in a place where passing a lambda makes sense.
template <typename T> using is_lambda = satisfies_none_of<remove_reference_t<T>,
        std::is_function, std::is_pointer, std::is_member_pointer>;

/// Ignore that a variable is unused in compiler warnings
inline void ignore_unused(const int *) { }

/// Apply a function over each element of a parameter pack
#ifdef __cpp_fold_expressions
#define PYBIND11_EXPAND_SIDE_EFFECTS(PATTERN) (((PATTERN), void()), ...)
#else
using expand_side_effects = bool[];
#define PYBIND11_EXPAND_SIDE_EFFECTS(PATTERN) pybind11::detail::expand_side_effects{ ((PATTERN), void(), false)..., false }
#endif

NAMESPACE_END(detail)

/// C++ bindings of builtin Python exceptions
class builtin_exception : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
    /// Set the error using the Python C API
    virtual void set_error() const = 0;
};

#define PYBIND11_RUNTIME_EXCEPTION(name, type) \
    class name : public builtin_exception { public: \
        using builtin_exception::builtin_exception; \
        name() : name("") { } \
        void set_error() const override { PyErr_SetString(type, what()); } \
    };

PYBIND11_RUNTIME_EXCEPTION(stop_iteration, PyExc_StopIteration)
PYBIND11_RUNTIME_EXCEPTION(index_error, PyExc_IndexError)
PYBIND11_RUNTIME_EXCEPTION(key_error, PyExc_KeyError)
PYBIND11_RUNTIME_EXCEPTION(value_error, PyExc_ValueError)
PYBIND11_RUNTIME_EXCEPTION(type_error, PyExc_TypeError)
PYBIND11_RUNTIME_EXCEPTION(cast_error, PyExc_RuntimeError) /// Thrown when pybind11::cast or handle::call fail due to a type casting error
PYBIND11_RUNTIME_EXCEPTION(reference_cast_error, PyExc_RuntimeError) /// Used internally

[[noreturn]] PYBIND11_NOINLINE inline void pybind11_fail(const char *reason) { throw std::runtime_error(reason); }
[[noreturn]] PYBIND11_NOINLINE inline void pybind11_fail(const std::string &reason) { throw std::runtime_error(reason); }

template <typename T, typename SFINAE = void> struct format_descriptor { };

NAMESPACE_BEGIN(detail)
// Returns the index of the given type in the type char array below, and in the list in numpy.h
// The order here is: bool; 8 ints ((signed,unsigned)x(8,16,32,64)bits); float,double,long double;
// complex float,double,long double.  Note that the long double types only participate when long
// double is actually longer than double (it isn't under MSVC).
// NB: not only the string below but also complex.h and numpy.h rely on this order.
template <typename T, typename SFINAE = void> struct is_fmt_numeric { static constexpr bool value = false; };
template <typename T> struct is_fmt_numeric<T, enable_if_t<std::is_arithmetic<T>::value>> {
    static constexpr bool value = true;
    static constexpr int index = std::is_same<T, bool>::value ? 0 : 1 + (
        std::is_integral<T>::value ? detail::log2(sizeof(T))*2 + std::is_unsigned<T>::value : 8 + (
        std::is_same<T, double>::value ? 1 : std::is_same<T, long double>::value ? 2 : 0));
};
NAMESPACE_END(detail)

template <typename T> struct format_descriptor<T, detail::enable_if_t<std::is_arithmetic<T>::value>> {
    static constexpr const char c = "?bBhHiIqQfdg"[detail::is_fmt_numeric<T>::index];
    static constexpr const char value[2] = { c, '\0' };
    static std::string format() { return std::string(1, c); }
};

#if !defined(PYBIND11_CPP17)

template <typename T> constexpr const char format_descriptor<
    T, detail::enable_if_t<std::is_arithmetic<T>::value>>::value[2];

#endif

/// RAII wrapper that temporarily clears any Python error state
struct error_scope {
    PyObject *type, *value, *trace;
    error_scope() { PyErr_Fetch(&type, &value, &trace); }
    ~error_scope() { PyErr_Restore(type, value, trace); }
};

/// Dummy destructor wrapper that can be used to expose classes with a private destructor
struct nodelete { template <typename T> void operator()(T*) { } };

// overload_cast requires variable templates: C++14
#if defined(PYBIND11_CPP14)
#define PYBIND11_OVERLOAD_CAST 1

NAMESPACE_BEGIN(detail)
template <typename... Args>
struct overload_cast_impl {
    constexpr overload_cast_impl() {} // MSVC 2015 needs this

    template <typename Return>
    constexpr auto operator()(Return (*pf)(Args...)) const noexcept
                              -> decltype(pf) { return pf; }

    template <typename Return, typename Class>
    constexpr auto operator()(Return (Class::*pmf)(Args...), std::false_type = {}) const noexcept
                              -> decltype(pmf) { return pmf; }

    template <typename Return, typename Class>
    constexpr auto operator()(Return (Class::*pmf)(Args...) const, std::true_type) const noexcept
                              -> decltype(pmf) { return pmf; }
};
NAMESPACE_END(detail)

/// Syntax sugar for resolving overloaded function pointers:
///  - regular: static_cast<Return (Class::*)(Arg0, Arg1, Arg2)>(&Class::func)
///  - sweet:   overload_cast<Arg0, Arg1, Arg2>(&Class::func)
template <typename... Args>
static constexpr detail::overload_cast_impl<Args...> overload_cast = {};
// MSVC 2015 only accepts this particular initialization syntax for this variable template.

/// Const member function selector for overload_cast
///  - regular: static_cast<Return (Class::*)(Arg) const>(&Class::func)
///  - sweet:   overload_cast<Arg>(&Class::func, const_)
static constexpr auto const_ = std::true_type{};

#else // no overload_cast: providing something that static_assert-fails:
template <typename... Args> struct overload_cast {
    static_assert(detail::deferred_t<std::false_type, Args...>::value,
                  "pybind11::overload_cast<...> requires compiling in C++14 mode");
};
#endif // overload_cast

NAMESPACE_BEGIN(detail)

// Adaptor for converting arbitrary container arguments into a vector; implicitly convertible from
// any standard container (or C-style array) supporting std::begin/std::end, any singleton
// arithmetic type (if T is arithmetic), or explicitly constructible from an iterator pair.
template <typename T>
class any_container {
    std::vector<T> v;
public:
    any_container() = default;

    // Can construct from a pair of iterators
    template <typename It, typename = enable_if_t<is_input_iterator<It>::value>>
    any_container(It first, It last) : v(first, last) { }

    // Implicit conversion constructor from any arbitrary container type with values convertible to T
    template <typename Container, typename = enable_if_t<std::is_convertible<decltype(*std::begin(std::declval<const Container &>())), T>::value>>
    any_container(const Container &c) : any_container(std::begin(c), std::end(c)) { }

    // initializer_list's aren't deducible, so don't get matched by the above template; we need this
    // to explicitly allow implicit conversion from one:
    template <typename TIn, typename = enable_if_t<std::is_convertible<TIn, T>::value>>
    any_container(const std::initializer_list<TIn> &c) : any_container(c.begin(), c.end()) { }

    // Avoid copying if given an rvalue vector of the correct type.
    any_container(std::vector<T> &&v) : v(std::move(v)) { }

    // Moves the vector out of an rvalue any_container
    operator std::vector<T> &&() && { return std::move(v); }

    // Dereferencing obtains a reference to the underlying vector
    std::vector<T> &operator*() { return v; }
    const std::vector<T> &operator*() const { return v; }

    // -> lets you call methods on the underlying vector
    std::vector<T> *operator->() { return &v; }
    const std::vector<T> *operator->() const { return &v; }
};

NAMESPACE_END(detail)



NAMESPACE_END(PYBIND11_NAMESPACE)
