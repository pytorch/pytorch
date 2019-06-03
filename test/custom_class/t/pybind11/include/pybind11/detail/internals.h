/*
    pybind11/detail/internals.h: Internal data structure and related functions

    Copyright (c) 2017 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE file.
*/

#pragma once

#include "../pytypes.h"

NAMESPACE_BEGIN(PYBIND11_NAMESPACE)
NAMESPACE_BEGIN(detail)
// Forward declarations
inline PyTypeObject *make_static_property_type();
inline PyTypeObject *make_default_metaclass();
inline PyObject *make_object_base_type(PyTypeObject *metaclass);

// The old Python Thread Local Storage (TLS) API is deprecated in Python 3.7 in favor of the new
// Thread Specific Storage (TSS) API.
#if PY_VERSION_HEX >= 0x03070000
#    define PYBIND11_TLS_KEY_INIT(var) Py_tss_t *var = nullptr
#    define PYBIND11_TLS_GET_VALUE(key) PyThread_tss_get((key))
#    define PYBIND11_TLS_REPLACE_VALUE(key, value) PyThread_tss_set((key), (tstate))
#    define PYBIND11_TLS_DELETE_VALUE(key) PyThread_tss_set((key), nullptr)
#else
    // Usually an int but a long on Cygwin64 with Python 3.x
#    define PYBIND11_TLS_KEY_INIT(var) decltype(PyThread_create_key()) var = 0
#    define PYBIND11_TLS_GET_VALUE(key) PyThread_get_key_value((key))
#    if PY_MAJOR_VERSION < 3
#        define PYBIND11_TLS_DELETE_VALUE(key)                               \
             PyThread_delete_key_value(key)
#        define PYBIND11_TLS_REPLACE_VALUE(key, value)                       \
             do {                                                            \
                 PyThread_delete_key_value((key));                           \
                 PyThread_set_key_value((key), (value));                     \
             } while (false)
#    else
#        define PYBIND11_TLS_DELETE_VALUE(key)                               \
             PyThread_set_key_value((key), nullptr)
#        define PYBIND11_TLS_REPLACE_VALUE(key, value)                       \
             PyThread_set_key_value((key), (value))
#    endif
#endif

// Python loads modules by default with dlopen with the RTLD_LOCAL flag; under libc++ and possibly
// other STLs, this means `typeid(A)` from one module won't equal `typeid(A)` from another module
// even when `A` is the same, non-hidden-visibility type (e.g. from a common include).  Under
// libstdc++, this doesn't happen: equality and the type_index hash are based on the type name,
// which works.  If not under a known-good stl, provide our own name-based hash and equality
// functions that use the type name.
#if defined(__GLIBCXX__)
inline bool same_type(const std::type_info &lhs, const std::type_info &rhs) { return lhs == rhs; }
using type_hash = std::hash<std::type_index>;
using type_equal_to = std::equal_to<std::type_index>;
#else
inline bool same_type(const std::type_info &lhs, const std::type_info &rhs) {
    return lhs.name() == rhs.name() || std::strcmp(lhs.name(), rhs.name()) == 0;
}

struct type_hash {
    size_t operator()(const std::type_index &t) const {
        size_t hash = 5381;
        const char *ptr = t.name();
        while (auto c = static_cast<unsigned char>(*ptr++))
            hash = (hash * 33) ^ c;
        return hash;
    }
};

struct type_equal_to {
    bool operator()(const std::type_index &lhs, const std::type_index &rhs) const {
        return lhs.name() == rhs.name() || std::strcmp(lhs.name(), rhs.name()) == 0;
    }
};
#endif

template <typename value_type>
using type_map = std::unordered_map<std::type_index, value_type, type_hash, type_equal_to>;

struct overload_hash {
    inline size_t operator()(const std::pair<const PyObject *, const char *>& v) const {
        size_t value = std::hash<const void *>()(v.first);
        value ^= std::hash<const void *>()(v.second)  + 0x9e3779b9 + (value<<6) + (value>>2);
        return value;
    }
};

/// Internal data structure used to track registered instances and types.
/// Whenever binary incompatible changes are made to this structure,
/// `PYBIND11_INTERNALS_VERSION` must be incremented.
struct internals {
    type_map<type_info *> registered_types_cpp; // std::type_index -> pybind11's type information
    std::unordered_map<PyTypeObject *, std::vector<type_info *>> registered_types_py; // PyTypeObject* -> base type_info(s)
    std::unordered_multimap<const void *, instance*> registered_instances; // void * -> instance*
    std::unordered_set<std::pair<const PyObject *, const char *>, overload_hash> inactive_overload_cache;
    type_map<std::vector<bool (*)(PyObject *, void *&)>> direct_conversions;
    std::unordered_map<const PyObject *, std::vector<PyObject *>> patients;
    std::forward_list<void (*) (std::exception_ptr)> registered_exception_translators;
    std::unordered_map<std::string, void *> shared_data; // Custom data to be shared across extensions
    std::vector<PyObject *> loader_patient_stack; // Used by `loader_life_support`
    std::forward_list<std::string> static_strings; // Stores the std::strings backing detail::c_str()
    PyTypeObject *static_property_type;
    PyTypeObject *default_metaclass;
    PyObject *instance_base;
#if defined(WITH_THREAD)
    PYBIND11_TLS_KEY_INIT(tstate);
    PyInterpreterState *istate = nullptr;
#endif
};

/// Additional type information which does not fit into the PyTypeObject.
/// Changes to this struct also require bumping `PYBIND11_INTERNALS_VERSION`.
struct type_info {
    PyTypeObject *type;
    const std::type_info *cpptype;
    size_t type_size, type_align, holder_size_in_ptrs;
    void *(*operator_new)(size_t);
    void (*init_instance)(instance *, const void *);
    void (*dealloc)(value_and_holder &v_h);
    std::vector<PyObject *(*)(PyObject *, PyTypeObject *)> implicit_conversions;
    std::vector<std::pair<const std::type_info *, void *(*)(void *)>> implicit_casts;
    std::vector<bool (*)(PyObject *, void *&)> *direct_conversions;
    buffer_info *(*get_buffer)(PyObject *, void *) = nullptr;
    void *get_buffer_data = nullptr;
    void *(*module_local_load)(PyObject *, const type_info *) = nullptr;
    /* A simple type never occurs as a (direct or indirect) parent
     * of a class that makes use of multiple inheritance */
    bool simple_type : 1;
    /* True if there is no multiple inheritance in this type's inheritance tree */
    bool simple_ancestors : 1;
    /* for base vs derived holder_type checks */
    bool default_holder : 1;
    /* true if this is a type registered with py::module_local */
    bool module_local : 1;
};

/// Tracks the `internals` and `type_info` ABI version independent of the main library version
#define PYBIND11_INTERNALS_VERSION 3

#if defined(_DEBUG)
#   define PYBIND11_BUILD_TYPE "_debug"
#else
#   define PYBIND11_BUILD_TYPE ""
#endif

#if defined(WITH_THREAD)
#  define PYBIND11_INTERNALS_KIND ""
#else
#  define PYBIND11_INTERNALS_KIND "_without_thread"
#endif

#define PYBIND11_INTERNALS_ID "__pybind11_internals_v" \
    PYBIND11_TOSTRING(PYBIND11_INTERNALS_VERSION) PYBIND11_INTERNALS_KIND PYBIND11_BUILD_TYPE "__"

#define PYBIND11_MODULE_LOCAL_ID "__pybind11_module_local_v" \
    PYBIND11_TOSTRING(PYBIND11_INTERNALS_VERSION) PYBIND11_INTERNALS_KIND PYBIND11_BUILD_TYPE "__"

/// Each module locally stores a pointer to the `internals` data. The data
/// itself is shared among modules with the same `PYBIND11_INTERNALS_ID`.
inline internals **&get_internals_pp() {
    static internals **internals_pp = nullptr;
    return internals_pp;
}

/// Return a reference to the current `internals` data
PYBIND11_NOINLINE inline internals &get_internals() {
    auto **&internals_pp = get_internals_pp();
    if (internals_pp && *internals_pp)
        return **internals_pp;

    constexpr auto *id = PYBIND11_INTERNALS_ID;
    auto builtins = handle(PyEval_GetBuiltins());
    if (builtins.contains(id) && isinstance<capsule>(builtins[id])) {
        internals_pp = static_cast<internals **>(capsule(builtins[id]));

        // We loaded builtins through python's builtins, which means that our `error_already_set`
        // and `builtin_exception` may be different local classes than the ones set up in the
        // initial exception translator, below, so add another for our local exception classes.
        //
        // libstdc++ doesn't require this (types there are identified only by name)
#if !defined(__GLIBCXX__)
        (*internals_pp)->registered_exception_translators.push_front(
            [](std::exception_ptr p) -> void {
                try {
                    if (p) std::rethrow_exception(p);
                } catch (error_already_set &e)       { e.restore();   return;
                } catch (const builtin_exception &e) { e.set_error(); return;
                }
            }
        );
#endif
    } else {
        if (!internals_pp) internals_pp = new internals*();
        auto *&internals_ptr = *internals_pp;
        internals_ptr = new internals();
#if defined(WITH_THREAD)
        PyEval_InitThreads();
        PyThreadState *tstate = PyThreadState_Get();
        #if PY_VERSION_HEX >= 0x03070000
            internals_ptr->tstate = PyThread_tss_alloc();
            if (!internals_ptr->tstate || PyThread_tss_create(internals_ptr->tstate))
                pybind11_fail("get_internals: could not successfully initialize the TSS key!");
            PyThread_tss_set(internals_ptr->tstate, tstate);
        #else
            internals_ptr->tstate = PyThread_create_key();
            if (internals_ptr->tstate == -1)
                pybind11_fail("get_internals: could not successfully initialize the TLS key!");
            PyThread_set_key_value(internals_ptr->tstate, tstate);
        #endif
        internals_ptr->istate = tstate->interp;
#endif
        builtins[id] = capsule(internals_pp);
        internals_ptr->registered_exception_translators.push_front(
            [](std::exception_ptr p) -> void {
                try {
                    if (p) std::rethrow_exception(p);
                } catch (error_already_set &e)           { e.restore();                                    return;
                } catch (const builtin_exception &e)     { e.set_error();                                  return;
                } catch (const std::bad_alloc &e)        { PyErr_SetString(PyExc_MemoryError,   e.what()); return;
                } catch (const std::domain_error &e)     { PyErr_SetString(PyExc_ValueError,    e.what()); return;
                } catch (const std::invalid_argument &e) { PyErr_SetString(PyExc_ValueError,    e.what()); return;
                } catch (const std::length_error &e)     { PyErr_SetString(PyExc_ValueError,    e.what()); return;
                } catch (const std::out_of_range &e)     { PyErr_SetString(PyExc_IndexError,    e.what()); return;
                } catch (const std::range_error &e)      { PyErr_SetString(PyExc_ValueError,    e.what()); return;
                } catch (const std::exception &e)        { PyErr_SetString(PyExc_RuntimeError,  e.what()); return;
                } catch (...) {
                    PyErr_SetString(PyExc_RuntimeError, "Caught an unknown exception!");
                    return;
                }
            }
        );
        internals_ptr->static_property_type = make_static_property_type();
        internals_ptr->default_metaclass = make_default_metaclass();
        internals_ptr->instance_base = make_object_base_type(internals_ptr->default_metaclass);
    }
    return **internals_pp;
}

/// Works like `internals.registered_types_cpp`, but for module-local registered types:
inline type_map<type_info *> &registered_local_types_cpp() {
    static type_map<type_info *> locals{};
    return locals;
}

/// Constructs a std::string with the given arguments, stores it in `internals`, and returns its
/// `c_str()`.  Such strings objects have a long storage duration -- the internal strings are only
/// cleared when the program exits or after interpreter shutdown (when embedding), and so are
/// suitable for c-style strings needed by Python internals (such as PyTypeObject's tp_name).
template <typename... Args>
const char *c_str(Args &&...args) {
    auto &strings = get_internals().static_strings;
    strings.emplace_front(std::forward<Args>(args)...);
    return strings.front().c_str();
}

NAMESPACE_END(detail)

/// Returns a named pointer that is shared among all extension modules (using the same
/// pybind11 version) running in the current interpreter. Names starting with underscores
/// are reserved for internal usage. Returns `nullptr` if no matching entry was found.
inline PYBIND11_NOINLINE void *get_shared_data(const std::string &name) {
    auto &internals = detail::get_internals();
    auto it = internals.shared_data.find(name);
    return it != internals.shared_data.end() ? it->second : nullptr;
}

/// Set the shared data that can be later recovered by `get_shared_data()`.
inline PYBIND11_NOINLINE void *set_shared_data(const std::string &name, void *data) {
    detail::get_internals().shared_data[name] = data;
    return data;
}

/// Returns a typed reference to a shared data entry (by using `get_shared_data()`) if
/// such entry exists. Otherwise, a new object of default-constructible type `T` is
/// added to the shared data under the given name and a reference to it is returned.
template<typename T>
T &get_or_create_shared_data(const std::string &name) {
    auto &internals = detail::get_internals();
    auto it = internals.shared_data.find(name);
    T *ptr = (T *) (it != internals.shared_data.end() ? it->second : nullptr);
    if (!ptr) {
        ptr = new T();
        internals.shared_data[name] = ptr;
    }
    return *ptr;
}

NAMESPACE_END(PYBIND11_NAMESPACE)
