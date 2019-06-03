/*
    pybind11/detail/class.h: Python C API implementation details for py::class_

    Copyright (c) 2017 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE file.
*/

#pragma once

#include "../attr.h"

NAMESPACE_BEGIN(PYBIND11_NAMESPACE)
NAMESPACE_BEGIN(detail)

#if PY_VERSION_HEX >= 0x03030000
#  define PYBIND11_BUILTIN_QUALNAME
#  define PYBIND11_SET_OLDPY_QUALNAME(obj, nameobj)
#else
// In pre-3.3 Python, we still set __qualname__ so that we can produce reliable function type
// signatures; in 3.3+ this macro expands to nothing:
#  define PYBIND11_SET_OLDPY_QUALNAME(obj, nameobj) setattr((PyObject *) obj, "__qualname__", nameobj)
#endif

inline PyTypeObject *type_incref(PyTypeObject *type) {
    Py_INCREF(type);
    return type;
}

#if !defined(PYPY_VERSION)

/// `pybind11_static_property.__get__()`: Always pass the class instead of the instance.
extern "C" inline PyObject *pybind11_static_get(PyObject *self, PyObject * /*ob*/, PyObject *cls) {
    return PyProperty_Type.tp_descr_get(self, cls, cls);
}

/// `pybind11_static_property.__set__()`: Just like the above `__get__()`.
extern "C" inline int pybind11_static_set(PyObject *self, PyObject *obj, PyObject *value) {
    PyObject *cls = PyType_Check(obj) ? obj : (PyObject *) Py_TYPE(obj);
    return PyProperty_Type.tp_descr_set(self, cls, value);
}

/** A `static_property` is the same as a `property` but the `__get__()` and `__set__()`
    methods are modified to always use the object type instead of a concrete instance.
    Return value: New reference. */
inline PyTypeObject *make_static_property_type() {
    constexpr auto *name = "pybind11_static_property";
    auto name_obj = reinterpret_steal<object>(PYBIND11_FROM_STRING(name));

    /* Danger zone: from now (and until PyType_Ready), make sure to
       issue no Python C API calls which could potentially invoke the
       garbage collector (the GC will call type_traverse(), which will in
       turn find the newly constructed type in an invalid state) */
    auto heap_type = (PyHeapTypeObject *) PyType_Type.tp_alloc(&PyType_Type, 0);
    if (!heap_type)
        pybind11_fail("make_static_property_type(): error allocating type!");

    heap_type->ht_name = name_obj.inc_ref().ptr();
#ifdef PYBIND11_BUILTIN_QUALNAME
    heap_type->ht_qualname = name_obj.inc_ref().ptr();
#endif

    auto type = &heap_type->ht_type;
    type->tp_name = name;
    type->tp_base = type_incref(&PyProperty_Type);
    type->tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE | Py_TPFLAGS_HEAPTYPE;
    type->tp_descr_get = pybind11_static_get;
    type->tp_descr_set = pybind11_static_set;

    if (PyType_Ready(type) < 0)
        pybind11_fail("make_static_property_type(): failure in PyType_Ready()!");

    setattr((PyObject *) type, "__module__", str("pybind11_builtins"));
    PYBIND11_SET_OLDPY_QUALNAME(type, name_obj);

    return type;
}

#else // PYPY

/** PyPy has some issues with the above C API, so we evaluate Python code instead.
    This function will only be called once so performance isn't really a concern.
    Return value: New reference. */
inline PyTypeObject *make_static_property_type() {
    auto d = dict();
    PyObject *result = PyRun_String(R"(\
        class pybind11_static_property(property):
            def __get__(self, obj, cls):
                return property.__get__(self, cls, cls)

            def __set__(self, obj, value):
                cls = obj if isinstance(obj, type) else type(obj)
                property.__set__(self, cls, value)
        )", Py_file_input, d.ptr(), d.ptr()
    );
    if (result == nullptr)
        throw error_already_set();
    Py_DECREF(result);
    return (PyTypeObject *) d["pybind11_static_property"].cast<object>().release().ptr();
}

#endif // PYPY

/** Types with static properties need to handle `Type.static_prop = x` in a specific way.
    By default, Python replaces the `static_property` itself, but for wrapped C++ types
    we need to call `static_property.__set__()` in order to propagate the new value to
    the underlying C++ data structure. */
extern "C" inline int pybind11_meta_setattro(PyObject* obj, PyObject* name, PyObject* value) {
    // Use `_PyType_Lookup()` instead of `PyObject_GetAttr()` in order to get the raw
    // descriptor (`property`) instead of calling `tp_descr_get` (`property.__get__()`).
    PyObject *descr = _PyType_Lookup((PyTypeObject *) obj, name);

    // The following assignment combinations are possible:
    //   1. `Type.static_prop = value`             --> descr_set: `Type.static_prop.__set__(value)`
    //   2. `Type.static_prop = other_static_prop` --> setattro:  replace existing `static_prop`
    //   3. `Type.regular_attribute = value`       --> setattro:  regular attribute assignment
    const auto static_prop = (PyObject *) get_internals().static_property_type;
    const auto call_descr_set = descr && PyObject_IsInstance(descr, static_prop)
                                && !PyObject_IsInstance(value, static_prop);
    if (call_descr_set) {
        // Call `static_property.__set__()` instead of replacing the `static_property`.
#if !defined(PYPY_VERSION)
        return Py_TYPE(descr)->tp_descr_set(descr, obj, value);
#else
        if (PyObject *result = PyObject_CallMethod(descr, "__set__", "OO", obj, value)) {
            Py_DECREF(result);
            return 0;
        } else {
            return -1;
        }
#endif
    } else {
        // Replace existing attribute.
        return PyType_Type.tp_setattro(obj, name, value);
    }
}

#if PY_MAJOR_VERSION >= 3
/**
 * Python 3's PyInstanceMethod_Type hides itself via its tp_descr_get, which prevents aliasing
 * methods via cls.attr("m2") = cls.attr("m1"): instead the tp_descr_get returns a plain function,
 * when called on a class, or a PyMethod, when called on an instance.  Override that behaviour here
 * to do a special case bypass for PyInstanceMethod_Types.
 */
extern "C" inline PyObject *pybind11_meta_getattro(PyObject *obj, PyObject *name) {
    PyObject *descr = _PyType_Lookup((PyTypeObject *) obj, name);
    if (descr && PyInstanceMethod_Check(descr)) {
        Py_INCREF(descr);
        return descr;
    }
    else {
        return PyType_Type.tp_getattro(obj, name);
    }
}
#endif

/** This metaclass is assigned by default to all pybind11 types and is required in order
    for static properties to function correctly. Users may override this using `py::metaclass`.
    Return value: New reference. */
inline PyTypeObject* make_default_metaclass() {
    constexpr auto *name = "pybind11_type";
    auto name_obj = reinterpret_steal<object>(PYBIND11_FROM_STRING(name));

    /* Danger zone: from now (and until PyType_Ready), make sure to
       issue no Python C API calls which could potentially invoke the
       garbage collector (the GC will call type_traverse(), which will in
       turn find the newly constructed type in an invalid state) */
    auto heap_type = (PyHeapTypeObject *) PyType_Type.tp_alloc(&PyType_Type, 0);
    if (!heap_type)
        pybind11_fail("make_default_metaclass(): error allocating metaclass!");

    heap_type->ht_name = name_obj.inc_ref().ptr();
#ifdef PYBIND11_BUILTIN_QUALNAME
    heap_type->ht_qualname = name_obj.inc_ref().ptr();
#endif

    auto type = &heap_type->ht_type;
    type->tp_name = name;
    type->tp_base = type_incref(&PyType_Type);
    type->tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE | Py_TPFLAGS_HEAPTYPE;

    type->tp_setattro = pybind11_meta_setattro;
#if PY_MAJOR_VERSION >= 3
    type->tp_getattro = pybind11_meta_getattro;
#endif

    if (PyType_Ready(type) < 0)
        pybind11_fail("make_default_metaclass(): failure in PyType_Ready()!");

    setattr((PyObject *) type, "__module__", str("pybind11_builtins"));
    PYBIND11_SET_OLDPY_QUALNAME(type, name_obj);

    return type;
}

/// For multiple inheritance types we need to recursively register/deregister base pointers for any
/// base classes with pointers that are difference from the instance value pointer so that we can
/// correctly recognize an offset base class pointer. This calls a function with any offset base ptrs.
inline void traverse_offset_bases(void *valueptr, const detail::type_info *tinfo, instance *self,
        bool (*f)(void * /*parentptr*/, instance * /*self*/)) {
    for (handle h : reinterpret_borrow<tuple>(tinfo->type->tp_bases)) {
        if (auto parent_tinfo = get_type_info((PyTypeObject *) h.ptr())) {
            for (auto &c : parent_tinfo->implicit_casts) {
                if (c.first == tinfo->cpptype) {
                    auto *parentptr = c.second(valueptr);
                    if (parentptr != valueptr)
                        f(parentptr, self);
                    traverse_offset_bases(parentptr, parent_tinfo, self, f);
                    break;
                }
            }
        }
    }
}

inline bool register_instance_impl(void *ptr, instance *self) {
    get_internals().registered_instances.emplace(ptr, self);
    return true; // unused, but gives the same signature as the deregister func
}
inline bool deregister_instance_impl(void *ptr, instance *self) {
    auto &registered_instances = get_internals().registered_instances;
    auto range = registered_instances.equal_range(ptr);
    for (auto it = range.first; it != range.second; ++it) {
        if (Py_TYPE(self) == Py_TYPE(it->second)) {
            registered_instances.erase(it);
            return true;
        }
    }
    return false;
}

inline void register_instance(instance *self, void *valptr, const type_info *tinfo) {
    register_instance_impl(valptr, self);
    if (!tinfo->simple_ancestors)
        traverse_offset_bases(valptr, tinfo, self, register_instance_impl);
}

inline bool deregister_instance(instance *self, void *valptr, const type_info *tinfo) {
    bool ret = deregister_instance_impl(valptr, self);
    if (!tinfo->simple_ancestors)
        traverse_offset_bases(valptr, tinfo, self, deregister_instance_impl);
    return ret;
}

/// Instance creation function for all pybind11 types. It allocates the internal instance layout for
/// holding C++ objects and holders.  Allocation is done lazily (the first time the instance is cast
/// to a reference or pointer), and initialization is done by an `__init__` function.
inline PyObject *make_new_instance(PyTypeObject *type) {
#if defined(PYPY_VERSION)
    // PyPy gets tp_basicsize wrong (issue 2482) under multiple inheritance when the first inherited
    // object is a a plain Python type (i.e. not derived from an extension type).  Fix it.
    ssize_t instance_size = static_cast<ssize_t>(sizeof(instance));
    if (type->tp_basicsize < instance_size) {
        type->tp_basicsize = instance_size;
    }
#endif
    PyObject *self = type->tp_alloc(type, 0);
    auto inst = reinterpret_cast<instance *>(self);
    // Allocate the value/holder internals:
    inst->allocate_layout();

    inst->owned = true;

    return self;
}

/// Instance creation function for all pybind11 types. It only allocates space for the
/// C++ object, but doesn't call the constructor -- an `__init__` function must do that.
extern "C" inline PyObject *pybind11_object_new(PyTypeObject *type, PyObject *, PyObject *) {
    return make_new_instance(type);
}

/// An `__init__` function constructs the C++ object. Users should provide at least one
/// of these using `py::init` or directly with `.def(__init__, ...)`. Otherwise, the
/// following default function will be used which simply throws an exception.
extern "C" inline int pybind11_object_init(PyObject *self, PyObject *, PyObject *) {
    PyTypeObject *type = Py_TYPE(self);
    std::string msg;
#if defined(PYPY_VERSION)
    msg += handle((PyObject *) type).attr("__module__").cast<std::string>() + ".";
#endif
    msg += type->tp_name;
    msg += ": No constructor defined!";
    PyErr_SetString(PyExc_TypeError, msg.c_str());
    return -1;
}

inline void add_patient(PyObject *nurse, PyObject *patient) {
    auto &internals = get_internals();
    auto instance = reinterpret_cast<detail::instance *>(nurse);
    instance->has_patients = true;
    Py_INCREF(patient);
    internals.patients[nurse].push_back(patient);
}

inline void clear_patients(PyObject *self) {
    auto instance = reinterpret_cast<detail::instance *>(self);
    auto &internals = get_internals();
    auto pos = internals.patients.find(self);
    assert(pos != internals.patients.end());
    // Clearing the patients can cause more Python code to run, which
    // can invalidate the iterator. Extract the vector of patients
    // from the unordered_map first.
    auto patients = std::move(pos->second);
    internals.patients.erase(pos);
    instance->has_patients = false;
    for (PyObject *&patient : patients)
        Py_CLEAR(patient);
}

/// Clears all internal data from the instance and removes it from registered instances in
/// preparation for deallocation.
inline void clear_instance(PyObject *self) {
    auto instance = reinterpret_cast<detail::instance *>(self);

    // Deallocate any values/holders, if present:
    for (auto &v_h : values_and_holders(instance)) {
        if (v_h) {

            // We have to deregister before we call dealloc because, for virtual MI types, we still
            // need to be able to get the parent pointers.
            if (v_h.instance_registered() && !deregister_instance(instance, v_h.value_ptr(), v_h.type))
                pybind11_fail("pybind11_object_dealloc(): Tried to deallocate unregistered instance!");

            if (instance->owned || v_h.holder_constructed())
                v_h.type->dealloc(v_h);
        }
    }
    // Deallocate the value/holder layout internals:
    instance->deallocate_layout();

    if (instance->weakrefs)
        PyObject_ClearWeakRefs(self);

    PyObject **dict_ptr = _PyObject_GetDictPtr(self);
    if (dict_ptr)
        Py_CLEAR(*dict_ptr);

    if (instance->has_patients)
        clear_patients(self);
}

/// Instance destructor function for all pybind11 types. It calls `type_info.dealloc`
/// to destroy the C++ object itself, while the rest is Python bookkeeping.
extern "C" inline void pybind11_object_dealloc(PyObject *self) {
    clear_instance(self);

    auto type = Py_TYPE(self);
    type->tp_free(self);

    // `type->tp_dealloc != pybind11_object_dealloc` means that we're being called
    // as part of a derived type's dealloc, in which case we're not allowed to decref
    // the type here. For cross-module compatibility, we shouldn't compare directly
    // with `pybind11_object_dealloc`, but with the common one stashed in internals.
    auto pybind11_object_type = (PyTypeObject *) get_internals().instance_base;
    if (type->tp_dealloc == pybind11_object_type->tp_dealloc)
        Py_DECREF(type);
}

/** Create the type which can be used as a common base for all classes.  This is
    needed in order to satisfy Python's requirements for multiple inheritance.
    Return value: New reference. */
inline PyObject *make_object_base_type(PyTypeObject *metaclass) {
    constexpr auto *name = "pybind11_object";
    auto name_obj = reinterpret_steal<object>(PYBIND11_FROM_STRING(name));

    /* Danger zone: from now (and until PyType_Ready), make sure to
       issue no Python C API calls which could potentially invoke the
       garbage collector (the GC will call type_traverse(), which will in
       turn find the newly constructed type in an invalid state) */
    auto heap_type = (PyHeapTypeObject *) metaclass->tp_alloc(metaclass, 0);
    if (!heap_type)
        pybind11_fail("make_object_base_type(): error allocating type!");

    heap_type->ht_name = name_obj.inc_ref().ptr();
#ifdef PYBIND11_BUILTIN_QUALNAME
    heap_type->ht_qualname = name_obj.inc_ref().ptr();
#endif

    auto type = &heap_type->ht_type;
    type->tp_name = name;
    type->tp_base = type_incref(&PyBaseObject_Type);
    type->tp_basicsize = static_cast<ssize_t>(sizeof(instance));
    type->tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE | Py_TPFLAGS_HEAPTYPE;

    type->tp_new = pybind11_object_new;
    type->tp_init = pybind11_object_init;
    type->tp_dealloc = pybind11_object_dealloc;

    /* Support weak references (needed for the keep_alive feature) */
    type->tp_weaklistoffset = offsetof(instance, weakrefs);

    if (PyType_Ready(type) < 0)
        pybind11_fail("PyType_Ready failed in make_object_base_type():" + error_string());

    setattr((PyObject *) type, "__module__", str("pybind11_builtins"));
    PYBIND11_SET_OLDPY_QUALNAME(type, name_obj);

    assert(!PyType_HasFeature(type, Py_TPFLAGS_HAVE_GC));
    return (PyObject *) heap_type;
}

/// dynamic_attr: Support for `d = instance.__dict__`.
extern "C" inline PyObject *pybind11_get_dict(PyObject *self, void *) {
    PyObject *&dict = *_PyObject_GetDictPtr(self);
    if (!dict)
        dict = PyDict_New();
    Py_XINCREF(dict);
    return dict;
}

/// dynamic_attr: Support for `instance.__dict__ = dict()`.
extern "C" inline int pybind11_set_dict(PyObject *self, PyObject *new_dict, void *) {
    if (!PyDict_Check(new_dict)) {
        PyErr_Format(PyExc_TypeError, "__dict__ must be set to a dictionary, not a '%.200s'",
                     Py_TYPE(new_dict)->tp_name);
        return -1;
    }
    PyObject *&dict = *_PyObject_GetDictPtr(self);
    Py_INCREF(new_dict);
    Py_CLEAR(dict);
    dict = new_dict;
    return 0;
}

/// dynamic_attr: Allow the garbage collector to traverse the internal instance `__dict__`.
extern "C" inline int pybind11_traverse(PyObject *self, visitproc visit, void *arg) {
    PyObject *&dict = *_PyObject_GetDictPtr(self);
    Py_VISIT(dict);
    return 0;
}

/// dynamic_attr: Allow the GC to clear the dictionary.
extern "C" inline int pybind11_clear(PyObject *self) {
    PyObject *&dict = *_PyObject_GetDictPtr(self);
    Py_CLEAR(dict);
    return 0;
}

/// Give instances of this type a `__dict__` and opt into garbage collection.
inline void enable_dynamic_attributes(PyHeapTypeObject *heap_type) {
    auto type = &heap_type->ht_type;
#if defined(PYPY_VERSION)
    pybind11_fail(std::string(type->tp_name) + ": dynamic attributes are "
                                               "currently not supported in "
                                               "conjunction with PyPy!");
#endif
    type->tp_flags |= Py_TPFLAGS_HAVE_GC;
    type->tp_dictoffset = type->tp_basicsize; // place dict at the end
    type->tp_basicsize += (ssize_t)sizeof(PyObject *); // and allocate enough space for it
    type->tp_traverse = pybind11_traverse;
    type->tp_clear = pybind11_clear;

    static PyGetSetDef getset[] = {
        {const_cast<char*>("__dict__"), pybind11_get_dict, pybind11_set_dict, nullptr, nullptr},
        {nullptr, nullptr, nullptr, nullptr, nullptr}
    };
    type->tp_getset = getset;
}

/// buffer_protocol: Fill in the view as specified by flags.
extern "C" inline int pybind11_getbuffer(PyObject *obj, Py_buffer *view, int flags) {
    // Look for a `get_buffer` implementation in this type's info or any bases (following MRO).
    type_info *tinfo = nullptr;
    for (auto type : reinterpret_borrow<tuple>(Py_TYPE(obj)->tp_mro)) {
        tinfo = get_type_info((PyTypeObject *) type.ptr());
        if (tinfo && tinfo->get_buffer)
            break;
    }
    if (view == nullptr || obj == nullptr || !tinfo || !tinfo->get_buffer) {
        if (view)
            view->obj = nullptr;
        PyErr_SetString(PyExc_BufferError, "pybind11_getbuffer(): Internal error");
        return -1;
    }
    std::memset(view, 0, sizeof(Py_buffer));
    buffer_info *info = tinfo->get_buffer(obj, tinfo->get_buffer_data);
    view->obj = obj;
    view->ndim = 1;
    view->internal = info;
    view->buf = info->ptr;
    view->itemsize = info->itemsize;
    view->len = view->itemsize;
    for (auto s : info->shape)
        view->len *= s;
    if ((flags & PyBUF_FORMAT) == PyBUF_FORMAT)
        view->format = const_cast<char *>(info->format.c_str());
    if ((flags & PyBUF_STRIDES) == PyBUF_STRIDES) {
        view->ndim = (int) info->ndim;
        view->strides = &info->strides[0];
        view->shape = &info->shape[0];
    }
    Py_INCREF(view->obj);
    return 0;
}

/// buffer_protocol: Release the resources of the buffer.
extern "C" inline void pybind11_releasebuffer(PyObject *, Py_buffer *view) {
    delete (buffer_info *) view->internal;
}

/// Give this type a buffer interface.
inline void enable_buffer_protocol(PyHeapTypeObject *heap_type) {
    heap_type->ht_type.tp_as_buffer = &heap_type->as_buffer;
#if PY_MAJOR_VERSION < 3
    heap_type->ht_type.tp_flags |= Py_TPFLAGS_HAVE_NEWBUFFER;
#endif

    heap_type->as_buffer.bf_getbuffer = pybind11_getbuffer;
    heap_type->as_buffer.bf_releasebuffer = pybind11_releasebuffer;
}

/** Create a brand new Python type according to the `type_record` specification.
    Return value: New reference. */
inline PyObject* make_new_python_type(const type_record &rec) {
    auto name = reinterpret_steal<object>(PYBIND11_FROM_STRING(rec.name));

    auto qualname = name;
    if (rec.scope && !PyModule_Check(rec.scope.ptr()) && hasattr(rec.scope, "__qualname__")) {
#if PY_MAJOR_VERSION >= 3
        qualname = reinterpret_steal<object>(
            PyUnicode_FromFormat("%U.%U", rec.scope.attr("__qualname__").ptr(), name.ptr()));
#else
        qualname = str(rec.scope.attr("__qualname__").cast<std::string>() + "." + rec.name);
#endif
    }

    object module;
    if (rec.scope) {
        if (hasattr(rec.scope, "__module__"))
            module = rec.scope.attr("__module__");
        else if (hasattr(rec.scope, "__name__"))
            module = rec.scope.attr("__name__");
    }

    auto full_name = c_str(
#if !defined(PYPY_VERSION)
        module ? str(module).cast<std::string>() + "." + rec.name :
#endif
        rec.name);

    char *tp_doc = nullptr;
    if (rec.doc && options::show_user_defined_docstrings()) {
        /* Allocate memory for docstring (using PyObject_MALLOC, since
           Python will free this later on) */
        size_t size = strlen(rec.doc) + 1;
        tp_doc = (char *) PyObject_MALLOC(size);
        memcpy((void *) tp_doc, rec.doc, size);
    }

    auto &internals = get_internals();
    auto bases = tuple(rec.bases);
    auto base = (bases.size() == 0) ? internals.instance_base
                                    : bases[0].ptr();

    /* Danger zone: from now (and until PyType_Ready), make sure to
       issue no Python C API calls which could potentially invoke the
       garbage collector (the GC will call type_traverse(), which will in
       turn find the newly constructed type in an invalid state) */
    auto metaclass = rec.metaclass.ptr() ? (PyTypeObject *) rec.metaclass.ptr()
                                         : internals.default_metaclass;

    auto heap_type = (PyHeapTypeObject *) metaclass->tp_alloc(metaclass, 0);
    if (!heap_type)
        pybind11_fail(std::string(rec.name) + ": Unable to create type object!");

    heap_type->ht_name = name.release().ptr();
#ifdef PYBIND11_BUILTIN_QUALNAME
    heap_type->ht_qualname = qualname.inc_ref().ptr();
#endif

    auto type = &heap_type->ht_type;
    type->tp_name = full_name;
    type->tp_doc = tp_doc;
    type->tp_base = type_incref((PyTypeObject *)base);
    type->tp_basicsize = static_cast<ssize_t>(sizeof(instance));
    if (bases.size() > 0)
        type->tp_bases = bases.release().ptr();

    /* Don't inherit base __init__ */
    type->tp_init = pybind11_object_init;

    /* Supported protocols */
    type->tp_as_number = &heap_type->as_number;
    type->tp_as_sequence = &heap_type->as_sequence;
    type->tp_as_mapping = &heap_type->as_mapping;

    /* Flags */
    type->tp_flags |= Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE | Py_TPFLAGS_HEAPTYPE;
#if PY_MAJOR_VERSION < 3
    type->tp_flags |= Py_TPFLAGS_CHECKTYPES;
#endif

    if (rec.dynamic_attr)
        enable_dynamic_attributes(heap_type);

    if (rec.buffer_protocol)
        enable_buffer_protocol(heap_type);

    if (PyType_Ready(type) < 0)
        pybind11_fail(std::string(rec.name) + ": PyType_Ready failed (" + error_string() + ")!");

    assert(rec.dynamic_attr ? PyType_HasFeature(type, Py_TPFLAGS_HAVE_GC)
                            : !PyType_HasFeature(type, Py_TPFLAGS_HAVE_GC));

    /* Register type with the parent scope */
    if (rec.scope)
        setattr(rec.scope, rec.name, (PyObject *) type);
    else
        Py_INCREF(type); // Keep it alive forever (reference leak)

    if (module) // Needed by pydoc
        setattr((PyObject *) type, "__module__", module);

    PYBIND11_SET_OLDPY_QUALNAME(type, qualname);

    return (PyObject *) type;
}

NAMESPACE_END(detail)
NAMESPACE_END(PYBIND11_NAMESPACE)
