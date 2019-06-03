#pragma once
/*
    tests/constructor_stats.h -- framework for printing and tracking object
    instance lifetimes in example/test code.

    Copyright (c) 2016 Jason Rhinelander <jason@imaginary.ca>

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE file.

This header provides a few useful tools for writing examples or tests that want to check and/or
display object instance lifetimes.  It requires that you include this header and add the following
function calls to constructors:

    class MyClass {
        MyClass() { ...; print_default_created(this); }
        ~MyClass() { ...; print_destroyed(this); }
        MyClass(const MyClass &c) { ...; print_copy_created(this); }
        MyClass(MyClass &&c) { ...; print_move_created(this); }
        MyClass(int a, int b) { ...; print_created(this, a, b); }
        MyClass &operator=(const MyClass &c) { ...; print_copy_assigned(this); }
        MyClass &operator=(MyClass &&c) { ...; print_move_assigned(this); }

        ...
    }

You can find various examples of these in several of the existing testing .cpp files.  (Of course
you don't need to add any of the above constructors/operators that you don't actually have, except
for the destructor).

Each of these will print an appropriate message such as:

    ### MyClass @ 0x2801910 created via default constructor
    ### MyClass @ 0x27fa780 created 100 200
    ### MyClass @ 0x2801910 destroyed
    ### MyClass @ 0x27fa780 destroyed

You can also include extra arguments (such as the 100, 200 in the output above, coming from the
value constructor) for all of the above methods which will be included in the output.

For testing, each of these also keeps track the created instances and allows you to check how many
of the various constructors have been invoked from the Python side via code such as:

    from pybind11_tests import ConstructorStats
    cstats = ConstructorStats.get(MyClass)
    print(cstats.alive())
    print(cstats.default_constructions)

Note that `.alive()` should usually be the first thing you call as it invokes Python's garbage
collector to actually destroy objects that aren't yet referenced.

For everything except copy and move constructors and destructors, any extra values given to the
print_...() function is stored in a class-specific values list which you can retrieve and inspect
from the ConstructorStats instance `.values()` method.

In some cases, when you need to track instances of a C++ class not registered with pybind11, you
need to add a function returning the ConstructorStats for the C++ class; this can be done with:

    m.def("get_special_cstats", &ConstructorStats::get<SpecialClass>, py::return_value_policy::reference)

Finally, you can suppress the output messages, but keep the constructor tracking (for
inspection/testing in python) by using the functions with `print_` replaced with `track_` (e.g.
`track_copy_created(this)`).

*/

#include "pybind11_tests.h"
#include <unordered_map>
#include <list>
#include <typeindex>
#include <sstream>

class ConstructorStats {
protected:
    std::unordered_map<void*, int> _instances; // Need a map rather than set because members can shared address with parents
    std::list<std::string> _values; // Used to track values (e.g. of value constructors)
public:
    int default_constructions = 0;
    int copy_constructions = 0;
    int move_constructions = 0;
    int copy_assignments = 0;
    int move_assignments = 0;

    void copy_created(void *inst) {
        created(inst);
        copy_constructions++;
    }

    void move_created(void *inst) {
        created(inst);
        move_constructions++;
    }

    void default_created(void *inst) {
        created(inst);
        default_constructions++;
    }

    void created(void *inst) {
        ++_instances[inst];
    }

    void destroyed(void *inst) {
        if (--_instances[inst] < 0)
            throw std::runtime_error("cstats.destroyed() called with unknown "
                                     "instance; potential double-destruction "
                                     "or a missing cstats.created()");
    }

    static void gc() {
        // Force garbage collection to ensure any pending destructors are invoked:
#if defined(PYPY_VERSION)
        PyObject *globals = PyEval_GetGlobals();
        PyObject *result = PyRun_String(
            "import gc\n"
            "for i in range(2):"
            "    gc.collect()\n",
            Py_file_input, globals, globals);
        if (result == nullptr)
            throw py::error_already_set();
        Py_DECREF(result);
#else
        py::module::import("gc").attr("collect")();
#endif
    }

    int alive() {
        gc();
        int total = 0;
        for (const auto &p : _instances)
            if (p.second > 0)
                total += p.second;
        return total;
    }

    void value() {} // Recursion terminator
    // Takes one or more values, converts them to strings, then stores them.
    template <typename T, typename... Tmore> void value(const T &v, Tmore &&...args) {
        std::ostringstream oss;
        oss << v;
        _values.push_back(oss.str());
        value(std::forward<Tmore>(args)...);
    }

    // Move out stored values
    py::list values() {
        py::list l;
        for (const auto &v : _values) l.append(py::cast(v));
        _values.clear();
        return l;
    }

    // Gets constructor stats from a C++ type index
    static ConstructorStats& get(std::type_index type) {
        static std::unordered_map<std::type_index, ConstructorStats> all_cstats;
        return all_cstats[type];
    }

    // Gets constructor stats from a C++ type
    template <typename T> static ConstructorStats& get() {
#if defined(PYPY_VERSION)
        gc();
#endif
        return get(typeid(T));
    }

    // Gets constructor stats from a Python class
    static ConstructorStats& get(py::object class_) {
        auto &internals = py::detail::get_internals();
        const std::type_index *t1 = nullptr, *t2 = nullptr;
        try {
            auto *type_info = internals.registered_types_py.at((PyTypeObject *) class_.ptr()).at(0);
            for (auto &p : internals.registered_types_cpp) {
                if (p.second == type_info) {
                    if (t1) {
                        t2 = &p.first;
                        break;
                    }
                    t1 = &p.first;
                }
            }
        }
        catch (const std::out_of_range &) {}
        if (!t1) throw std::runtime_error("Unknown class passed to ConstructorStats::get()");
        auto &cs1 = get(*t1);
        // If we have both a t1 and t2 match, one is probably the trampoline class; return whichever
        // has more constructions (typically one or the other will be 0)
        if (t2) {
            auto &cs2 = get(*t2);
            int cs1_total = cs1.default_constructions + cs1.copy_constructions + cs1.move_constructions + (int) cs1._values.size();
            int cs2_total = cs2.default_constructions + cs2.copy_constructions + cs2.move_constructions + (int) cs2._values.size();
            if (cs2_total > cs1_total) return cs2;
        }
        return cs1;
    }
};

// To track construction/destruction, you need to call these methods from the various
// constructors/operators.  The ones that take extra values record the given values in the
// constructor stats values for later inspection.
template <class T> void track_copy_created(T *inst) { ConstructorStats::get<T>().copy_created(inst); }
template <class T> void track_move_created(T *inst) { ConstructorStats::get<T>().move_created(inst); }
template <class T, typename... Values> void track_copy_assigned(T *, Values &&...values) {
    auto &cst = ConstructorStats::get<T>();
    cst.copy_assignments++;
    cst.value(std::forward<Values>(values)...);
}
template <class T, typename... Values> void track_move_assigned(T *, Values &&...values) {
    auto &cst = ConstructorStats::get<T>();
    cst.move_assignments++;
    cst.value(std::forward<Values>(values)...);
}
template <class T, typename... Values> void track_default_created(T *inst, Values &&...values) {
    auto &cst = ConstructorStats::get<T>();
    cst.default_created(inst);
    cst.value(std::forward<Values>(values)...);
}
template <class T, typename... Values> void track_created(T *inst, Values &&...values) {
    auto &cst = ConstructorStats::get<T>();
    cst.created(inst);
    cst.value(std::forward<Values>(values)...);
}
template <class T, typename... Values> void track_destroyed(T *inst) {
    ConstructorStats::get<T>().destroyed(inst);
}
template <class T, typename... Values> void track_values(T *, Values &&...values) {
    ConstructorStats::get<T>().value(std::forward<Values>(values)...);
}

/// Don't cast pointers to Python, print them as strings
inline const char *format_ptrs(const char *p) { return p; }
template <typename T>
py::str format_ptrs(T *p) { return "{:#x}"_s.format(reinterpret_cast<std::uintptr_t>(p)); }
template <typename T>
auto format_ptrs(T &&x) -> decltype(std::forward<T>(x)) { return std::forward<T>(x); }

template <class T, typename... Output>
void print_constr_details(T *inst, const std::string &action, Output &&...output) {
    py::print("###", py::type_id<T>(), "@", format_ptrs(inst), action,
              format_ptrs(std::forward<Output>(output))...);
}

// Verbose versions of the above:
template <class T, typename... Values> void print_copy_created(T *inst, Values &&...values) { // NB: this prints, but doesn't store, given values
    print_constr_details(inst, "created via copy constructor", values...);
    track_copy_created(inst);
}
template <class T, typename... Values> void print_move_created(T *inst, Values &&...values) { // NB: this prints, but doesn't store, given values
    print_constr_details(inst, "created via move constructor", values...);
    track_move_created(inst);
}
template <class T, typename... Values> void print_copy_assigned(T *inst, Values &&...values) {
    print_constr_details(inst, "assigned via copy assignment", values...);
    track_copy_assigned(inst, values...);
}
template <class T, typename... Values> void print_move_assigned(T *inst, Values &&...values) {
    print_constr_details(inst, "assigned via move assignment", values...);
    track_move_assigned(inst, values...);
}
template <class T, typename... Values> void print_default_created(T *inst, Values &&...values) {
    print_constr_details(inst, "created via default constructor", values...);
    track_default_created(inst, values...);
}
template <class T, typename... Values> void print_created(T *inst, Values &&...values) {
    print_constr_details(inst, "created", values...);
    track_created(inst, values...);
}
template <class T, typename... Values> void print_destroyed(T *inst, Values &&...values) { // Prints but doesn't store given values
    print_constr_details(inst, "destroyed", values...);
    track_destroyed(inst);
}
template <class T, typename... Values> void print_values(T *inst, Values &&...values) {
    print_constr_details(inst, ":", values...);
    track_values(inst, values...);
}

