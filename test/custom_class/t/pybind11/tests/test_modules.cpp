/*
    tests/test_modules.cpp -- nested modules, importing modules, and
                            internal references

    Copyright (c) 2016 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE file.
*/

#include "pybind11_tests.h"
#include "constructor_stats.h"

TEST_SUBMODULE(modules, m) {
    // test_nested_modules
    py::module m_sub = m.def_submodule("subsubmodule");
    m_sub.def("submodule_func", []() { return "submodule_func()"; });

    // test_reference_internal
    class A {
    public:
        A(int v) : v(v) { print_created(this, v); }
        ~A() { print_destroyed(this); }
        A(const A&) { print_copy_created(this); }
        A& operator=(const A &copy) { print_copy_assigned(this); v = copy.v; return *this; }
        std::string toString() { return "A[" + std::to_string(v) + "]"; }
    private:
        int v;
    };
    py::class_<A>(m_sub, "A")
        .def(py::init<int>())
        .def("__repr__", &A::toString);

    class B {
    public:
        B() { print_default_created(this); }
        ~B() { print_destroyed(this); }
        B(const B&) { print_copy_created(this); }
        B& operator=(const B &copy) { print_copy_assigned(this); a1 = copy.a1; a2 = copy.a2; return *this; }
        A &get_a1() { return a1; }
        A &get_a2() { return a2; }

        A a1{1};
        A a2{2};
    };
    py::class_<B>(m_sub, "B")
        .def(py::init<>())
        .def("get_a1", &B::get_a1, "Return the internal A 1", py::return_value_policy::reference_internal)
        .def("get_a2", &B::get_a2, "Return the internal A 2", py::return_value_policy::reference_internal)
        .def_readwrite("a1", &B::a1)  // def_readonly uses an internal reference return policy by default
        .def_readwrite("a2", &B::a2);

    m.attr("OD") = py::module::import("collections").attr("OrderedDict");

    // test_duplicate_registration
    // Registering two things with the same name
    m.def("duplicate_registration", []() {
        class Dupe1 { };
        class Dupe2 { };
        class Dupe3 { };
        class DupeException { };

        auto dm = py::module("dummy");
        auto failures = py::list();

        py::class_<Dupe1>(dm, "Dupe1");
        py::class_<Dupe2>(dm, "Dupe2");
        dm.def("dupe1_factory", []() { return Dupe1(); });
        py::exception<DupeException>(dm, "DupeException");

        try {
            py::class_<Dupe1>(dm, "Dupe1");
            failures.append("Dupe1 class");
        } catch (std::runtime_error &) {}
        try {
            dm.def("Dupe1", []() { return Dupe1(); });
            failures.append("Dupe1 function");
        } catch (std::runtime_error &) {}
        try {
            py::class_<Dupe3>(dm, "dupe1_factory");
            failures.append("dupe1_factory");
        } catch (std::runtime_error &) {}
        try {
            py::exception<Dupe3>(dm, "Dupe2");
            failures.append("Dupe2");
        } catch (std::runtime_error &) {}
        try {
            dm.def("DupeException", []() { return 30; });
            failures.append("DupeException1");
        } catch (std::runtime_error &) {}
        try {
            py::class_<DupeException>(dm, "DupeException");
            failures.append("DupeException2");
        } catch (std::runtime_error &) {}

        return failures;
    });
}
