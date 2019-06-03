/*
    tests/test_call_policies.cpp -- keep_alive and call_guard

    Copyright (c) 2016 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE file.
*/

#include "pybind11_tests.h"

struct CustomGuard {
    static bool enabled;

    CustomGuard() { enabled = true; }
    ~CustomGuard() { enabled = false; }

    static const char *report_status() { return enabled ? "guarded" : "unguarded"; }
};
bool CustomGuard::enabled = false;

struct DependentGuard {
    static bool enabled;

    DependentGuard() { enabled = CustomGuard::enabled; }
    ~DependentGuard() { enabled = false; }

    static const char *report_status() { return enabled ? "guarded" : "unguarded"; }
};
bool DependentGuard::enabled = false;

TEST_SUBMODULE(call_policies, m) {
    // Parent/Child are used in:
    // test_keep_alive_argument, test_keep_alive_return_value, test_alive_gc_derived,
    // test_alive_gc_multi_derived, test_return_none, test_keep_alive_constructor
    class Child {
    public:
        Child() { py::print("Allocating child."); }
        Child(const Child &) = default;
        Child(Child &&) = default;
        ~Child() { py::print("Releasing child."); }
    };
    py::class_<Child>(m, "Child")
        .def(py::init<>());

    class Parent {
    public:
        Parent() { py::print("Allocating parent."); }
        ~Parent() { py::print("Releasing parent."); }
        void addChild(Child *) { }
        Child *returnChild() { return new Child(); }
        Child *returnNullChild() { return nullptr; }
    };
    py::class_<Parent>(m, "Parent")
        .def(py::init<>())
        .def(py::init([](Child *) { return new Parent(); }), py::keep_alive<1, 2>())
        .def("addChild", &Parent::addChild)
        .def("addChildKeepAlive", &Parent::addChild, py::keep_alive<1, 2>())
        .def("returnChild", &Parent::returnChild)
        .def("returnChildKeepAlive", &Parent::returnChild, py::keep_alive<1, 0>())
        .def("returnNullChildKeepAliveChild", &Parent::returnNullChild, py::keep_alive<1, 0>())
        .def("returnNullChildKeepAliveParent", &Parent::returnNullChild, py::keep_alive<0, 1>());

#if !defined(PYPY_VERSION)
    // test_alive_gc
    class ParentGC : public Parent {
    public:
        using Parent::Parent;
    };
    py::class_<ParentGC, Parent>(m, "ParentGC", py::dynamic_attr())
        .def(py::init<>());
#endif

    // test_call_guard
    m.def("unguarded_call", &CustomGuard::report_status);
    m.def("guarded_call", &CustomGuard::report_status, py::call_guard<CustomGuard>());

    m.def("multiple_guards_correct_order", []() {
        return CustomGuard::report_status() + std::string(" & ") + DependentGuard::report_status();
    }, py::call_guard<CustomGuard, DependentGuard>());

    m.def("multiple_guards_wrong_order", []() {
        return DependentGuard::report_status() + std::string(" & ") + CustomGuard::report_status();
    }, py::call_guard<DependentGuard, CustomGuard>());

#if defined(WITH_THREAD) && !defined(PYPY_VERSION)
    // `py::call_guard<py::gil_scoped_release>()` should work in PyPy as well,
    // but it's unclear how to test it without `PyGILState_GetThisThreadState`.
    auto report_gil_status = []() {
        auto is_gil_held = false;
        if (auto tstate = py::detail::get_thread_state_unchecked())
            is_gil_held = (tstate == PyGILState_GetThisThreadState());

        return is_gil_held ? "GIL held" : "GIL released";
    };

    m.def("with_gil", report_gil_status);
    m.def("without_gil", report_gil_status, py::call_guard<py::gil_scoped_release>());
#endif
}
