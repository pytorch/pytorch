/*
    tests/pybind11_tests.cpp -- pybind example plugin

    Copyright (c) 2016 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE file.
*/

#include "pybind11_tests.h"
#include "constructor_stats.h"

#include <functional>
#include <list>

/*
For testing purposes, we define a static global variable here in a function that each individual
test .cpp calls with its initialization lambda.  It's convenient here because we can just not
compile some test files to disable/ignore some of the test code.

It is NOT recommended as a way to use pybind11 in practice, however: the initialization order will
be essentially random, which is okay for our test scripts (there are no dependencies between the
individual pybind11 test .cpp files), but most likely not what you want when using pybind11
productively.

Instead, see the "How can I reduce the build time?" question in the "Frequently asked questions"
section of the documentation for good practice on splitting binding code over multiple files.
*/
std::list<std::function<void(py::module &)>> &initializers() {
    static std::list<std::function<void(py::module &)>> inits;
    return inits;
}

test_initializer::test_initializer(Initializer init) {
    initializers().push_back(init);
}

test_initializer::test_initializer(const char *submodule_name, Initializer init) {
    initializers().push_back([=](py::module &parent) {
        auto m = parent.def_submodule(submodule_name);
        init(m);
    });
}

void bind_ConstructorStats(py::module &m) {
    py::class_<ConstructorStats>(m, "ConstructorStats")
        .def("alive", &ConstructorStats::alive)
        .def("values", &ConstructorStats::values)
        .def_readwrite("default_constructions", &ConstructorStats::default_constructions)
        .def_readwrite("copy_assignments", &ConstructorStats::copy_assignments)
        .def_readwrite("move_assignments", &ConstructorStats::move_assignments)
        .def_readwrite("copy_constructions", &ConstructorStats::copy_constructions)
        .def_readwrite("move_constructions", &ConstructorStats::move_constructions)
        .def_static("get", (ConstructorStats &(*)(py::object)) &ConstructorStats::get, py::return_value_policy::reference_internal)

        // Not exactly ConstructorStats, but related: expose the internal pybind number of registered instances
        // to allow instance cleanup checks (invokes a GC first)
        .def_static("detail_reg_inst", []() {
            ConstructorStats::gc();
            return py::detail::get_internals().registered_instances.size();
        })
        ;
}

PYBIND11_MODULE(pybind11_tests, m) {
    m.doc() = "pybind11 test module";

    bind_ConstructorStats(m);

#if !defined(NDEBUG)
    m.attr("debug_enabled") = true;
#else
    m.attr("debug_enabled") = false;
#endif

    py::class_<UserType>(m, "UserType", "A `py::class_` type for testing")
        .def(py::init<>())
        .def(py::init<int>())
        .def("get_value", &UserType::value, "Get value using a method")
        .def("set_value", &UserType::set, "Set value using a method")
        .def_property("value", &UserType::value, &UserType::set, "Get/set value using a property")
        .def("__repr__", [](const UserType& u) { return "UserType({})"_s.format(u.value()); });

    py::class_<IncType, UserType>(m, "IncType")
        .def(py::init<>())
        .def(py::init<int>())
        .def("__repr__", [](const IncType& u) { return "IncType({})"_s.format(u.value()); });

    for (const auto &initializer : initializers())
        initializer(m);

    if (!py::hasattr(m, "have_eigen")) m.attr("have_eigen") = false;
}
