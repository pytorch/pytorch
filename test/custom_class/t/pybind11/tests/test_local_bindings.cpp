/*
    tests/test_local_bindings.cpp -- tests the py::module_local class feature which makes a class
                                     binding local to the module in which it is defined.

    Copyright (c) 2017 Jason Rhinelander <jason@imaginary.ca>

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE file.
*/

#include "pybind11_tests.h"
#include "local_bindings.h"
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <numeric>

TEST_SUBMODULE(local_bindings, m) {
    // test_load_external
    m.def("load_external1", [](ExternalType1 &e) { return e.i; });
    m.def("load_external2", [](ExternalType2 &e) { return e.i; });

    // test_local_bindings
    // Register a class with py::module_local:
    bind_local<LocalType, -1>(m, "LocalType", py::module_local())
        .def("get3", [](LocalType &t) { return t.i + 3; })
        ;

    m.def("local_value", [](LocalType &l) { return l.i; });

    // test_nonlocal_failure
    // The main pybind11 test module is loaded first, so this registration will succeed (the second
    // one, in pybind11_cross_module_tests.cpp, is designed to fail):
    bind_local<NonLocalType, 0>(m, "NonLocalType")
        .def(py::init<int>())
        .def("get", [](LocalType &i) { return i.i; })
        ;

    // test_duplicate_local
    // py::module_local declarations should be visible across compilation units that get linked together;
    // this tries to register a duplicate local.  It depends on a definition in test_class.cpp and
    // should raise a runtime error from the duplicate definition attempt.  If test_class isn't
    // available it *also* throws a runtime error (with "test_class not enabled" as value).
    m.def("register_local_external", [m]() {
        auto main = py::module::import("pybind11_tests");
        if (py::hasattr(main, "class_")) {
            bind_local<LocalExternal, 7>(m, "LocalExternal", py::module_local());
        }
        else throw std::runtime_error("test_class not enabled");
    });

    // test_stl_bind_local
    // stl_bind.h binders defaults to py::module_local if the types are local or converting:
    py::bind_vector<LocalVec>(m, "LocalVec");
    py::bind_map<LocalMap>(m, "LocalMap");
    // and global if the type (or one of the types, for the map) is global:
    py::bind_vector<NonLocalVec>(m, "NonLocalVec");
    py::bind_map<NonLocalMap>(m, "NonLocalMap");

    // test_stl_bind_global
    // They can, however, be overridden to global using `py::module_local(false)`:
    bind_local<NonLocal2, 10>(m, "NonLocal2");
    py::bind_vector<LocalVec2>(m, "LocalVec2", py::module_local());
    py::bind_map<NonLocalMap2>(m, "NonLocalMap2", py::module_local(false));

    // test_mixed_local_global
    // We try this both with the global type registered first and vice versa (the order shouldn't
    // matter).
    m.def("register_mixed_global", [m]() {
        bind_local<MixedGlobalLocal, 100>(m, "MixedGlobalLocal", py::module_local(false));
    });
    m.def("register_mixed_local", [m]() {
        bind_local<MixedLocalGlobal, 1000>(m, "MixedLocalGlobal", py::module_local());
    });
    m.def("get_mixed_gl", [](int i) { return MixedGlobalLocal(i); });
    m.def("get_mixed_lg", [](int i) { return MixedLocalGlobal(i); });

    // test_internal_locals_differ
    m.def("local_cpp_types_addr", []() { return (uintptr_t) &py::detail::registered_local_types_cpp(); });

    // test_stl_caster_vs_stl_bind
    m.def("load_vector_via_caster", [](std::vector<int> v) {
        return std::accumulate(v.begin(), v.end(), 0);
    });

    // test_cross_module_calls
    m.def("return_self", [](LocalVec *v) { return v; });
    m.def("return_copy", [](const LocalVec &v) { return LocalVec(v); });

    class Cat : public pets::Pet { public: Cat(std::string name) : Pet(name) {}; };
    py::class_<pets::Pet>(m, "Pet", py::module_local())
        .def("get_name", &pets::Pet::name);
    // Binding for local extending class:
    py::class_<Cat, pets::Pet>(m, "Cat")
        .def(py::init<std::string>());
    m.def("pet_name", [](pets::Pet &p) { return p.name(); });

    py::class_<MixGL>(m, "MixGL").def(py::init<int>());
    m.def("get_gl_value", [](MixGL &o) { return o.i + 10; });

    py::class_<MixGL2>(m, "MixGL2").def(py::init<int>());
}
