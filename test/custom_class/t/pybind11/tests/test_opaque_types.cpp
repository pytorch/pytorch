/*
    tests/test_opaque_types.cpp -- opaque types, passing void pointers

    Copyright (c) 2016 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE file.
*/

#include "pybind11_tests.h"
#include <pybind11/stl.h>
#include <vector>

// IMPORTANT: Disable internal pybind11 translation mechanisms for STL data structures
//
// This also deliberately doesn't use the below StringList type alias to test
// that MAKE_OPAQUE can handle a type containing a `,`.  (The `std::allocator`
// bit is just the default `std::vector` allocator).
PYBIND11_MAKE_OPAQUE(std::vector<std::string, std::allocator<std::string>>);

using StringList = std::vector<std::string, std::allocator<std::string>>;

TEST_SUBMODULE(opaque_types, m) {
    // test_string_list
    py::class_<StringList>(m, "StringList")
        .def(py::init<>())
        .def("pop_back", &StringList::pop_back)
        /* There are multiple versions of push_back(), etc. Select the right ones. */
        .def("push_back", (void (StringList::*)(const std::string &)) &StringList::push_back)
        .def("back", (std::string &(StringList::*)()) &StringList::back)
        .def("__len__", [](const StringList &v) { return v.size(); })
        .def("__iter__", [](StringList &v) {
           return py::make_iterator(v.begin(), v.end());
        }, py::keep_alive<0, 1>());

    class ClassWithSTLVecProperty {
    public:
        StringList stringList;
    };
    py::class_<ClassWithSTLVecProperty>(m, "ClassWithSTLVecProperty")
        .def(py::init<>())
        .def_readwrite("stringList", &ClassWithSTLVecProperty::stringList);

    m.def("print_opaque_list", [](const StringList &l) {
        std::string ret = "Opaque list: [";
        bool first = true;
        for (auto entry : l) {
            if (!first)
                ret += ", ";
            ret += entry;
            first = false;
        }
        return ret + "]";
    });

    // test_pointers
    m.def("return_void_ptr", []() { return (void *) 0x1234; });
    m.def("get_void_ptr_value", [](void *ptr) { return reinterpret_cast<std::intptr_t>(ptr); });
    m.def("return_null_str", []() { return (char *) nullptr; });
    m.def("get_null_str_value", [](char *ptr) { return reinterpret_cast<std::intptr_t>(ptr); });

    m.def("return_unique_ptr", []() -> std::unique_ptr<StringList> {
        StringList *result = new StringList();
        result->push_back("some value");
        return std::unique_ptr<StringList>(result);
    });
}
