/*
    tests/test_enums.cpp -- enumerations

    Copyright (c) 2016 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE file.
*/

#include "pybind11_tests.h"

TEST_SUBMODULE(enums, m) {
    // test_unscoped_enum
    enum UnscopedEnum {
        EOne = 1,
        ETwo
    };
    py::enum_<UnscopedEnum>(m, "UnscopedEnum", py::arithmetic(), "An unscoped enumeration")
        .value("EOne", EOne, "Docstring for EOne")
        .value("ETwo", ETwo, "Docstring for ETwo")
        .export_values();

    // test_scoped_enum
    enum class ScopedEnum {
        Two = 2,
        Three
    };
    py::enum_<ScopedEnum>(m, "ScopedEnum", py::arithmetic())
        .value("Two", ScopedEnum::Two)
        .value("Three", ScopedEnum::Three);

    m.def("test_scoped_enum", [](ScopedEnum z) {
        return "ScopedEnum::" + std::string(z == ScopedEnum::Two ? "Two" : "Three");
    });

    // test_binary_operators
    enum Flags {
        Read = 4,
        Write = 2,
        Execute = 1
    };
    py::enum_<Flags>(m, "Flags", py::arithmetic())
        .value("Read", Flags::Read)
        .value("Write", Flags::Write)
        .value("Execute", Flags::Execute)
        .export_values();

    // test_implicit_conversion
    class ClassWithUnscopedEnum {
    public:
        enum EMode {
            EFirstMode = 1,
            ESecondMode
        };

        static EMode test_function(EMode mode) {
            return mode;
        }
    };
    py::class_<ClassWithUnscopedEnum> exenum_class(m, "ClassWithUnscopedEnum");
    exenum_class.def_static("test_function", &ClassWithUnscopedEnum::test_function);
    py::enum_<ClassWithUnscopedEnum::EMode>(exenum_class, "EMode")
        .value("EFirstMode", ClassWithUnscopedEnum::EFirstMode)
        .value("ESecondMode", ClassWithUnscopedEnum::ESecondMode)
        .export_values();

    // test_enum_to_int
    m.def("test_enum_to_int", [](int) { });
    m.def("test_enum_to_uint", [](uint32_t) { });
    m.def("test_enum_to_long_long", [](long long) { });

    // test_duplicate_enum_name
    enum SimpleEnum
    {
        ONE, TWO, THREE
    };

    m.def("register_bad_enum", [m]() {
        py::enum_<SimpleEnum>(m, "SimpleEnum")
            .value("ONE", SimpleEnum::ONE)          //NOTE: all value function calls are called with the same first parameter value
            .value("ONE", SimpleEnum::TWO)
            .value("ONE", SimpleEnum::THREE)
            .export_values();
    });
}
