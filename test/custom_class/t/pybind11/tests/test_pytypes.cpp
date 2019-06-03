/*
    tests/test_pytypes.cpp -- Python type casters

    Copyright (c) 2017 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE file.
*/

#include "pybind11_tests.h"


TEST_SUBMODULE(pytypes, m) {
    // test_list
    m.def("get_list", []() {
        py::list list;
        list.append("value");
        py::print("Entry at position 0:", list[0]);
        list[0] = py::str("overwritten");
        return list;
    });
    m.def("print_list", [](py::list list) {
        int index = 0;
        for (auto item : list)
            py::print("list item {}: {}"_s.format(index++, item));
    });

    // test_set
    m.def("get_set", []() {
        py::set set;
        set.add(py::str("key1"));
        set.add("key2");
        set.add(std::string("key3"));
        return set;
    });
    m.def("print_set", [](py::set set) {
        for (auto item : set)
            py::print("key:", item);
    });

    // test_dict
    m.def("get_dict", []() { return py::dict("key"_a="value"); });
    m.def("print_dict", [](py::dict dict) {
        for (auto item : dict)
            py::print("key: {}, value={}"_s.format(item.first, item.second));
    });
    m.def("dict_keyword_constructor", []() {
        auto d1 = py::dict("x"_a=1, "y"_a=2);
        auto d2 = py::dict("z"_a=3, **d1);
        return d2;
    });

    // test_str
    m.def("str_from_string", []() { return py::str(std::string("baz")); });
    m.def("str_from_bytes", []() { return py::str(py::bytes("boo", 3)); });
    m.def("str_from_object", [](const py::object& obj) { return py::str(obj); });
    m.def("repr_from_object", [](const py::object& obj) { return py::repr(obj); });

    m.def("str_format", []() {
        auto s1 = "{} + {} = {}"_s.format(1, 2, 3);
        auto s2 = "{a} + {b} = {c}"_s.format("a"_a=1, "b"_a=2, "c"_a=3);
        return py::make_tuple(s1, s2);
    });

    // test_bytes
    m.def("bytes_from_string", []() { return py::bytes(std::string("foo")); });
    m.def("bytes_from_str", []() { return py::bytes(py::str("bar", 3)); });

    // test_capsule
    m.def("return_capsule_with_destructor", []() {
        py::print("creating capsule");
        return py::capsule([]() {
            py::print("destructing capsule");
        });
    });

    m.def("return_capsule_with_destructor_2", []() {
        py::print("creating capsule");
        return py::capsule((void *) 1234, [](void *ptr) {
            py::print("destructing capsule: {}"_s.format((size_t) ptr));
        });
    });

    m.def("return_capsule_with_name_and_destructor", []() {
        auto capsule = py::capsule((void *) 1234, "pointer type description", [](PyObject *ptr) {
            if (ptr) {
                auto name = PyCapsule_GetName(ptr);
                py::print("destructing capsule ({}, '{}')"_s.format(
                    (size_t) PyCapsule_GetPointer(ptr, name), name
                ));
            }
        });
        void *contents = capsule;
        py::print("created capsule ({}, '{}')"_s.format((size_t) contents, capsule.name()));
        return capsule;
    });

    // test_accessors
    m.def("accessor_api", [](py::object o) {
        auto d = py::dict();

        d["basic_attr"] = o.attr("basic_attr");

        auto l = py::list();
        for (const auto &item : o.attr("begin_end")) {
            l.append(item);
        }
        d["begin_end"] = l;

        d["operator[object]"] = o.attr("d")["operator[object]"_s];
        d["operator[char *]"] = o.attr("d")["operator[char *]"];

        d["attr(object)"] = o.attr("sub").attr("attr_obj");
        d["attr(char *)"] = o.attr("sub").attr("attr_char");
        try {
            o.attr("sub").attr("missing").ptr();
        } catch (const py::error_already_set &) {
            d["missing_attr_ptr"] = "raised"_s;
        }
        try {
            o.attr("missing").attr("doesn't matter");
        } catch (const py::error_already_set &) {
            d["missing_attr_chain"] = "raised"_s;
        }

        d["is_none"] = o.attr("basic_attr").is_none();

        d["operator()"] = o.attr("func")(1);
        d["operator*"] = o.attr("func")(*o.attr("begin_end"));

        // Test implicit conversion
        py::list implicit_list = o.attr("begin_end");
        d["implicit_list"] = implicit_list;
        py::dict implicit_dict = o.attr("__dict__");
        d["implicit_dict"] = implicit_dict;

        return d;
    });

    m.def("tuple_accessor", [](py::tuple existing_t) {
        try {
            existing_t[0] = 1;
        } catch (const py::error_already_set &) {
            // --> Python system error
            // Only new tuples (refcount == 1) are mutable
            auto new_t = py::tuple(3);
            for (size_t i = 0; i < new_t.size(); ++i) {
                new_t[i] = i;
            }
            return new_t;
        }
        return py::tuple();
    });

    m.def("accessor_assignment", []() {
        auto l = py::list(1);
        l[0] = 0;

        auto d = py::dict();
        d["get"] = l[0];
        auto var = l[0];
        d["deferred_get"] = var;
        l[0] = 1;
        d["set"] = l[0];
        var = 99; // this assignment should not overwrite l[0]
        d["deferred_set"] = l[0];
        d["var"] = var;

        return d;
    });

    // test_constructors
    m.def("default_constructors", []() {
        return py::dict(
            "str"_a=py::str(),
            "bool"_a=py::bool_(),
            "int"_a=py::int_(),
            "float"_a=py::float_(),
            "tuple"_a=py::tuple(),
            "list"_a=py::list(),
            "dict"_a=py::dict(),
            "set"_a=py::set()
        );
    });

    m.def("converting_constructors", [](py::dict d) {
        return py::dict(
            "str"_a=py::str(d["str"]),
            "bool"_a=py::bool_(d["bool"]),
            "int"_a=py::int_(d["int"]),
            "float"_a=py::float_(d["float"]),
            "tuple"_a=py::tuple(d["tuple"]),
            "list"_a=py::list(d["list"]),
            "dict"_a=py::dict(d["dict"]),
            "set"_a=py::set(d["set"]),
            "memoryview"_a=py::memoryview(d["memoryview"])
        );
    });

    m.def("cast_functions", [](py::dict d) {
        // When converting between Python types, obj.cast<T>() should be the same as T(obj)
        return py::dict(
            "str"_a=d["str"].cast<py::str>(),
            "bool"_a=d["bool"].cast<py::bool_>(),
            "int"_a=d["int"].cast<py::int_>(),
            "float"_a=d["float"].cast<py::float_>(),
            "tuple"_a=d["tuple"].cast<py::tuple>(),
            "list"_a=d["list"].cast<py::list>(),
            "dict"_a=d["dict"].cast<py::dict>(),
            "set"_a=d["set"].cast<py::set>(),
            "memoryview"_a=d["memoryview"].cast<py::memoryview>()
        );
    });

    m.def("get_implicit_casting", []() {
        py::dict d;
        d["char*_i1"] = "abc";
        const char *c2 = "abc";
        d["char*_i2"] = c2;
        d["char*_e"] = py::cast(c2);
        d["char*_p"] = py::str(c2);

        d["int_i1"] = 42;
        int i = 42;
        d["int_i2"] = i;
        i++;
        d["int_e"] = py::cast(i);
        i++;
        d["int_p"] = py::int_(i);

        d["str_i1"] = std::string("str");
        std::string s2("str1");
        d["str_i2"] = s2;
        s2[3] = '2';
        d["str_e"] = py::cast(s2);
        s2[3] = '3';
        d["str_p"] = py::str(s2);

        py::list l(2);
        l[0] = 3;
        l[1] = py::cast(6);
        l.append(9);
        l.append(py::cast(12));
        l.append(py::int_(15));

        return py::dict(
            "d"_a=d,
            "l"_a=l
        );
    });

    // test_print
    m.def("print_function", []() {
        py::print("Hello, World!");
        py::print(1, 2.0, "three", true, std::string("-- multiple args"));
        auto args = py::make_tuple("and", "a", "custom", "separator");
        py::print("*args", *args, "sep"_a="-");
        py::print("no new line here", "end"_a=" -- ");
        py::print("next print");

        auto py_stderr = py::module::import("sys").attr("stderr");
        py::print("this goes to stderr", "file"_a=py_stderr);

        py::print("flush", "flush"_a=true);

        py::print("{a} + {b} = {c}"_s.format("a"_a="py::print", "b"_a="str.format", "c"_a="this"));
    });

    m.def("print_failure", []() { py::print(42, UnregisteredType()); });

    m.def("hash_function", [](py::object obj) { return py::hash(obj); });

    m.def("test_number_protocol", [](py::object a, py::object b) {
        py::list l;
        l.append(a.equal(b));
        l.append(a.not_equal(b));
        l.append(a < b);
        l.append(a <= b);
        l.append(a > b);
        l.append(a >= b);
        l.append(a + b);
        l.append(a - b);
        l.append(a * b);
        l.append(a / b);
        l.append(a | b);
        l.append(a & b);
        l.append(a ^ b);
        l.append(a >> b);
        l.append(a << b);
        return l;
    });

    m.def("test_list_slicing", [](py::list a) {
        return a[py::slice(0, -1, 2)];
    });
}
