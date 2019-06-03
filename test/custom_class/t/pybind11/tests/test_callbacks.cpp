/*
    tests/test_callbacks.cpp -- callbacks

    Copyright (c) 2016 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE file.
*/

#include "pybind11_tests.h"
#include "constructor_stats.h"
#include <pybind11/functional.h>


int dummy_function(int i) { return i + 1; }

TEST_SUBMODULE(callbacks, m) {
    // test_callbacks, test_function_signatures
    m.def("test_callback1", [](py::object func) { return func(); });
    m.def("test_callback2", [](py::object func) { return func("Hello", 'x', true, 5); });
    m.def("test_callback3", [](const std::function<int(int)> &func) {
        return "func(43) = " + std::to_string(func(43)); });
    m.def("test_callback4", []() -> std::function<int(int)> { return [](int i) { return i+1; }; });
    m.def("test_callback5", []() {
        return py::cpp_function([](int i) { return i+1; }, py::arg("number"));
    });

    // test_keyword_args_and_generalized_unpacking
    m.def("test_tuple_unpacking", [](py::function f) {
        auto t1 = py::make_tuple(2, 3);
        auto t2 = py::make_tuple(5, 6);
        return f("positional", 1, *t1, 4, *t2);
    });

    m.def("test_dict_unpacking", [](py::function f) {
        auto d1 = py::dict("key"_a="value", "a"_a=1);
        auto d2 = py::dict();
        auto d3 = py::dict("b"_a=2);
        return f("positional", 1, **d1, **d2, **d3);
    });

    m.def("test_keyword_args", [](py::function f) {
        return f("x"_a=10, "y"_a=20);
    });

    m.def("test_unpacking_and_keywords1", [](py::function f) {
        auto args = py::make_tuple(2);
        auto kwargs = py::dict("d"_a=4);
        return f(1, *args, "c"_a=3, **kwargs);
    });

    m.def("test_unpacking_and_keywords2", [](py::function f) {
        auto kwargs1 = py::dict("a"_a=1);
        auto kwargs2 = py::dict("c"_a=3, "d"_a=4);
        return f("positional", *py::make_tuple(1), 2, *py::make_tuple(3, 4), 5,
                 "key"_a="value", **kwargs1, "b"_a=2, **kwargs2, "e"_a=5);
    });

    m.def("test_unpacking_error1", [](py::function f) {
        auto kwargs = py::dict("x"_a=3);
        return f("x"_a=1, "y"_a=2, **kwargs); // duplicate ** after keyword
    });

    m.def("test_unpacking_error2", [](py::function f) {
        auto kwargs = py::dict("x"_a=3);
        return f(**kwargs, "x"_a=1); // duplicate keyword after **
    });

    m.def("test_arg_conversion_error1", [](py::function f) {
        f(234, UnregisteredType(), "kw"_a=567);
    });

    m.def("test_arg_conversion_error2", [](py::function f) {
        f(234, "expected_name"_a=UnregisteredType(), "kw"_a=567);
    });

    // test_lambda_closure_cleanup
    struct Payload {
        Payload() { print_default_created(this); }
        ~Payload() { print_destroyed(this); }
        Payload(const Payload &) { print_copy_created(this); }
        Payload(Payload &&) { print_move_created(this); }
    };
    // Export the payload constructor statistics for testing purposes:
    m.def("payload_cstats", &ConstructorStats::get<Payload>);
    /* Test cleanup of lambda closure */
    m.def("test_cleanup", []() -> std::function<void(void)> {
        Payload p;

        return [p]() {
            /* p should be cleaned up when the returned function is garbage collected */
            (void) p;
        };
    });

    // test_cpp_function_roundtrip
    /* Test if passing a function pointer from C++ -> Python -> C++ yields the original pointer */
    m.def("dummy_function", &dummy_function);
    m.def("dummy_function2", [](int i, int j) { return i + j; });
    m.def("roundtrip", [](std::function<int(int)> f, bool expect_none = false) {
        if (expect_none && f)
            throw std::runtime_error("Expected None to be converted to empty std::function");
        return f;
    }, py::arg("f"), py::arg("expect_none")=false);
    m.def("test_dummy_function", [](const std::function<int(int)> &f) -> std::string {
        using fn_type = int (*)(int);
        auto result = f.target<fn_type>();
        if (!result) {
            auto r = f(1);
            return "can't convert to function pointer: eval(1) = " + std::to_string(r);
        } else if (*result == dummy_function) {
            auto r = (*result)(1);
            return "matches dummy_function: eval(1) = " + std::to_string(r);
        } else {
            return "argument does NOT match dummy_function. This should never happen!";
        }
    });

    class AbstractBase { public: virtual unsigned int func() = 0; };
    m.def("func_accepting_func_accepting_base", [](std::function<double(AbstractBase&)>) { });

    struct MovableObject {
        bool valid = true;

        MovableObject() = default;
        MovableObject(const MovableObject &) = default;
        MovableObject &operator=(const MovableObject &) = default;
        MovableObject(MovableObject &&o) : valid(o.valid) { o.valid = false; }
        MovableObject &operator=(MovableObject &&o) {
            valid = o.valid;
            o.valid = false;
            return *this;
        }
    };
    py::class_<MovableObject>(m, "MovableObject");

    // test_movable_object
    m.def("callback_with_movable", [](std::function<void(MovableObject &)> f) {
        auto x = MovableObject();
        f(x); // lvalue reference shouldn't move out object
        return x.valid; // must still return `true`
    });

    // test_bound_method_callback
    struct CppBoundMethodTest {};
    py::class_<CppBoundMethodTest>(m, "CppBoundMethodTest")
        .def(py::init<>())
        .def("triple", [](CppBoundMethodTest &, int val) { return 3 * val; });
}
