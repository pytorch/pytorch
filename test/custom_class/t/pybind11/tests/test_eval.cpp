/*
    tests/test_eval.cpp -- Usage of eval() and eval_file()

    Copyright (c) 2016 Klemens D. Morgenstern

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE file.
*/


#include <pybind11/eval.h>
#include "pybind11_tests.h"

TEST_SUBMODULE(eval_, m) {
    // test_evals

    auto global = py::dict(py::module::import("__main__").attr("__dict__"));

    m.def("test_eval_statements", [global]() {
        auto local = py::dict();
        local["call_test"] = py::cpp_function([&]() -> int {
            return 42;
        });

        // Regular string literal
        py::exec(
            "message = 'Hello World!'\n"
            "x = call_test()",
            global, local
        );

        // Multi-line raw string literal
        py::exec(R"(
            if x == 42:
                print(message)
            else:
                raise RuntimeError
            )", global, local
        );
        auto x = local["x"].cast<int>();

        return x == 42;
    });

    m.def("test_eval", [global]() {
        auto local = py::dict();
        local["x"] = py::int_(42);
        auto x = py::eval("x", global, local);
        return x.cast<int>() == 42;
    });

    m.def("test_eval_single_statement", []() {
        auto local = py::dict();
        local["call_test"] = py::cpp_function([&]() -> int {
            return 42;
        });

        auto result = py::eval<py::eval_single_statement>("x = call_test()", py::dict(), local);
        auto x = local["x"].cast<int>();
        return result.is_none() && x == 42;
    });

    m.def("test_eval_file", [global](py::str filename) {
        auto local = py::dict();
        local["y"] = py::int_(43);

        int val_out;
        local["call_test2"] = py::cpp_function([&](int value) { val_out = value; });

        auto result = py::eval_file(filename, global, local);
        return val_out == 43 && result.is_none();
    });

    m.def("test_eval_failure", []() {
        try {
            py::eval("nonsense code ...");
        } catch (py::error_already_set &) {
            return true;
        }
        return false;
    });

    m.def("test_eval_file_failure", []() {
        try {
            py::eval_file("non-existing file");
        } catch (std::exception &) {
            return true;
        }
        return false;
    });
}
