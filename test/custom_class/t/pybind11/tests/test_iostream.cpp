/*
    tests/test_iostream.cpp -- Usage of scoped_output_redirect

    Copyright (c) 2017 Henry F. Schreiner

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE file.
*/


#include <pybind11/iostream.h>
#include "pybind11_tests.h"
#include <iostream>


void noisy_function(std::string msg, bool flush) {

    std::cout << msg;
    if (flush)
        std::cout << std::flush;
}

void noisy_funct_dual(std::string msg, std::string emsg) {
    std::cout << msg;
    std::cerr << emsg;
}

TEST_SUBMODULE(iostream, m) {

    add_ostream_redirect(m);

    // test_evals

    m.def("captured_output_default", [](std::string msg) {
        py::scoped_ostream_redirect redir;
        std::cout << msg << std::flush;
    });

    m.def("captured_output", [](std::string msg) {
        py::scoped_ostream_redirect redir(std::cout, py::module::import("sys").attr("stdout"));
        std::cout << msg << std::flush;
    });

    m.def("guard_output", &noisy_function,
            py::call_guard<py::scoped_ostream_redirect>(),
            py::arg("msg"), py::arg("flush")=true);

    m.def("captured_err", [](std::string msg) {
        py::scoped_ostream_redirect redir(std::cerr, py::module::import("sys").attr("stderr"));
        std::cerr << msg << std::flush;
    });

    m.def("noisy_function", &noisy_function, py::arg("msg"), py::arg("flush") = true);

    m.def("dual_guard", &noisy_funct_dual,
            py::call_guard<py::scoped_ostream_redirect, py::scoped_estream_redirect>(),
            py::arg("msg"), py::arg("emsg"));

    m.def("raw_output", [](std::string msg) {
        std::cout << msg << std::flush;
    });

    m.def("raw_err", [](std::string msg) {
        std::cerr << msg << std::flush;
    });

    m.def("captured_dual", [](std::string msg, std::string emsg) {
        py::scoped_ostream_redirect redirout(std::cout, py::module::import("sys").attr("stdout"));
        py::scoped_ostream_redirect redirerr(std::cerr, py::module::import("sys").attr("stderr"));
        std::cout << msg << std::flush;
        std::cerr << emsg << std::flush;
    });
}
