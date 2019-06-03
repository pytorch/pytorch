/*
    tests/test_numpy_vectorize.cpp -- auto-vectorize functions over NumPy array
    arguments

    Copyright (c) 2016 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE file.
*/

#include "pybind11_tests.h"
#include <pybind11/numpy.h>

double my_func(int x, float y, double z) {
    py::print("my_func(x:int={}, y:float={:.0f}, z:float={:.0f})"_s.format(x, y, z));
    return (float) x*y*z;
}

TEST_SUBMODULE(numpy_vectorize, m) {
    try { py::module::import("numpy"); }
    catch (...) { return; }

    // test_vectorize, test_docs, test_array_collapse
    // Vectorize all arguments of a function (though non-vector arguments are also allowed)
    m.def("vectorized_func", py::vectorize(my_func));

    // Vectorize a lambda function with a capture object (e.g. to exclude some arguments from the vectorization)
    m.def("vectorized_func2",
        [](py::array_t<int> x, py::array_t<float> y, float z) {
            return py::vectorize([z](int x, float y) { return my_func(x, y, z); })(x, y);
        }
    );

    // Vectorize a complex-valued function
    m.def("vectorized_func3", py::vectorize(
        [](std::complex<double> c) { return c * std::complex<double>(2.f); }
    ));

    // test_type_selection
    // Numpy function which only accepts specific data types
    m.def("selective_func", [](py::array_t<int, py::array::c_style>) { return "Int branch taken."; });
    m.def("selective_func", [](py::array_t<float, py::array::c_style>) { return "Float branch taken."; });
    m.def("selective_func", [](py::array_t<std::complex<float>, py::array::c_style>) { return "Complex float branch taken."; });


    // test_passthrough_arguments
    // Passthrough test: references and non-pod types should be automatically passed through (in the
    // function definition below, only `b`, `d`, and `g` are vectorized):
    struct NonPODClass {
        NonPODClass(int v) : value{v} {}
        int value;
    };
    py::class_<NonPODClass>(m, "NonPODClass").def(py::init<int>());
    m.def("vec_passthrough", py::vectorize(
        [](double *a, double b, py::array_t<double> c, const int &d, int &e, NonPODClass f, const double g) {
            return *a + b + c.at(0) + d + e + f.value + g;
        }
    ));

    // test_method_vectorization
    struct VectorizeTestClass {
        VectorizeTestClass(int v) : value{v} {};
        float method(int x, float y) { return y + (float) (x + value); }
        int value = 0;
    };
    py::class_<VectorizeTestClass> vtc(m, "VectorizeTestClass");
    vtc .def(py::init<int>())
        .def_readwrite("value", &VectorizeTestClass::value);

    // Automatic vectorizing of methods
    vtc.def("method", py::vectorize(&VectorizeTestClass::method));

    // test_trivial_broadcasting
    // Internal optimization test for whether the input is trivially broadcastable:
    py::enum_<py::detail::broadcast_trivial>(m, "trivial")
        .value("f_trivial", py::detail::broadcast_trivial::f_trivial)
        .value("c_trivial", py::detail::broadcast_trivial::c_trivial)
        .value("non_trivial", py::detail::broadcast_trivial::non_trivial);
    m.def("vectorized_is_trivial", [](
                py::array_t<int, py::array::forcecast> arg1,
                py::array_t<float, py::array::forcecast> arg2,
                py::array_t<double, py::array::forcecast> arg3
                ) {
        ssize_t ndim;
        std::vector<ssize_t> shape;
        std::array<py::buffer_info, 3> buffers {{ arg1.request(), arg2.request(), arg3.request() }};
        return py::detail::broadcast(buffers, ndim, shape);
    });
}
