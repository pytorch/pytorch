/*
    tests/test_numpy_array.cpp -- test core array functionality

    Copyright (c) 2016 Ivan Smirnov <i.s.smirnov@gmail.com>

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE file.
*/

#include "pybind11_tests.h"

#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <cstdint>

using arr = py::array;
using arr_t = py::array_t<uint16_t, 0>;
static_assert(std::is_same<arr_t::value_type, uint16_t>::value, "");

template<typename... Ix> arr data(const arr& a, Ix... index) {
    return arr(a.nbytes() - a.offset_at(index...), (const uint8_t *) a.data(index...));
}

template<typename... Ix> arr data_t(const arr_t& a, Ix... index) {
    return arr(a.size() - a.index_at(index...), a.data(index...));
}

template<typename... Ix> arr& mutate_data(arr& a, Ix... index) {
    auto ptr = (uint8_t *) a.mutable_data(index...);
    for (ssize_t i = 0; i < a.nbytes() - a.offset_at(index...); i++)
        ptr[i] = (uint8_t) (ptr[i] * 2);
    return a;
}

template<typename... Ix> arr_t& mutate_data_t(arr_t& a, Ix... index) {
    auto ptr = a.mutable_data(index...);
    for (ssize_t i = 0; i < a.size() - a.index_at(index...); i++)
        ptr[i]++;
    return a;
}

template<typename... Ix> ssize_t index_at(const arr& a, Ix... idx) { return a.index_at(idx...); }
template<typename... Ix> ssize_t index_at_t(const arr_t& a, Ix... idx) { return a.index_at(idx...); }
template<typename... Ix> ssize_t offset_at(const arr& a, Ix... idx) { return a.offset_at(idx...); }
template<typename... Ix> ssize_t offset_at_t(const arr_t& a, Ix... idx) { return a.offset_at(idx...); }
template<typename... Ix> ssize_t at_t(const arr_t& a, Ix... idx) { return a.at(idx...); }
template<typename... Ix> arr_t& mutate_at_t(arr_t& a, Ix... idx) { a.mutable_at(idx...)++; return a; }

#define def_index_fn(name, type) \
    sm.def(#name, [](type a) { return name(a); }); \
    sm.def(#name, [](type a, int i) { return name(a, i); }); \
    sm.def(#name, [](type a, int i, int j) { return name(a, i, j); }); \
    sm.def(#name, [](type a, int i, int j, int k) { return name(a, i, j, k); });

template <typename T, typename T2> py::handle auxiliaries(T &&r, T2 &&r2) {
    if (r.ndim() != 2) throw std::domain_error("error: ndim != 2");
    py::list l;
    l.append(*r.data(0, 0));
    l.append(*r2.mutable_data(0, 0));
    l.append(r.data(0, 1) == r2.mutable_data(0, 1));
    l.append(r.ndim());
    l.append(r.itemsize());
    l.append(r.shape(0));
    l.append(r.shape(1));
    l.append(r.size());
    l.append(r.nbytes());
    return l.release();
}

TEST_SUBMODULE(numpy_array, sm) {
    try { py::module::import("numpy"); }
    catch (...) { return; }

    // test_array_attributes
    sm.def("ndim", [](const arr& a) { return a.ndim(); });
    sm.def("shape", [](const arr& a) { return arr(a.ndim(), a.shape()); });
    sm.def("shape", [](const arr& a, ssize_t dim) { return a.shape(dim); });
    sm.def("strides", [](const arr& a) { return arr(a.ndim(), a.strides()); });
    sm.def("strides", [](const arr& a, ssize_t dim) { return a.strides(dim); });
    sm.def("writeable", [](const arr& a) { return a.writeable(); });
    sm.def("size", [](const arr& a) { return a.size(); });
    sm.def("itemsize", [](const arr& a) { return a.itemsize(); });
    sm.def("nbytes", [](const arr& a) { return a.nbytes(); });
    sm.def("owndata", [](const arr& a) { return a.owndata(); });

    // test_index_offset
    def_index_fn(index_at, const arr&);
    def_index_fn(index_at_t, const arr_t&);
    def_index_fn(offset_at, const arr&);
    def_index_fn(offset_at_t, const arr_t&);
    // test_data
    def_index_fn(data, const arr&);
    def_index_fn(data_t, const arr_t&);
    // test_mutate_data, test_mutate_readonly
    def_index_fn(mutate_data, arr&);
    def_index_fn(mutate_data_t, arr_t&);
    def_index_fn(at_t, const arr_t&);
    def_index_fn(mutate_at_t, arr_t&);

    // test_make_c_f_array
    sm.def("make_f_array", [] { return py::array_t<float>({ 2, 2 }, { 4, 8 }); });
    sm.def("make_c_array", [] { return py::array_t<float>({ 2, 2 }, { 8, 4 }); });

    // test_empty_shaped_array
    sm.def("make_empty_shaped_array", [] { return py::array(py::dtype("f"), {}, {}); });

    // test_wrap
    sm.def("wrap", [](py::array a) {
        return py::array(
            a.dtype(),
            {a.shape(), a.shape() + a.ndim()},
            {a.strides(), a.strides() + a.ndim()},
            a.data(),
            a
        );
    });

    // test_numpy_view
    struct ArrayClass {
        int data[2] = { 1, 2 };
        ArrayClass() { py::print("ArrayClass()"); }
        ~ArrayClass() { py::print("~ArrayClass()"); }
    };
    py::class_<ArrayClass>(sm, "ArrayClass")
        .def(py::init<>())
        .def("numpy_view", [](py::object &obj) {
            py::print("ArrayClass::numpy_view()");
            ArrayClass &a = obj.cast<ArrayClass&>();
            return py::array_t<int>({2}, {4}, a.data, obj);
        }
    );

    // test_cast_numpy_int64_to_uint64
    sm.def("function_taking_uint64", [](uint64_t) { });

    // test_isinstance
    sm.def("isinstance_untyped", [](py::object yes, py::object no) {
        return py::isinstance<py::array>(yes) && !py::isinstance<py::array>(no);
    });
    sm.def("isinstance_typed", [](py::object o) {
        return py::isinstance<py::array_t<double>>(o) && !py::isinstance<py::array_t<int>>(o);
    });

    // test_constructors
    sm.def("default_constructors", []() {
        return py::dict(
            "array"_a=py::array(),
            "array_t<int32>"_a=py::array_t<std::int32_t>(),
            "array_t<double>"_a=py::array_t<double>()
        );
    });
    sm.def("converting_constructors", [](py::object o) {
        return py::dict(
            "array"_a=py::array(o),
            "array_t<int32>"_a=py::array_t<std::int32_t>(o),
            "array_t<double>"_a=py::array_t<double>(o)
        );
    });

    // test_overload_resolution
    sm.def("overloaded", [](py::array_t<double>) { return "double"; });
    sm.def("overloaded", [](py::array_t<float>) { return "float"; });
    sm.def("overloaded", [](py::array_t<int>) { return "int"; });
    sm.def("overloaded", [](py::array_t<unsigned short>) { return "unsigned short"; });
    sm.def("overloaded", [](py::array_t<long long>) { return "long long"; });
    sm.def("overloaded", [](py::array_t<std::complex<double>>) { return "double complex"; });
    sm.def("overloaded", [](py::array_t<std::complex<float>>) { return "float complex"; });

    sm.def("overloaded2", [](py::array_t<std::complex<double>>) { return "double complex"; });
    sm.def("overloaded2", [](py::array_t<double>) { return "double"; });
    sm.def("overloaded2", [](py::array_t<std::complex<float>>) { return "float complex"; });
    sm.def("overloaded2", [](py::array_t<float>) { return "float"; });

    // Only accept the exact types:
    sm.def("overloaded3", [](py::array_t<int>) { return "int"; }, py::arg().noconvert());
    sm.def("overloaded3", [](py::array_t<double>) { return "double"; }, py::arg().noconvert());

    // Make sure we don't do unsafe coercion (e.g. float to int) when not using forcecast, but
    // rather that float gets converted via the safe (conversion to double) overload:
    sm.def("overloaded4", [](py::array_t<long long, 0>) { return "long long"; });
    sm.def("overloaded4", [](py::array_t<double, 0>) { return "double"; });

    // But we do allow conversion to int if forcecast is enabled (but only if no overload matches
    // without conversion)
    sm.def("overloaded5", [](py::array_t<unsigned int>) { return "unsigned int"; });
    sm.def("overloaded5", [](py::array_t<double>) { return "double"; });

    // test_greedy_string_overload
    // Issue 685: ndarray shouldn't go to std::string overload
    sm.def("issue685", [](std::string) { return "string"; });
    sm.def("issue685", [](py::array) { return "array"; });
    sm.def("issue685", [](py::object) { return "other"; });

    // test_array_unchecked_fixed_dims
    sm.def("proxy_add2", [](py::array_t<double> a, double v) {
        auto r = a.mutable_unchecked<2>();
        for (ssize_t i = 0; i < r.shape(0); i++)
            for (ssize_t j = 0; j < r.shape(1); j++)
                r(i, j) += v;
    }, py::arg().noconvert(), py::arg());

    sm.def("proxy_init3", [](double start) {
        py::array_t<double, py::array::c_style> a({ 3, 3, 3 });
        auto r = a.mutable_unchecked<3>();
        for (ssize_t i = 0; i < r.shape(0); i++)
        for (ssize_t j = 0; j < r.shape(1); j++)
        for (ssize_t k = 0; k < r.shape(2); k++)
            r(i, j, k) = start++;
        return a;
    });
    sm.def("proxy_init3F", [](double start) {
        py::array_t<double, py::array::f_style> a({ 3, 3, 3 });
        auto r = a.mutable_unchecked<3>();
        for (ssize_t k = 0; k < r.shape(2); k++)
        for (ssize_t j = 0; j < r.shape(1); j++)
        for (ssize_t i = 0; i < r.shape(0); i++)
            r(i, j, k) = start++;
        return a;
    });
    sm.def("proxy_squared_L2_norm", [](py::array_t<double> a) {
        auto r = a.unchecked<1>();
        double sumsq = 0;
        for (ssize_t i = 0; i < r.shape(0); i++)
            sumsq += r[i] * r(i); // Either notation works for a 1D array
        return sumsq;
    });

    sm.def("proxy_auxiliaries2", [](py::array_t<double> a) {
        auto r = a.unchecked<2>();
        auto r2 = a.mutable_unchecked<2>();
        return auxiliaries(r, r2);
    });

    // test_array_unchecked_dyn_dims
    // Same as the above, but without a compile-time dimensions specification:
    sm.def("proxy_add2_dyn", [](py::array_t<double> a, double v) {
        auto r = a.mutable_unchecked();
        if (r.ndim() != 2) throw std::domain_error("error: ndim != 2");
        for (ssize_t i = 0; i < r.shape(0); i++)
            for (ssize_t j = 0; j < r.shape(1); j++)
                r(i, j) += v;
    }, py::arg().noconvert(), py::arg());
    sm.def("proxy_init3_dyn", [](double start) {
        py::array_t<double, py::array::c_style> a({ 3, 3, 3 });
        auto r = a.mutable_unchecked();
        if (r.ndim() != 3) throw std::domain_error("error: ndim != 3");
        for (ssize_t i = 0; i < r.shape(0); i++)
        for (ssize_t j = 0; j < r.shape(1); j++)
        for (ssize_t k = 0; k < r.shape(2); k++)
            r(i, j, k) = start++;
        return a;
    });
    sm.def("proxy_auxiliaries2_dyn", [](py::array_t<double> a) {
        return auxiliaries(a.unchecked(), a.mutable_unchecked());
    });

    sm.def("array_auxiliaries2", [](py::array_t<double> a) {
        return auxiliaries(a, a);
    });

    // test_array_failures
    // Issue #785: Uninformative "Unknown internal error" exception when constructing array from empty object:
    sm.def("array_fail_test", []() { return py::array(py::object()); });
    sm.def("array_t_fail_test", []() { return py::array_t<double>(py::object()); });
    // Make sure the error from numpy is being passed through:
    sm.def("array_fail_test_negative_size", []() { int c = 0; return py::array(-1, &c); });

    // test_initializer_list
    // Issue (unnumbered; reported in #788): regression: initializer lists can be ambiguous
    sm.def("array_initializer_list1", []() { return py::array_t<float>(1); }); // { 1 } also works, but clang warns about it
    sm.def("array_initializer_list2", []() { return py::array_t<float>({ 1, 2 }); });
    sm.def("array_initializer_list3", []() { return py::array_t<float>({ 1, 2, 3 }); });
    sm.def("array_initializer_list4", []() { return py::array_t<float>({ 1, 2, 3, 4 }); });

    // test_array_resize
    // reshape array to 2D without changing size
    sm.def("array_reshape2", [](py::array_t<double> a) {
        const ssize_t dim_sz = (ssize_t)std::sqrt(a.size());
        if (dim_sz * dim_sz != a.size())
            throw std::domain_error("array_reshape2: input array total size is not a squared integer");
        a.resize({dim_sz, dim_sz});
    });

    // resize to 3D array with each dimension = N
    sm.def("array_resize3", [](py::array_t<double> a, size_t N, bool refcheck) {
        a.resize({N, N, N}, refcheck);
    });

    // test_array_create_and_resize
    // return 2D array with Nrows = Ncols = N
    sm.def("create_and_resize", [](size_t N) {
        py::array_t<double> a;
        a.resize({N, N});
        std::fill(a.mutable_data(), a.mutable_data() + a.size(), 42.);
        return a;
    });

#if PY_MAJOR_VERSION >= 3
        sm.def("index_using_ellipsis", [](py::array a) {
            return a[py::make_tuple(0, py::ellipsis(), 0)];
        });
#endif
}
