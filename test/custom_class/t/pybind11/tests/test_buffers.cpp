/*
    tests/test_buffers.cpp -- supporting Pythons' buffer protocol

    Copyright (c) 2016 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE file.
*/

#include "pybind11_tests.h"
#include "constructor_stats.h"

TEST_SUBMODULE(buffers, m) {
    // test_from_python / test_to_python:
    class Matrix {
    public:
        Matrix(ssize_t rows, ssize_t cols) : m_rows(rows), m_cols(cols) {
            print_created(this, std::to_string(m_rows) + "x" + std::to_string(m_cols) + " matrix");
            m_data = new float[(size_t) (rows*cols)];
            memset(m_data, 0, sizeof(float) * (size_t) (rows * cols));
        }

        Matrix(const Matrix &s) : m_rows(s.m_rows), m_cols(s.m_cols) {
            print_copy_created(this, std::to_string(m_rows) + "x" + std::to_string(m_cols) + " matrix");
            m_data = new float[(size_t) (m_rows * m_cols)];
            memcpy(m_data, s.m_data, sizeof(float) * (size_t) (m_rows * m_cols));
        }

        Matrix(Matrix &&s) : m_rows(s.m_rows), m_cols(s.m_cols), m_data(s.m_data) {
            print_move_created(this);
            s.m_rows = 0;
            s.m_cols = 0;
            s.m_data = nullptr;
        }

        ~Matrix() {
            print_destroyed(this, std::to_string(m_rows) + "x" + std::to_string(m_cols) + " matrix");
            delete[] m_data;
        }

        Matrix &operator=(const Matrix &s) {
            print_copy_assigned(this, std::to_string(m_rows) + "x" + std::to_string(m_cols) + " matrix");
            delete[] m_data;
            m_rows = s.m_rows;
            m_cols = s.m_cols;
            m_data = new float[(size_t) (m_rows * m_cols)];
            memcpy(m_data, s.m_data, sizeof(float) * (size_t) (m_rows * m_cols));
            return *this;
        }

        Matrix &operator=(Matrix &&s) {
            print_move_assigned(this, std::to_string(m_rows) + "x" + std::to_string(m_cols) + " matrix");
            if (&s != this) {
                delete[] m_data;
                m_rows = s.m_rows; m_cols = s.m_cols; m_data = s.m_data;
                s.m_rows = 0; s.m_cols = 0; s.m_data = nullptr;
            }
            return *this;
        }

        float operator()(ssize_t i, ssize_t j) const {
            return m_data[(size_t) (i*m_cols + j)];
        }

        float &operator()(ssize_t i, ssize_t j) {
            return m_data[(size_t) (i*m_cols + j)];
        }

        float *data() { return m_data; }

        ssize_t rows() const { return m_rows; }
        ssize_t cols() const { return m_cols; }
    private:
        ssize_t m_rows;
        ssize_t m_cols;
        float *m_data;
    };
    py::class_<Matrix>(m, "Matrix", py::buffer_protocol())
        .def(py::init<ssize_t, ssize_t>())
        /// Construct from a buffer
        .def(py::init([](py::buffer b) {
            py::buffer_info info = b.request();
            if (info.format != py::format_descriptor<float>::format() || info.ndim != 2)
                throw std::runtime_error("Incompatible buffer format!");

            auto v = new Matrix(info.shape[0], info.shape[1]);
            memcpy(v->data(), info.ptr, sizeof(float) * (size_t) (v->rows() * v->cols()));
            return v;
        }))

       .def("rows", &Matrix::rows)
       .def("cols", &Matrix::cols)

        /// Bare bones interface
       .def("__getitem__", [](const Matrix &m, std::pair<ssize_t, ssize_t> i) {
            if (i.first >= m.rows() || i.second >= m.cols())
                throw py::index_error();
            return m(i.first, i.second);
        })
       .def("__setitem__", [](Matrix &m, std::pair<ssize_t, ssize_t> i, float v) {
            if (i.first >= m.rows() || i.second >= m.cols())
                throw py::index_error();
            m(i.first, i.second) = v;
        })
       /// Provide buffer access
       .def_buffer([](Matrix &m) -> py::buffer_info {
            return py::buffer_info(
                m.data(),                               /* Pointer to buffer */
                { m.rows(), m.cols() },                 /* Buffer dimensions */
                { sizeof(float) * size_t(m.cols()),     /* Strides (in bytes) for each index */
                  sizeof(float) }
            );
        })
        ;


    // test_inherited_protocol
    class SquareMatrix : public Matrix {
    public:
        SquareMatrix(ssize_t n) : Matrix(n, n) { }
    };
    // Derived classes inherit the buffer protocol and the buffer access function
    py::class_<SquareMatrix, Matrix>(m, "SquareMatrix")
        .def(py::init<ssize_t>());


    // test_pointer_to_member_fn
    // Tests that passing a pointer to member to the base class works in
    // the derived class.
    struct Buffer {
        int32_t value = 0;

        py::buffer_info get_buffer_info() {
            return py::buffer_info(&value, sizeof(value),
                                   py::format_descriptor<int32_t>::format(), 1);
        }
    };
    py::class_<Buffer>(m, "Buffer", py::buffer_protocol())
        .def(py::init<>())
        .def_readwrite("value", &Buffer::value)
        .def_buffer(&Buffer::get_buffer_info);


    class ConstBuffer {
        std::unique_ptr<int32_t> value;

    public:
        int32_t get_value() const { return *value; }
        void set_value(int32_t v) { *value = v; }

        py::buffer_info get_buffer_info() const {
            return py::buffer_info(value.get(), sizeof(*value),
                                   py::format_descriptor<int32_t>::format(), 1);
        }

        ConstBuffer() : value(new int32_t{0}) { };
    };
    py::class_<ConstBuffer>(m, "ConstBuffer", py::buffer_protocol())
        .def(py::init<>())
        .def_property("value", &ConstBuffer::get_value, &ConstBuffer::set_value)
        .def_buffer(&ConstBuffer::get_buffer_info);

    struct DerivedBuffer : public Buffer { };
    py::class_<DerivedBuffer>(m, "DerivedBuffer", py::buffer_protocol())
        .def(py::init<>())
        .def_readwrite("value", (int32_t DerivedBuffer::*) &DerivedBuffer::value)
        .def_buffer(&DerivedBuffer::get_buffer_info);

}
