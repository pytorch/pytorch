/*
    tests/test_gil_scoped.cpp -- acquire and release gil

    Copyright (c) 2017 Borja Zarco (Google LLC) <bzarco@google.com>

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE file.
*/

#include "pybind11_tests.h"
#include <pybind11/functional.h>


class VirtClass  {
public:
    virtual void virtual_func() {}
    virtual void pure_virtual_func() = 0;
};

class PyVirtClass : public VirtClass {
    void virtual_func() override {
        PYBIND11_OVERLOAD(void, VirtClass, virtual_func,);
    }
    void pure_virtual_func() override {
        PYBIND11_OVERLOAD_PURE(void, VirtClass, pure_virtual_func,);
    }
};

TEST_SUBMODULE(gil_scoped, m) {
  py::class_<VirtClass, PyVirtClass>(m, "VirtClass")
      .def(py::init<>())
      .def("virtual_func", &VirtClass::virtual_func)
      .def("pure_virtual_func", &VirtClass::pure_virtual_func);

    m.def("test_callback_py_obj",
          [](py::object func) { func(); });
    m.def("test_callback_std_func",
          [](const std::function<void()> &func) { func(); });
    m.def("test_callback_virtual_func",
          [](VirtClass &virt) { virt.virtual_func(); });
    m.def("test_callback_pure_virtual_func",
          [](VirtClass &virt) { virt.pure_virtual_func(); });
}
