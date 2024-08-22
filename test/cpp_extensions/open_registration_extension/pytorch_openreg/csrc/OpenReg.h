#pragma once
// Shared header for OpenReg module

#include <torch/csrc/utils/pybind.h>

namespace openreg {

    void set_impl_registry(PyObject* registry);

}