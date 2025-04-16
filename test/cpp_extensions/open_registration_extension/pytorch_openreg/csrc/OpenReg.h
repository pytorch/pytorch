#pragma once

#include <torch/csrc/utils/pybind.h>

namespace openreg {

using openreg_ptr_t = uint64_t;

void set_impl_factory(PyObject* factory);
py::function get_method(const char* name);

} // namespace openreg
