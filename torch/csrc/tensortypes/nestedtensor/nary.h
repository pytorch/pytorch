#pragma once

#include <pybind11/pybind11.h>

namespace torch {
namespace tensortypes {
namespace nestedtensor {

void add_nary_functions(py::module& m);

} // namespace nestedtensor
} // namespace tensortypes
} // namespace torch
