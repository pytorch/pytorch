#pragma once

#include "torch/csrc/jit/pybind.h"

#include <tuple>

namespace torch { namespace jit { namespace python {

// (in_vars, in_key, is_volatile)
using flattened_args = std::tuple<py::tuple, py::bytes, bool>;

flattened_args flatten(py::handle obj);
py::object unflatten(py::tuple vars, py::bytes descriptor);

}}}
