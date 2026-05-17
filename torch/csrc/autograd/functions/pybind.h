#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/csrc/python_headers.h>
#include <torch/csrc/utils/pybind.h>

#include <torch/csrc/autograd/python_cpp_function.h>
#include <torch/csrc/autograd/python_function.h>

// NOLINTNEXTLINE(misc-unused-alias-decls)
namespace py = pybind11;

namespace pybind11::detail {} // namespace pybind11::detail
