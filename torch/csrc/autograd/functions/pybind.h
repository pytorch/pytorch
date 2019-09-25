#pragma once

#include <torch/csrc/python_headers.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <torch/csrc/autograd/python_function.h>
#include <torch/csrc/autograd/python_cpp_function.h>

namespace py = pybind11;

namespace pybind11 { namespace detail {

}} // namespace pybind11::detail
