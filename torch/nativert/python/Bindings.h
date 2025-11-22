#pragma once

#include <torch/csrc/utils/pybind.h>

namespace py = pybind11;

namespace torch {
namespace nativert {

void initModelRunnerPybind(pybind11::module& m);

} // namespace nativert
} // namespace torch
