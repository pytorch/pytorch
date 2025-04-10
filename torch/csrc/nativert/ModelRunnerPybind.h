#pragma once

#include <torch/csrc/utils/pybind.h> // @manual=//caffe2:torch-cpp

namespace py = pybind11;

namespace torch {
namespace nativert {

void initModelRunnerPybind(pybind11::module& m);

} // namespace nativert
} // namespace torch
