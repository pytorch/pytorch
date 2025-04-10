#pragma once
#include <torch/csrc/utils/pybind.h> // @manual=//caffe2:torch-cpp

namespace torch::nativert {
void initPt2ArchiveConstantsPybind(pybind11::module& m);
} // namespace torch::nativert
