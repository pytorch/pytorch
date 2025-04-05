#pragma once
#include <pybind11/pybind11.h>

namespace torch::nativert {
void initPt2ArchiveConstantsPybind(pybind11::module& m);
} // namespace torch::nativert
