#pragma once

#include <torch/csrc/utils/pybind.h>

namespace py = pybind11;

#if USE_DISTRIBUTED
void initProcessGroupBindings(py::module& m);
#endif
