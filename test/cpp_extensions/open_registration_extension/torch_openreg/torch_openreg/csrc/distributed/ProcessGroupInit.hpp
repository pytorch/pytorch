#pragma once

#include <pybind11/pybind11.h>

namespace py = pybind11;

#if USE_DISTRIBUTED
void initProcessGroupBindings(py::module& m);
#endif