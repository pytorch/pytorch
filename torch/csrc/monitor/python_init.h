#pragma once

#include <torch/csrc/utils/pybind.h>

namespace torch::monitor {

void initMonitorBindings(PyObject* module);

}
