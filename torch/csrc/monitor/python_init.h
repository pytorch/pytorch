#pragma once

#include <torch/csrc/utils/pybind.h>

namespace torch {
namespace monitor {

void initMonitorBindings(PyObject* module);

}
} // namespace torch
