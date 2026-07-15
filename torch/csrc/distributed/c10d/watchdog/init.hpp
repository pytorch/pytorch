#pragma once

#include <torch/csrc/utils/pybind.h>

namespace c10d::watchdog {

// Registers the `_distributed_c10d_watchdog` pybind submodule on the given
// parent module (torch._C).
void initWatchdogBindings(pybind11::module& module);

} // namespace c10d::watchdog
