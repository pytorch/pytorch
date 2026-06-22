#pragma once

#include <torch/csrc/utils/pybind.h>

namespace torch::profiler::impl {

// Registers the torch._C._profiler._cupti_monitor submodule on `m` (the
// _profiler module). The GIL-free buffer pool, native decode worker, and
// metadata store these bindings expose live in monitor_native.h.
void initCuptiMonitorBindings(pybind11::module& m);

} // namespace torch::profiler::impl
