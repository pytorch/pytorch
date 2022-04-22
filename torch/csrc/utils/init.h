#pragma once

#include <torch/csrc/utils/pybind.h>

namespace torch {
namespace throughput_benchmark {

void initThroughputBenchmarkBindings(PyObject* module);

} // namespace throughput_benchmark

namespace crash_handler {
void initCrashHandlerBindings(PyObject* module);

} // namespace crash_handler
} // namespace torch
