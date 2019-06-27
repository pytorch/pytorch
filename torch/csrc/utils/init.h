#pragma once

#include <torch/csrc/utils/pybind.h>

namespace torch {
namespace throughput_benchmark {

void initThroughputBenchmarkBindings(PyObject* module);

} // namespace throughput_benchmark
} // namespace torch
