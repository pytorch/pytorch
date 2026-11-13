#pragma once

#include <torch/csrc/utils/pybind.h>

namespace torch::throughput_benchmark {

void initThroughputBenchmarkBindings(PyObject* module);

} // namespace torch::throughput_benchmark
