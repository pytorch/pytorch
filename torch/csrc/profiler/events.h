#pragma once

#include <array>
#include <cstdint>
#include <cstring>
#include <vector>

namespace torch::profiler {

/* A vector type to hold a list of performance counters */
using perf_counters_t = std::vector<uint64_t>;

/* Standard list of performance events independent of hardware or backend */
constexpr std::array<const char*, 2> ProfilerPerfEvents = {
    /*
     * Number of Processing Element (PE) cycles between two points of interest
     * in time. This should correlate positively with wall-time. Measured in
     * uint64_t. PE can be non cpu. TBD reporting behavior for multiple PEs
     * participating (i.e. threadpool).
     */
    "cycles",

    /* Number of PE instructions between two points of interest in time. This
     * should correlate positively with wall time and the amount of computation
     * (i.e. work). Across repeat executions, the number of instructions should
     * be more or less invariant. Measured in uint64_t. PE can be non cpu.
     */
    "instructions"};
} // namespace torch::profiler
