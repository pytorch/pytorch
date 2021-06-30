/* C++ template for Timer.timeit

This template will be consumed by `cpp_jit.py`, and will replace:
    `GLOBAL_SETUP_TEMPLATE_LOCATION`,
    `SETUP_TEMPLATE_LOCATION`
      and
    `STMT_TEMPLATE_LOCATION`
sections with user provided statements.
*/
#include <chrono>

#include <c10/util/irange.h>
#include <pybind11/pybind11.h>
#include <c10/util/irange.h>
#include <torch/extension.h>

// Global setup. (e.g. #includes)
// GLOBAL_SETUP_TEMPLATE_LOCATION

double timeit(int n) {
    pybind11::gil_scoped_release no_gil;

    // Setup
    // SETUP_TEMPLATE_LOCATION

    {
        // Warmup
        // STMT_TEMPLATE_LOCATION
    }

    // Main loop
    auto start_time = std::chrono::high_resolution_clock::now();
    for(const auto loop_idx : c10::irange(n)) {
        // STMT_TEMPLATE_LOCATION
    }
    auto end_time = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double>(end_time - start_time).count();
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("timeit", &timeit);
}
