/* C++ template for Timer.timeit

This template will be consumed by `cpp_jit.py`, and will replace:
    `GLOBAL_SETUP_TEMPLATE_LOCATION`,
    `SETUP_TEMPLATE_LOCATION`
      and
    `STMT_TEMPLATE_LOCATION`
sections with user provided statements.
*/
#include <chrono>

#include <pybind11/pybind11.h>
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
    for (int loop_idx = 0; loop_idx < n; loop_idx++) {
        // STMT_TEMPLATE_LOCATION
    }
    auto end_time = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double>(end_time - start_time).count();
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("timeit", &timeit);
}
