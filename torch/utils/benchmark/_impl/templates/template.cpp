/* C++ template for Timer methods.

This template will replace:
    `GLOBAL_SETUP_TEMPLATE_LOCATION`,
    `SETUP_TEMPLATE_LOCATION`
      and
    `STMT_TEMPLATE_LOCATION`
sections with user provided statements.
*/

#include <chrono>

#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <c10/util/irange.h>
#include <callgrind.h>

// Global setup. (e.g. #includes)
// GLOBAL_SETUP_TEMPLATE_LOCATION

void call(int n_iter) {
    pybind11::gil_scoped_release no_gil;
    // SETUP_TEMPLATE_LOCATION
    for(const auto loop_idx : c10::irange(n_iter)) {
        // STMT_TEMPLATE_LOCATION
    }
}

float measure_wall_time(int n_iter, int n_warmup_iter, bool cuda_sync) {
    pybind11::gil_scoped_release no_gil;
    // SETUP_TEMPLATE_LOCATION

    for(const auto warmup_idx : c10::irange(n_warmup_iter)) {
        // STMT_TEMPLATE_LOCATION
    }

    if (cuda_sync) torch::cuda::synchronize();
    auto start_time = std::chrono::high_resolution_clock::now();

    for(const auto warmup_idx : c10::irange(n_iter)) {
        // STMT_TEMPLATE_LOCATION
    }

    if (cuda_sync) torch::cuda::synchronize();
    auto end_time = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double>(end_time - start_time).count();
}

void collect_callgrind(int n_iter, int n_warmup_iter) {
    // NB:
    //  There is no `pybind11::gil_scoped_release`, because Callgrind is
    //  process level instrumentation and we don't want Python to context
    //  switch and perturb the measurement. (And we don't expect concurrent
    //  measurements.)

    // SETUP_TEMPLATE_LOCATION

    for(const auto warmup_idx : c10::irange(n_warmup_iter)) {
        // STMT_TEMPLATE_LOCATION
    }

    // torch._C._valgrind_toggle()
    CALLGRIND_TOGGLE_COLLECT;

    for(const auto warmup_idx : c10::irange(n_iter)) {
        // STMT_TEMPLATE_LOCATION
    }

    // torch._C._valgrind_toggle_and_dump_stats()
    CALLGRIND_TOGGLE_COLLECT;
    CALLGRIND_DUMP_STATS;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "call",
        &call,
        py::arg("n_iter"));
    m.def(
        "measure_wall_time",
        &measure_wall_time,
        py::arg("n_iter"),
        py::arg("n_warmup_iter"),
        py::arg("cuda_sync"));
    m.def(
        "collect_callgrind",
        &collect_callgrind,
        py::arg("n_iter"),
        py::arg("n_warmup_iter"));
}
