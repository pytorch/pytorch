#include <pybind11/pybind11.h>
#include <callgrind.h>

namespace py = pybind11;

void start() {
    CALLGRIND_START_INSTRUMENTATION;
}

void stop() {
    CALLGRIND_STOP_INSTRUMENTATION;
}

void zero() {
    CALLGRIND_ZERO_STATS;
}

PYBIND11_MODULE(bindings, m) {
    m.def("start", &start);
    m.def("stop", &stop);
    m.def("zero", &zero);
}
