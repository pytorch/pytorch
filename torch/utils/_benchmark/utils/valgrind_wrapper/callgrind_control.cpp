#include <pybind11/pybind11.h>
#include <callgrind.h>

namespace py = pybind11;

void toggle() {
    CALLGRIND_TOGGLE_COLLECT;
}

PYBIND11_MODULE(bindings, m) {
    m.def("toggle", &toggle);
}
