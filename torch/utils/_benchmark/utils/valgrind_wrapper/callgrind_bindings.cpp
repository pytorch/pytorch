#include <pybind11/pybind11.h>
#include <valgrind/callgrind.h>

namespace py = pybind11;

bool supported_platform(){
    #if defined(NVALGRIND)
    return false;
    #else
    return true;
    #endif
}

void toggle() {
    #if defined(NVALGRIND)
    TORCH_CHECK(false, "Valgrind is not supported.");
    #else
    CALLGRIND_TOGGLE_COLLECT;
    #endif
}

PYBIND11_MODULE(callgrind_bindings, m) {
    m.doc() = "Wraps NVALGRIND and CALLGRIND_TOGGLE_COLLECT symbols in Valgrind/Callgrind.";
    m.def("supported_platform", &supported_platform);
    m.def("toggle", &toggle);
}
