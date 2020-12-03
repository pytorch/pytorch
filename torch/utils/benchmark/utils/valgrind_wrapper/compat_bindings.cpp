/* Used to collect profiles of old versions of PyTorch. */
#include <callgrind.h>
#include <pybind11/pybind11.h>


bool _valgrind_supported_platform() {
    #if defined(NVALGRIND)
    return false;
    #else
    return true;
    #endif
}

void _valgrind_toggle() {
    #if defined(NVALGRIND)
    TORCH_CHECK(false, "Valgrind is not supported.");
    #else
    CALLGRIND_TOGGLE_COLLECT;
    #endif
}

PYBIND11_MODULE(callgrind_bindings, m) {
    m.def("_valgrind_supported_platform", &_valgrind_supported_platform);
    m.def("_valgrind_toggle", &_valgrind_toggle);
}
