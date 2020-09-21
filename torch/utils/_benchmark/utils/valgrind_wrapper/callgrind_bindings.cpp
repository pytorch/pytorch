#include <callgrind.h>

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
