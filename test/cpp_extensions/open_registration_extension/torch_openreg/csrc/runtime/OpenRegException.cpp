#include "OpenRegException.h"

void orCheckFail(
    const char* func,
    const char* file,
    uint32_t line,
    const char* msg) {
  throw ::c10::Error({func, file, line}, msg);
}
