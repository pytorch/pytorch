#include "OpenRegException.h"

void orCheckFail(
    const char* func,
    const char* file,
    uint32_t line,
    const char* msg) {
  // The point of this helper is to report the *caller's* location, which is
  // passed in; a TORCH_CHECK here would stamp this file's own __FILE__/__LINE__
  // instead.
  // @allow-raw-throw: re-raises with a caller-supplied SourceLocation
  throw ::c10::Error({func, file, line}, msg);
}
