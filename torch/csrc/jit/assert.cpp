#include "torch/csrc/jit/assert.h"

#include <cstdarg>
#include <cstdio>

namespace torch { namespace jit {

void
barf(const char *fmt, ...)
{
  char msg[2048];
  va_list args;

  va_start(args, fmt);
  vsnprintf(msg, 2048, fmt, args);
  va_end(args);

  // Convenience for gdb; helps you avoid fumbling around
  // to figure out how to trap on throwing a C++ exception.
  __builtin_trap();

  throw assert_error(msg);
}

}}
