
#pragma once

#include <stdarg.h>
#include <stdio.h>

enum { GL_ERR = -1, GL_LOG = 0, GL_VERBOSE = 1 };

static constexpr int GL_LOG_LEVEL = GL_LOG;

static inline int gl_log(int level, const char* format, ...) {
  int r = 0;
  if (level <= GL_LOG_LEVEL) {
    va_list args;
    va_start(args, format);
    r = vfprintf(stderr, format, args);
    va_end(args);
  }
  return r;
}
