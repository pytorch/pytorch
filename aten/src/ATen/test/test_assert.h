#pragma once
#include <stdexcept>
#include <stdarg.h>

static inline void barf(const char *fmt, ...) {
  char msg[2048];
  va_list args;
  va_start(args, fmt);
  vsnprintf(msg, 2048, fmt, args);
  va_end(args);
  throw std::runtime_error(msg);
}

#define ASSERT(cond) \
  if (__builtin_expect(!(cond), 0)) { \
    barf("%s:%u: %s: Assertion `%s` failed.", __FILE__, __LINE__, __func__, #cond); \
  }

//note: msg must be a string literal
//node: In, ##__VA_ARGS '##' supresses the comma if __VA_ARGS__ is empty
#define ASSERTM(cond, msg, ...) \
  if (__builtin_expect(!(cond), 0)) { \
    barf("%s:%u: %s: Assertion `%s` failed: " msg , __FILE__, __LINE__, __func__, #cond,##__VA_ARGS__); \
  }
