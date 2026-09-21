#pragma once

#include <include/openreg.h>

#include <c10/util/Exception.h>

void orCheckFail(const char* func, const char* file, uint32_t line, const char* msg = "");

#define OPENREG_CHECK(EXPR, ...)                                                       \
  do {                                                                                 \
    const orError_t __err = EXPR;                                                      \
    if (C10_UNLIKELY(__err != orSuccess)) {                                            \
      orCheckFail(__func__, __FILE__, static_cast<uint32_t>(__LINE__), ##__VA_ARGS__); \
    }                                                                                  \
  } while (0)
