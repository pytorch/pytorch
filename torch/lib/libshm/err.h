#pragma once

#include <system_error>
#include <cerrno>

#define SYSCHECK(expr)                                      \
{                                                           \
  do {                                                      \
    errno = 0;                                              \
    auto ___output = (expr);                                \
    (void)___output;                                        \
    } while (errno == EINTR);                               \
  if (errno != 0)                                           \
    throw std::system_error(errno, std::system_category()); \
}
