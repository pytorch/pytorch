#pragma once

#include <cerrno>
#include <system_error>

// `errno` is only meaningful when it fails. E.g., a  successful `fork()` sets
// `errno` to `EINVAL` in child process on some macos
// (https://stackoverflow.com/a/20295079), and thus `errno` should really only
// be inspected if an error occurred.
//
// All functions used in `libshm` (so far) indicate error by returning `-1`. If
// you want to use a function with a different error reporting mechanism, you
// need to port `SYSCHECK` from `torch/lib/c10d/Utils.hpp`.
#define SYSCHECK_ERR_RETURN_NEG1(expr)                          \
  while (true) {                                                \
    if ((expr) == -1) {                                         \
      if (errno == EINTR) {                                     \
        continue;                                               \
      } else {                                                  \
        throw std::system_error(errno, std::system_category()); \
      }                                                         \
    } else {                                                    \
      break;                                                    \
    }                                                           \
  }
