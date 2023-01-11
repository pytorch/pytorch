#pragma once

#include <c10/util/Exception.h>
#include <c10/util/Optional.h>
#include <cstring>

namespace c10 {
namespace utils {
// Reads an environment variable and returns
// - optional<true>,              if set equal to "1"
// - optional<false>,             if set equal to "0"
// - nullopt,   otherwise
//
// NB:
// Issues a warning if the value of the environment variable is not 0 or 1.
inline optional<bool> check_env(const char* name) {
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4996)
#endif
  auto envar = std::getenv(name);
#ifdef _MSC_VER
#pragma warning(pop)
#endif
  if (envar) {
    if (strcmp(envar, "0") == 0) {
      return false;
    }
    if (strcmp(envar, "1") == 0) {
      return true;
    }
    TORCH_WARN(
        "Ignoring invalid value for boolean flag ",
        name,
        ": ",
        envar,
        "valid values are 0 or 1.");
  }
  return c10::nullopt;
}
} // namespace utils
} // namespace c10
