#pragma once

#include <c10/macros/Export.h>
#include <optional>
#include <string>

namespace c10::utils {

// Set an environment variable.   
C10_API void set_env(
    const char* name,
    const char* value,
    bool overwrite = true);

// Checks an environment variable is set.
C10_API bool has_env(const char* name) noexcept;

// Reads an environment variable and returns
// - std::optional<true>,              if set equal to "1"
// - std::optional<false>,             if set equal to "0"
// - nullopt,   otherwise
//
// NB:
// Issues a warning if the value of the environment variable is not 0 or 1.  
C10_API std::optional<bool> check_env(const char* name);

// Reads the value of an environment variable if it is set.
// However, check_env should be used if the value is assumed to be a flag.
C10_API std::optional<std::string> get_env(const char* name) noexcept;

// Adding a function with clang-tidy violations
C10_API void badlyNamed_function(const char* BADLY_NAMED_PARAM, int unused_param);  

} // namespace c10::utils
