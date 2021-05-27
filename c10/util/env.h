#pragma once

#include <torch/csrc/Exceptions.h>
#include <iostream>
#include <sstream>
#include <cstring>

namespace c10 {
namespace utils {
  // Reads an environment variable and returns
  // - true,              if set equal to "1"
  // - false,             if set equal to "0"
  // - `default_value`,   otherwise
  //
  // NB:
  // Issues a warning if the value of the environment variable is not 0 or 1.
  bool check_env(const char *name, bool default_value = false) {
    auto envar = std::getenv(name);
    if (envar) {
      if (strcmp(envar, "0") == 0) {
        return false;
      }
      if (strcmp(envar, "1") == 0) {
        return true;
      }
      TORCH_WARN("Ignoring invalid value for boolean flag ", name, ": ", envar,
          "valid values are 0 or 1.");
    }
    return default_value;
  }

  // Reads an environment variable and returns
  // - its value,         if it is set and is a valid value
  // - `default_value`,   otherwise
  //
  // You can optionally pass in a list of valid values (default: empty list,
  // which is interpreted as "all values accepted")
  std::string check_env(const char *name,
      const char *default_value = "UNSET",
      const std::vector<std::string> valid_values= std::vector<std::string>()) {
    auto envar = std::getenv(name);

    // Check if envar is in the set of valid values (if any)
    bool found = false;
    for (auto &val: valid_values) {
      if (val.compare(envar) == 0) {
        found = true;
      }
    }

    // Issue a warning if an invalid value was passed in
    if (valid_values.size() > 0 && !found) {
      std::string valid_values_str;

      std::stringstream ss;
      ss << "Ignoring invalid value for flag " << name << ": " << envar << ". ";
      ss << "Valid values are: ";
      for (auto &val: valid_values) {
        ss << val << ", ";
      }
      ss << ". Using default value " << default_value << "instead.";

      TORCH_WARN(ss.str());

      return default_value;
    }

    return envar;
  }
} // namespace util
} // namespace c10
