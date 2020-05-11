#pragma once

#include <torch/csrc/THP_export.h>

namespace torch {
namespace utils {

struct THP_CLASS CPPStackTraces {
  static bool is_enabled() {
    return _enabled;
  }
  static void set_enabled(bool enabled) {
    _enabled = enabled;
  }

private:
  static bool _enabled;
};

} // namespace utils
} // namespace torch
