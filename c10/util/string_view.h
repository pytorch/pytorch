#pragma once

#include <functional>
#include <string_view>

namespace c10 {

struct TransparentStringHash {
  using is_transparent = void;

  size_t operator()(std::string_view value) const noexcept {
    return std::hash<std::string_view>{}(value);
  }
};

} // namespace c10
