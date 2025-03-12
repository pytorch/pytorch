#include "torch/csrc/nativert/common/Conv.h"

#include <charconv>
#include <string>

namespace torch::nativert {

template <>
std::optional<int64_t> tryTo<int64_t>(std::string_view symbol) {
  int64_t value;
  auto [ptr, ec] =
      std::from_chars(symbol.data(), symbol.data() + symbol.size(), value);
  if (ec != std::errc()) {
    return std::nullopt;
  }
  if (ptr != symbol.data() + symbol.size()) {
    return std::nullopt;
  }
  return value;
}

template <>
std::optional<double> tryTo<double>(std::string_view symbol) {
  double value;
#ifdef __APPLE__
  char extra; // to detect any extra characters after the number
  // Try to parse the string using sscanf
  auto str = std::string{symbol};
  if (sscanf(str.c_str(), "%lf %c", &value, &extra) != 1) {
    // If sscanf returns anything other than 1, it means parsing failed or there
    // were extra characters
    return std::nullopt;
  }
#else
  auto [ptr, ec] =
      std::from_chars(symbol.data(), symbol.data() + symbol.size(), value);
  if (ec != std::errc()) {
    return std::nullopt;
  }
  if (ptr != symbol.data() + symbol.size()) {
    return std::nullopt;
  }
#endif
  return value;
}

} // namespace torch::nativert
