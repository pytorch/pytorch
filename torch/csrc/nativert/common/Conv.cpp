#include <torch/csrc/nativert/common/Conv.h>

#include <cerrno>
#include <cstdlib>
#include <string>

namespace torch::nativert {

template <>
std::optional<int64_t> tryTo<int64_t>(std::string_view symbol) {
  // TODO Using strtoll for portability. Consider using std::from_chars in the
  // future.
  std::string symbol_str(symbol);
  char* end;
  errno = 0;
  int64_t value = strtoll(symbol_str.c_str(), &end, 10);
  if (errno != 0) {
    errno = 0;
    return std::nullopt;
  }
  if (end != symbol_str.c_str() + symbol_str.size()) {
    return std::nullopt;
  }
  return value;
}

template <>
std::optional<double> tryTo<double>(std::string_view symbol) {
  // TODO Using strtod for portability. Consider using std::from_chars in the
  // future.
  std::string symbol_str(symbol);
  char* end;
  errno = 0;
  double value = strtod(symbol_str.c_str(), &end);
  if (errno != 0) {
    errno = 0;
    return std::nullopt;
  }
  if (end != symbol_str.c_str() + symbol_str.size()) {
    return std::nullopt;
  }
  return value;
}

} // namespace torch::nativert
