#include "torch/csrc/nativert/common/String.h"

#include <sstream>

namespace torch::nativert {

std::vector<std::string_view> split(std::string_view target, char delimiter) {
  std::vector<std::string_view> atoms;
  std::string_view buffer = target;
  while (buffer.size() > 0) {
    auto i = buffer.find(delimiter);
    if (i == std::string_view::npos) {
      atoms.push_back(buffer);
      buffer.remove_prefix(buffer.size());
    } else {
      atoms.push_back(buffer.substr(0, i));
      buffer.remove_prefix(i + 1);
    }
  }
  return atoms;
}

std::string join(
    std::string_view delimiter,
    const std::vector<std::string>& keys) {
  std::ostringstream result;
  for (size_t i = 0; i < keys.size(); i++) {
    result << keys[i];
    if (i != keys.size() - 1) {
      result << delimiter;
    }
  }
  return result.str();
}

bool starts_with(std::string_view str, std::string_view prefix) {
  return str.size() >= prefix.size() &&
      str.compare(0, prefix.size(), prefix) == 0;
}

bool ends_with(std::string_view str, std::string_view suffix) {
  return str.size() >= suffix.size() &&
      str.compare(str.size() - suffix.size(), suffix.size(), suffix) == 0;
}

} // namespace torch::nativert
