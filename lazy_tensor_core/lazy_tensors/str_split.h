#pragma once

#include <c10/util/string_view.h>

#include <string>
#include <vector>

namespace lazy_tensors {

inline std::vector<std::string> StrSplit(c10::string_view text, char delim) {
  size_t start;
  size_t end = 0;

  std::vector<std::string> tokens;
  while ((start = text.find_first_not_of(delim, end)) != std::string::npos) {
    end = text.find(delim, start);
    auto token = text.substr(start, end - start);
    tokens.emplace_back(token.begin(), token.end());
  }
  return tokens;
}

}  // namespace lazy_tensors
