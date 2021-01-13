#pragma once

#include <algorithm>
#include <memory>
#include <string>
#include <vector>

#include "caffe2/core/common.h"

namespace caffe2 {

TORCH_API std::vector<std::string>
split(char separator, const std::string& string, bool ignore_empty = false);

TORCH_API std::string trim(const std::string& str);

TORCH_API size_t editDistance(
    const std::string& s1,
    const std::string& s2,
    size_t max_distance = 0);

TORCH_API inline bool StartsWith(
    const std::string& str,
    const std::string& prefix) {
  return str.length() >= prefix.length() &&
      std::mismatch(prefix.begin(), prefix.end(), str.begin()).first ==
      prefix.end();
}

TORCH_API inline bool EndsWith(
    const std::string& full,
    const std::string& ending) {
  if (full.length() >= ending.length()) {
    return (
        0 ==
        full.compare(full.length() - ending.length(), ending.length(), ending));
  } else {
    return false;
  }
}

TORCH_API int32_t editDistanceHelper(
    const char* s1,
    size_t s1_len,
    const char* s2,
    size_t s2_len,
    std::vector<size_t>& current,
    std::vector<size_t>& previous,
    std::vector<size_t>& previous1,
    size_t max_distance);
} // namespace caffe2
