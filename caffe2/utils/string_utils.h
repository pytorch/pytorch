#pragma once

#include <algorithm>
#include <memory>
#include <string>
#include <vector>

namespace caffe2 {

std::vector<std::string> split(char separator, const std::string& string);

std::string trim(const std::string& str);

size_t editDistance(
  const std::string& s1, const std::string& s2, size_t max_distance = 0);

inline bool StartsWith(const std::string& str, const std::string& prefix) {
  return std::mismatch(prefix.begin(), prefix.end(), str.begin()).first ==
      prefix.end();
}

int32_t editDistanceHelper(const char* s1,
  size_t s1_len,
  const char* s2,
  size_t s2_len,
  std::vector<size_t> &current,
  std::vector<size_t> &previous,
  std::vector<size_t> &previous1,
  size_t max_distance);
} // namespace caffe2
