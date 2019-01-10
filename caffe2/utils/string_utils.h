#pragma once

#include <memory>
#include <string>
#include <vector>

namespace caffe2 {

std::vector<std::string> split(char separator, const std::string& string);
size_t editDistance(
  const std::string& s1, const std::string& s2, size_t max_distance = 0);

int32_t editDistanceHelper(const char* s1,
  size_t s1_len,
  const char* s2,
  size_t s2_len,
  std::vector<size_t> &current,
  std::vector<size_t> &previous,
  std::vector<size_t> &previous1,
  size_t max_distance);
} // namespace caffe2
