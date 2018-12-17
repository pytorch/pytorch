#pragma once
#include <fstream>

#include "caffe2/core/logging.h"

namespace caffe2 {
namespace emulator {

/*
 * Replace a @substring in a given @line with @target
 */
inline std::string replace(
    std::string line,
    const std::string& substring,
    const std::string& target) {
  size_t index = 0;
  while (true) {
    index = line.find(substring, index);
    if (index == std::string::npos) {
      break;
    }
    line.replace(index, substring.length(), target);
    index += substring.length();
  }
  return line;
}

/*
 * Split given @str into a vector of strings delimited by @delim
 */
inline std::vector<std::string> split(const string& str, const string& delim) {
  std::vector<std::string> tokens;
  size_t prev = 0, pos = 0;
  do {
    pos = str.find(delim, prev);
    if (pos == std::string::npos) {
      pos = str.length();
    }
    std::string token = str.substr(prev, pos - prev);
    if (!token.empty()) {
      tokens.push_back(token);
    }
    prev = pos + delim.length();
  } while (pos < str.length() && prev < str.length());
  return tokens;
}

/*
 * Check if the given @path is valid.
 * Remove the file/folder if @remove is specified
 */
inline bool check_path_valid(std::string path, bool remove = true) {
  CAFFE_ENFORCE(!path.empty());
  std::ifstream file(path.c_str());
  // The file should exist or the path is valid
  if (!file.good() && !static_cast<bool>(std::ofstream(path).put('t'))) {
    return false;
  }
  file.close();
  if (remove) {
    std::remove(path.c_str());
  }
  return true;
}

} // namespace emulator
} // namespace caffe2
