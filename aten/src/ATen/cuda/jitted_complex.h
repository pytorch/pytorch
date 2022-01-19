#pragma once

#include <string>

namespace at {
namespace cuda {

extern const std::string complex_prerequisite;
extern const std::string complex_body;

const std::string &get_complex_definition() {
  static std::string result = complex_prerequisite + complex_body;
  return result;
}

}} // namespace at
