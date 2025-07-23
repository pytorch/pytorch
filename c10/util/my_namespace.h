#pragma once

#include <c10/util/Half.h>
#include <ostream>

namespace my_namespace {

// Reexpose the Half type from c10
using Half = c10::Half;

// Define the operator<< for Half in my_namespace
inline std::ostream& operator<<(std::ostream& out, const Half& value) {
  // Convert to float and output
  out << static_cast<float>(value);
  return out;
}

} // namespace my_namespace