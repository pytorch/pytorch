#include <c10/util/Half.h>
#include <iostream>

namespace c10 {

static_assert(
    std::is_standard_layout<Half>::value,
    "c10::Half must be standard layout.");

std::ostream& operator<<(std::ostream& out, const Half& value) {
  out << (float)value;
  return out;
}
} // namespace c10
