#include <c10/util/Float8_e5m2fnuz.h>
#include <iostream>

namespace c10 {

static_assert(
    std::is_standard_layout_v<Float8_e5m2fnuz>,
    "c10::Float8_e5m2 must be standard layout.");

std::ostream& operator<<(std::ostream& out, const Float8_e5m2fnuz& value) {
  out << (float)value;
  return out;
}

} // namespace c10
