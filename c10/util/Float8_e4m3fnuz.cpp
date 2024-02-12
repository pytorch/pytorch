#include <c10/util/Float8_e4m3fnuz.h>
#include <iostream>

namespace c10 {

static_assert(
    std::is_standard_layout_v<Float8_e4m3fnuz>,
    "c10::Float8_e4m3fnuz must be standard layout.");

std::ostream& operator<<(std::ostream& out, const Float8_e4m3fnuz& value) {
  out << (float)value;
  return out;
}

} // namespace c10
