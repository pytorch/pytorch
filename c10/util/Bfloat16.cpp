#include <c10/util/BFloat16.h>
#include <ostream>
#include <type_traits>

namespace c10 {

static_assert(
    std::is_standard_layout_v<BFloat16>,
    "c10::BFloat16 must be standard layout.");

std::ostream& operator<<(std::ostream& out, const BFloat16& value) {
  out << (float)value;
  return out;
}
} // namespace c10
