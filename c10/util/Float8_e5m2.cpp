#include <c10/util/Float8_e5m2.h>

namespace c10 {

static_assert(
    std::is_standard_layout_v<Float8_e5m2>,
    "c10::Float8_e5m2 must be standard layout.");

} // namespace c10
