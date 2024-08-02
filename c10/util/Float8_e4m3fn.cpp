#include <c10/util/Float8_e4m3fn.h>
#include <type_traits>

namespace c10 {

static_assert(
    std::is_standard_layout_v<Float8_e4m3fn>,
    "c10::Float8_e4m3fn must be standard layout.");

} // namespace c10
