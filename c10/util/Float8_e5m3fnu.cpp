#include <c10/macros/Macros.h>
#include <c10/util/Float8_e5m3fnu.h>

namespace c10 {

static_assert(
    std::is_standard_layout_v<Float8_e5m3fnu>,
    "c10::Float8_e5m3fnu must be standard layout.");

} // namespace c10
