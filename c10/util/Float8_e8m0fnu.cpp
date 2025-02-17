#include <c10/macros/Macros.h>
#include <c10/util/Float8_e8m0fnu.h>

namespace c10 {

static_assert(
    std::is_standard_layout_v<Float8_e8m0fnu>,
    "c10::Float8_e8m0fnu must be standard layout.");

} // namespace c10
