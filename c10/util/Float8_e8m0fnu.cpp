#include <c10/macros/Macros.h>
#include <c10/util/Float8_e8m0fnu.h>

namespace c10 {

// TODO(#146647): Can we have these in a single shared cpp file
// built with macro to remove the need for a new cpp file?
static_assert(
    std::is_standard_layout_v<Float8_e8m0fnu>,
    "c10::Float8_e8m0fnu must be standard layout.");

} // namespace c10
