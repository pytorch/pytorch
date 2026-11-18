#include <c10/util/Half.h>
#include <type_traits>

namespace c10 {

static_assert(
    std::is_standard_layout_v<Half>,
    "c10::Half must be standard layout.");

} // namespace c10
