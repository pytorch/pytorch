#include <c10/util/C++17.h>
#include <gtest/gtest.h>

namespace {

namespace test_min {
using c10::guts::min;
static_assert(min(3, 5) == 3, "");
static_assert(min(5, 3) == 3, "");
static_assert(min(3, 3) == 3, "");
static_assert(min(3.0, 3.1) == 3.0, "");
} // namespace test_min

namespace test_max {
using c10::guts::max;
static_assert(max(3, 5) == 5, "");
static_assert(max(5, 3) == 5, "");
static_assert(max(3, 3) == 3, "");
static_assert(max(3.0, 3.1) == 3.1, "");
} // namespace test_max

} // namespace
