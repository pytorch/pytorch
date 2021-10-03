#include <c10/util/Array.h>
#include <gtest/gtest.h>

using c10::guts::array;
using c10::guts::to_array;

namespace {
namespace test_equals {
static_assert(array<int, 0>{{}} == array<int, 0>{{}}, "");
static_assert(array<int, 3>{{2, 3, 4}} == array<int, 3>{{2, 3, 4}}, "");
static_assert(!(array<int, 3>{{2, 3, 4}} == array<int, 3>{{1, 3, 4}}), "");
static_assert(!(array<int, 3>{{2, 3, 4}} == array<int, 3>{{2, 1, 4}}), "");
static_assert(!(array<int, 3>{{2, 3, 4}} == array<int, 3>{{2, 3, 1}}), "");
} // namespace test_equals

namespace test_notequals {
static_assert(!(array<int, 0>{{}} != array<int, 0>{{}}), "");
static_assert(!(array<int, 3>{{2, 3, 4}} != array<int, 3>{{2, 3, 4}}), "");
static_assert(array<int, 3>{{2, 3, 4}} != array<int, 3>{{1, 3, 4}}, "");
static_assert(array<int, 3>{{2, 3, 4}} != array<int, 3>{{2, 1, 4}}, "");
static_assert(array<int, 3>{{2, 3, 4}} != array<int, 3>{{2, 3, 1}}, "");
} // namespace test_notequals

namespace test_lessthan {
static_assert(!(array<int, 0>{{}} < array<int, 0>{{}}), "");
static_assert(!(array<int, 1>{{2}} < array<int, 1>{{1}}), "");
static_assert(array<int, 1>{{1}} < array<int, 1>{{2}}, "");
static_assert(!(array<int, 3>{{1, 2, 3}} < array<int, 3>{{1, 2, 3}}), "");
static_assert(array<int, 3>{{1, 2, 3}} < array<int, 3>{{2, 2, 3}}, "");
static_assert(!(array<int, 3>{{1, 2, 3}} < array<int, 3>{{0, 2, 3}}), "");
static_assert(array<int, 3>{{1, 2, 3}} < array<int, 3>{{1, 3, 3}}, "");
static_assert(!(array<int, 3>{{1, 2, 3}} < array<int, 3>{{1, 1, 3}}), "");
static_assert(array<int, 3>{{1, 2, 3}} < array<int, 3>{{1, 2, 4}}, "");
static_assert(!(array<int, 3>{{1, 2, 3}} < array<int, 3>{{1, 2, 2}}), "");
} // namespace test_lessthan

namespace test_greaterthan {
static_assert(!(array<int, 0>{{}} > array<int, 0>{{}}), "");
static_assert(!(array<int, 1>{{1}} > array<int, 1>{{2}}), "");
static_assert(array<int, 1>{{2}} > array<int, 1>{{1}}, "");
static_assert(!(array<int, 3>{{1, 2, 3}} > array<int, 3>{{1, 2, 3}}), "");
static_assert(array<int, 3>{{2, 2, 3}} > array<int, 3>{{1, 2, 3}}, "");
static_assert(!(array<int, 3>{{0, 2, 3}} > array<int, 3>{{1, 2, 3}}), "");
static_assert(array<int, 3>{{1, 3, 3}} > array<int, 3>{{1, 2, 3}}, "");
static_assert(!(array<int, 3>{{1, 1, 3}} > array<int, 3>{{1, 2, 3}}), "");
static_assert(array<int, 3>{{1, 2, 4}} > array<int, 3>{{1, 2, 3}}, "");
static_assert(!(array<int, 3>{{1, 2, 2}} > array<int, 3>{{1, 2, 3}}), "");
} // namespace test_greaterthan

namespace test_lessequals {
static_assert(array<int, 0>{{}} <= array<int, 0>{{}}, "");
static_assert(!(array<int, 1>{{2}} <= array<int, 1>{{1}}), "");
static_assert(array<int, 1>{{1}} <= array<int, 1>{{2}}, "");
static_assert(array<int, 3>{{1, 2, 3}} <= array<int, 3>{{1, 2, 3}}, "");
static_assert(array<int, 3>{{1, 2, 3}} <= array<int, 3>{{2, 2, 3}}, "");
static_assert(!(array<int, 3>{{1, 2, 3}} <= array<int, 3>{{0, 2, 3}}), "");
static_assert(array<int, 3>{{1, 2, 3}} <= array<int, 3>{{1, 3, 3}}, "");
static_assert(!(array<int, 3>{{1, 2, 3}} <= array<int, 3>{{1, 1, 3}}), "");
static_assert(array<int, 3>{{1, 2, 3}} <= array<int, 3>{{1, 2, 4}}, "");
static_assert(!(array<int, 3>{{1, 2, 3}} <= array<int, 3>{{1, 2, 2}}), "");
} // namespace test_lessequals

namespace test_greaterequals {
static_assert(array<int, 0>{{}} >= array<int, 0>{{}}, "");
static_assert(!(array<int, 1>{{1}} >= array<int, 1>{{2}}), "");
static_assert(array<int, 1>{{2}} >= array<int, 1>{{1}}, "");
static_assert(array<int, 3>{{1, 2, 3}} >= array<int, 3>{{1, 2, 3}}, "");
static_assert(array<int, 3>{{2, 2, 3}} >= array<int, 3>{{1, 2, 3}}, "");
static_assert(!(array<int, 3>{{0, 2, 3}} >= array<int, 3>{{1, 2, 3}}), "");
static_assert(array<int, 3>{{1, 3, 3}} >= array<int, 3>{{1, 2, 3}}, "");
static_assert(!(array<int, 3>{{1, 1, 3}} >= array<int, 3>{{1, 2, 3}}), "");
static_assert(array<int, 3>{{1, 2, 4}} >= array<int, 3>{{1, 2, 3}}, "");
static_assert(!(array<int, 3>{{1, 2, 2}} >= array<int, 3>{{1, 2, 3}}), "");
} // namespace test_greaterequals

namespace test_tail {
static_assert(array<int, 2>{{3, 4}} == tail(array<int, 3>{{2, 3, 4}}), "");
static_assert(array<int, 0>{{}} == tail(array<int, 1>{{3}}), "");
} // namespace test_tail

namespace test_prepend {
static_assert(
    array<int, 3>{{2, 3, 4}} == prepend(2, array<int, 2>{{3, 4}}),
    "");
static_assert(array<int, 1>{{3}} == prepend(3, array<int, 0>{{}}), "");
} // namespace test_prepend

namespace test_to_std_array {
// NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
constexpr int obj2[3] = {3, 5, 6};
static_assert(array<int, 3>{{3, 5, 6}} == to_array(obj2), "");
static_assert(array<int, 3>{{3, 5, 6}} == to_array<int, 3>({3, 5, 6}), "");
} // namespace test_to_std_array

} // namespace
