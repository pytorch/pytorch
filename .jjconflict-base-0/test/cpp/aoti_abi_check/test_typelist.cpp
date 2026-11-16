#include <torch/headeronly/util/TypeList.h>

using namespace torch::headeronly::guts::typelist;
// NOLINTBEGIN(modernize-unary-static-assert)
namespace test_size {
class MyClass {};
static_assert(0 == size<typelist<>>::value, "");
static_assert(1 == size<typelist<int>>::value, "");
static_assert(3 == size<typelist<int, float&, const MyClass&&>>::value, "");
} // namespace test_size

namespace test_from_tuple {
class MyClass {};
static_assert(
    std::is_same_v<
        typelist<int, float&, const MyClass&&>,
        from_tuple_t<std::tuple<int, float&, const MyClass&&>>>,
    "");
static_assert(std::is_same_v<typelist<>, from_tuple_t<std::tuple<>>>, "");
} // namespace test_from_tuple

namespace test_to_tuple {
class MyClass {};
static_assert(
    std::is_same_v<
        std::tuple<int, float&, const MyClass&&>,
        to_tuple_t<typelist<int, float&, const MyClass&&>>>,
    "");
static_assert(std::is_same_v<std::tuple<>, to_tuple_t<typelist<>>>, "");
} // namespace test_to_tuple

namespace test_concat {
class MyClass {};
static_assert(std::is_same_v<typelist<>, concat_t<>>, "");
static_assert(std::is_same_v<typelist<>, concat_t<typelist<>>>, "");
static_assert(std::is_same_v<typelist<>, concat_t<typelist<>, typelist<>>>, "");
static_assert(std::is_same_v<typelist<int>, concat_t<typelist<int>>>, "");
static_assert(
    std::is_same_v<typelist<int>, concat_t<typelist<int>, typelist<>>>,
    "");
static_assert(
    std::is_same_v<typelist<int>, concat_t<typelist<>, typelist<int>>>,
    "");
static_assert(
    std::is_same_v<
        typelist<int>,
        concat_t<typelist<>, typelist<int>, typelist<>>>,
    "");
static_assert(
    std::is_same_v<
        typelist<int, float&>,
        concat_t<typelist<int>, typelist<float&>>>,
    "");
static_assert(
    std::is_same_v<
        typelist<int, float&>,
        concat_t<typelist<>, typelist<int, float&>, typelist<>>>,
    "");
static_assert(
    std::is_same_v<
        typelist<int, float&, const MyClass&&>,
        concat_t<typelist<>, typelist<int, float&>, typelist<const MyClass&&>>>,
    "");
} // namespace test_concat

namespace test_true_for_each_type {
template <class>
class Test;
class MyClass {};
static_assert(
    all<std::is_reference,
        typelist<int&, const float&&, const MyClass&>>::value,
    "");
static_assert(
    !all<std::is_reference, typelist<int&, const float, const MyClass&>>::value,
    "");
static_assert(all<std::is_reference, typelist<>>::value, "");
} // namespace test_true_for_each_type

namespace test_true_for_any_type {
template <class>
class Test;
class MyClass {};
static_assert(
    true_for_any_type<
        std::is_reference,
        typelist<int&, const float&&, const MyClass&>>::value,
    "");
static_assert(
    true_for_any_type<
        std::is_reference,
        typelist<int&, const float, const MyClass&>>::value,
    "");
static_assert(
    !true_for_any_type<
        std::is_reference,
        typelist<int, const float, const MyClass>>::value,
    "");
static_assert(!true_for_any_type<std::is_reference, typelist<>>::value, "");
} // namespace test_true_for_any_type

namespace test_head {
class MyClass {};
static_assert(std::is_same_v<int, head_t<typelist<int, double>>>, "");
static_assert(
    std::is_same_v<const MyClass&, head_t<typelist<const MyClass&, double>>>,
    "");
static_assert(
    std::is_same_v<MyClass&&, head_t<typelist<MyClass&&, MyClass>>>,
    "");
static_assert(std::is_same_v<bool, head_t<typelist<bool>>>, "");
} // namespace test_head

namespace test_head_with_default {
class MyClass {};
static_assert(
    std::is_same_v<int, head_with_default_t<bool, typelist<int, double>>>,
    "");
static_assert(
    std::is_same_v<
        const MyClass&,
        head_with_default_t<bool, typelist<const MyClass&, double>>>,
    "");
static_assert(
    std::is_same_v<
        MyClass&&,
        head_with_default_t<bool, typelist<MyClass&&, MyClass>>>,
    "");
static_assert(
    std::is_same_v<int, head_with_default_t<bool, typelist<int>>>,
    "");
static_assert(std::is_same_v<bool, head_with_default_t<bool, typelist<>>>, "");
} // namespace test_head_with_default

namespace test_contains {
static_assert(contains<typelist<double>, double>::value, "");
static_assert(contains<typelist<int, double>, double>::value, "");
static_assert(!contains<typelist<int, double>, float>::value, "");
static_assert(!contains<typelist<>, double>::value, "");
} // namespace test_contains

namespace test_take {
static_assert(std::is_same_v<typelist<>, take_t<typelist<>, 0>>, "");
static_assert(std::is_same_v<typelist<>, take_t<typelist<int64_t>, 0>>, "");
static_assert(
    std::is_same_v<typelist<int64_t>, take_t<typelist<int64_t>, 1>>,
    "");
static_assert(
    std::is_same_v<typelist<>, take_t<typelist<int64_t, int32_t>, 0>>,
    "");
static_assert(
    std::is_same_v<typelist<int64_t>, take_t<typelist<int64_t, int32_t>, 1>>,
    "");
static_assert(
    std::is_same_v<
        typelist<int64_t, int32_t>,
        take_t<typelist<int64_t, int32_t>, 2>>,
    "");
} // namespace test_take

namespace test_drop {
static_assert(std::is_same_v<typelist<>, drop_t<typelist<>, 0>>, "");
static_assert(
    std::is_same_v<typelist<int64_t>, drop_t<typelist<int64_t>, 0>>,
    "");
static_assert(std::is_same_v<typelist<>, drop_t<typelist<int64_t>, 1>>, "");
static_assert(
    std::is_same_v<
        typelist<int64_t, int32_t>,
        drop_t<typelist<int64_t, int32_t>, 0>>,
    "");
static_assert(
    std::is_same_v<typelist<int32_t>, drop_t<typelist<int64_t, int32_t>, 1>>,
    "");
static_assert(
    std::is_same_v<typelist<>, drop_t<typelist<int64_t, int32_t>, 2>>,
    "");
} // namespace test_drop

namespace test_drop_if_nonempty {
static_assert(
    std::is_same_v<typelist<>, drop_if_nonempty_t<typelist<>, 0>>,
    "");
static_assert(
    std::is_same_v<typelist<int64_t>, drop_if_nonempty_t<typelist<int64_t>, 0>>,
    "");
static_assert(
    std::is_same_v<typelist<>, drop_if_nonempty_t<typelist<int64_t>, 1>>,
    "");
static_assert(
    std::is_same_v<
        typelist<int64_t, int32_t>,
        drop_if_nonempty_t<typelist<int64_t, int32_t>, 0>>,
    "");
static_assert(
    std::is_same_v<
        typelist<int32_t>,
        drop_if_nonempty_t<typelist<int64_t, int32_t>, 1>>,
    "");
static_assert(
    std::is_same_v<
        typelist<>,
        drop_if_nonempty_t<typelist<int64_t, int32_t>, 2>>,
    "");
static_assert(
    std::is_same_v<typelist<>, drop_if_nonempty_t<typelist<>, 1>>,
    "");
static_assert(
    std::is_same_v<
        typelist<>,
        drop_if_nonempty_t<typelist<int64_t, int32_t>, 3>>,
    "");
} // namespace test_drop_if_nonempty
// NOLINTEND(modernize-unary-static-assert)
