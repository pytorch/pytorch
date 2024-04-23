#include <c10/util/TypeList.h>
#include <gtest/gtest.h>
#include <memory>

using namespace c10::guts::typelist;
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
    std::is_same<
        typelist<int, float&, const MyClass&&>,
        from_tuple_t<std::tuple<int, float&, const MyClass&&>>>::value,
    "");
static_assert(std::is_same<typelist<>, from_tuple_t<std::tuple<>>>::value, "");
} // namespace test_from_tuple

namespace test_to_tuple {
class MyClass {};
static_assert(
    std::is_same<
        std::tuple<int, float&, const MyClass&&>,
        to_tuple_t<typelist<int, float&, const MyClass&&>>>::value,
    "");
static_assert(std::is_same<std::tuple<>, to_tuple_t<typelist<>>>::value, "");
} // namespace test_to_tuple

namespace test_concat {
class MyClass {};
static_assert(std::is_same<typelist<>, concat_t<>>::value, "");
static_assert(std::is_same<typelist<>, concat_t<typelist<>>>::value, "");
static_assert(
    std::is_same<typelist<>, concat_t<typelist<>, typelist<>>>::value,
    "");
static_assert(std::is_same<typelist<int>, concat_t<typelist<int>>>::value, "");
static_assert(
    std::is_same<typelist<int>, concat_t<typelist<int>, typelist<>>>::value,
    "");
static_assert(
    std::is_same<typelist<int>, concat_t<typelist<>, typelist<int>>>::value,
    "");
static_assert(
    std::is_same<
        typelist<int>,
        concat_t<typelist<>, typelist<int>, typelist<>>>::value,
    "");
static_assert(
    std::is_same<
        typelist<int, float&>,
        concat_t<typelist<int>, typelist<float&>>>::value,
    "");
static_assert(
    std::is_same<
        typelist<int, float&>,
        concat_t<typelist<>, typelist<int, float&>, typelist<>>>::value,
    "");
static_assert(
    std::is_same<
        typelist<int, float&, const MyClass&&>,
        concat_t<
            typelist<>,
            typelist<int, float&>,
            typelist<const MyClass&&>>>::value,
    "");
} // namespace test_concat

namespace test_filter {
class MyClass {};
static_assert(
    std::is_same<typelist<>, filter_t<std::is_reference, typelist<>>>::value,
    "");
static_assert(
    std::is_same<
        typelist<>,
        filter_t<std::is_reference, typelist<int, float, double, MyClass>>>::
        value,
    "");
static_assert(
    std::is_same<
        typelist<float&, const MyClass&&>,
        filter_t<
            std::is_reference,
            typelist<int, float&, double, const MyClass&&>>>::value,
    "");
} // namespace test_filter

namespace test_count_if {
class MyClass final {};
static_assert(
    count_if<
        std::is_reference,
        typelist<int, bool&, const MyClass&&, float, double>>::value == 2,
    "");
static_assert(count_if<std::is_reference, typelist<int, bool>>::value == 0, "");
static_assert(count_if<std::is_reference, typelist<>>::value == 0, "");
} // namespace test_count_if

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

namespace test_map {
class MyClass {};
static_assert(
    std::is_same<typelist<>, map_t<std::add_lvalue_reference_t, typelist<>>>::
        value,
    "");
static_assert(
    std::is_same<
        typelist<int&>,
        map_t<std::add_lvalue_reference_t, typelist<int>>>::value,
    "");
static_assert(
    std::is_same<
        typelist<int&, double&, const MyClass&>,
        map_t<
            std::add_lvalue_reference_t,
            typelist<int, double, const MyClass>>>::value,
    "");
} // namespace test_map

namespace test_head {
class MyClass {};
static_assert(std::is_same<int, head_t<typelist<int, double>>>::value, "");
static_assert(
    std::is_same<const MyClass&, head_t<typelist<const MyClass&, double>>>::
        value,
    "");
static_assert(
    std::is_same<MyClass&&, head_t<typelist<MyClass&&, MyClass>>>::value,
    "");
static_assert(std::is_same<bool, head_t<typelist<bool>>>::value, "");
} // namespace test_head

namespace test_head_with_default {
class MyClass {};
static_assert(
    std::is_same<int, head_with_default_t<bool, typelist<int, double>>>::value,
    "");
static_assert(
    std::is_same<
        const MyClass&,
        head_with_default_t<bool, typelist<const MyClass&, double>>>::value,
    "");
static_assert(
    std::is_same<
        MyClass&&,
        head_with_default_t<bool, typelist<MyClass&&, MyClass>>>::value,
    "");
static_assert(
    std::is_same<int, head_with_default_t<bool, typelist<int>>>::value,
    "");
static_assert(
    std::is_same<bool, head_with_default_t<bool, typelist<>>>::value,
    "");
} // namespace test_head_with_default

namespace test_reverse {
class MyClass {};
static_assert(
    std::is_same<
        typelist<int, double, MyClass*, const MyClass&&>,
        reverse_t<typelist<const MyClass&&, MyClass*, double, int>>>::value,
    "");
static_assert(std::is_same<typelist<>, reverse_t<typelist<>>>::value, "");
} // namespace test_reverse

namespace test_map_types_to_values {
struct map_to_size {
  template <class T>
  constexpr size_t operator()(T) const {
    return sizeof(typename T::type);
  }
};

TEST(TypeListTest, MapTypesToValues_sametype) {
  auto sizes =
      map_types_to_values<typelist<int64_t, bool, uint32_t>>(map_to_size());
  std::tuple<size_t, size_t, size_t> expected(8, 1, 4);
  static_assert(std::is_same<decltype(expected), decltype(sizes)>::value, "");
  EXPECT_EQ(expected, sizes);
}

struct map_make_shared {
  template <class T>
  std::shared_ptr<typename T::type> operator()(T) {
    return std::make_shared<typename T::type>();
  }
};

TEST(TypeListTest, MapTypesToValues_differenttypes) {
  auto shared_ptrs =
      map_types_to_values<typelist<int, double>>(map_make_shared());
  static_assert(
      std::is_same<
          std::tuple<std::shared_ptr<int>, std::shared_ptr<double>>,
          decltype(shared_ptrs)>::value,
      "");
}

struct Class1 {
  static int func() {
    return 3;
  }
};
struct Class2 {
  static double func() {
    return 2.0;
  }
};

struct mapper_call_func {
  template <class T>
  decltype(auto) operator()(T) {
    return T::type::func();
  }
};

TEST(TypeListTest, MapTypesToValues_members) {
  auto result =
      map_types_to_values<typelist<Class1, Class2>>(mapper_call_func());
  std::tuple<int, double> expected(3, 2.0);
  static_assert(std::is_same<decltype(expected), decltype(result)>::value, "");
  EXPECT_EQ(expected, result);
}

struct mapper_call_nonexistent_function {
  template <class T>
  decltype(auto) operator()(T) {
    return T::type::this_doesnt_exist();
  }
};

TEST(TypeListTest, MapTypesToValues_empty) {
  auto result =
      map_types_to_values<typelist<>>(mapper_call_nonexistent_function());
  std::tuple<> expected;
  static_assert(std::is_same<decltype(expected), decltype(result)>::value, "");
  EXPECT_EQ(expected, result);
}
} // namespace test_map_types_to_values

namespace test_find_if {
static_assert(0 == find_if<typelist<char&>, std::is_reference>::value, "");
static_assert(
    0 == find_if<typelist<char&, int, char&, int&>, std::is_reference>::value,
    "");
static_assert(
    2 == find_if<typelist<char, int, char&, int&>, std::is_reference>::value,
    "");
static_assert(
    3 == find_if<typelist<char, int, char, int&>, std::is_reference>::value,
    "");
} // namespace test_find_if

namespace test_contains {
static_assert(contains<typelist<double>, double>::value, "");
static_assert(contains<typelist<int, double>, double>::value, "");
static_assert(!contains<typelist<int, double>, float>::value, "");
static_assert(!contains<typelist<>, double>::value, "");
} // namespace test_contains

namespace test_take {
static_assert(std::is_same<typelist<>, take_t<typelist<>, 0>>::value, "");
static_assert(
    std::is_same<typelist<>, take_t<typelist<int64_t>, 0>>::value,
    "");
static_assert(
    std::is_same<typelist<int64_t>, take_t<typelist<int64_t>, 1>>::value,
    "");
static_assert(
    std::is_same<typelist<>, take_t<typelist<int64_t, int32_t>, 0>>::value,
    "");
static_assert(
    std::is_same<typelist<int64_t>, take_t<typelist<int64_t, int32_t>, 1>>::
        value,
    "");
static_assert(
    std::is_same<
        typelist<int64_t, int32_t>,
        take_t<typelist<int64_t, int32_t>, 2>>::value,
    "");
} // namespace test_take

namespace test_drop {
static_assert(std::is_same<typelist<>, drop_t<typelist<>, 0>>::value, "");
static_assert(
    std::is_same<typelist<int64_t>, drop_t<typelist<int64_t>, 0>>::value,
    "");
static_assert(
    std::is_same<typelist<>, drop_t<typelist<int64_t>, 1>>::value,
    "");
static_assert(
    std::is_same<
        typelist<int64_t, int32_t>,
        drop_t<typelist<int64_t, int32_t>, 0>>::value,
    "");
static_assert(
    std::is_same<typelist<int32_t>, drop_t<typelist<int64_t, int32_t>, 1>>::
        value,
    "");
static_assert(
    std::is_same<typelist<>, drop_t<typelist<int64_t, int32_t>, 2>>::value,
    "");
} // namespace test_drop

namespace test_drop_if_nonempty {
static_assert(
    std::is_same<typelist<>, drop_if_nonempty_t<typelist<>, 0>>::value,
    "");
static_assert(
    std::is_same<typelist<int64_t>, drop_if_nonempty_t<typelist<int64_t>, 0>>::
        value,
    "");
static_assert(
    std::is_same<typelist<>, drop_if_nonempty_t<typelist<int64_t>, 1>>::value,
    "");
static_assert(
    std::is_same<
        typelist<int64_t, int32_t>,
        drop_if_nonempty_t<typelist<int64_t, int32_t>, 0>>::value,
    "");
static_assert(
    std::is_same<
        typelist<int32_t>,
        drop_if_nonempty_t<typelist<int64_t, int32_t>, 1>>::value,
    "");
static_assert(
    std::is_same<
        typelist<>,
        drop_if_nonempty_t<typelist<int64_t, int32_t>, 2>>::value,
    "");
static_assert(
    std::is_same<typelist<>, drop_if_nonempty_t<typelist<>, 1>>::value,
    "");
static_assert(
    std::is_same<
        typelist<>,
        drop_if_nonempty_t<typelist<int64_t, int32_t>, 3>>::value,
    "");
} // namespace test_drop_if_nonempty
// NOLINTEND(modernize-unary-static-assert)
