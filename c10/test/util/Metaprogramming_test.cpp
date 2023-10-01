#include <c10/test/util/Macros.h>
#include <c10/util/Metaprogramming.h>
#include <gtest/gtest.h>
#include <cstdlib>

using namespace c10::guts;

// NOLINTBEGIN(modernize*)
namespace {

namespace test_function_traits {
static_assert(
    std::is_same<
        void,
        typename function_traits<void(int, float)>::return_type>::value,
    "");
static_assert(
    std::is_same<int, typename function_traits<int(int, float)>::return_type>::
        value,
    "");
static_assert(
    std::is_same<
        typelist::typelist<int, float>,
        typename function_traits<void(int, float)>::parameter_types>::value,
    "");
static_assert(
    std::is_same<
        typelist::typelist<int, float>,
        typename function_traits<int(int, float)>::parameter_types>::value,
    "");

static_assert(
    std::is_same<
        bool,
        typename make_function_traits_t<bool, typelist::typelist<int, float>>::
            return_type>::value,
    "");
static_assert(
    std::is_same<
        void,
        typename make_function_traits_t<void, typelist::typelist<int, float>>::
            return_type>::value,
    "");
static_assert(
    std::is_same<
        typelist::typelist<int, float>,
        typename make_function_traits_t<bool, typelist::typelist<int, float>>::
            parameter_types>::value,
    "");
static_assert(
    std::is_same<
        typelist::typelist<int, float>,
        typename make_function_traits_t<void, typelist::typelist<int, float>>::
            parameter_types>::value,
    "");
static_assert(
    std::is_same<
        bool(int, float),
        typename make_function_traits_t<bool, typelist::typelist<int, float>>::
            func_type>::value,
    "");
static_assert(
    std::is_same<
        void(int, float),
        typename make_function_traits_t<void, typelist::typelist<int, float>>::
            func_type>::value,
    "");
} // namespace test_function_traits

struct MovableOnly {
  constexpr MovableOnly(int val_) : val(val_) { /* no default constructor */
  }
  MovableOnly(const MovableOnly&) = delete;
  MovableOnly(MovableOnly&&) = default;
  MovableOnly& operator=(const MovableOnly&) = delete;
  MovableOnly& operator=(MovableOnly&&) = default;

  friend bool operator==(const MovableOnly& lhs, const MovableOnly& rhs) {
    return lhs.val == rhs.val;
  }

 private:
  int val;
};

template <class T>
using is_my_movable_only_class =
    std::is_same<MovableOnly, std::remove_cv_t<std::remove_reference_t<T>>>;

struct CopyCounting {
  int move_count;
  int copy_count;

  CopyCounting() : move_count(0), copy_count(0) {}
  CopyCounting(const CopyCounting& rhs)
      : move_count(rhs.move_count), copy_count(rhs.copy_count + 1) {}
  CopyCounting(CopyCounting&& rhs)
      : move_count(rhs.move_count + 1), copy_count(rhs.copy_count) {}
  CopyCounting& operator=(const CopyCounting& rhs) {
    move_count = rhs.move_count;
    copy_count = rhs.copy_count + 1;
    return *this;
  }
  CopyCounting& operator=(CopyCounting&& rhs) {
    move_count = rhs.move_count + 1;
    copy_count = rhs.copy_count;
    return *this;
  }
};

template <class T>
using is_my_copy_counting_class =
    std::is_same<CopyCounting, std::remove_cv_t<std::remove_reference_t<T>>>;

namespace test_tuple_elements {
// note: not testing empty selection, as some compilers will raise
// "parameter set but not used" in tuple_elements(). a good example
// of the friction that comes with using these tools

TEST(MetaprogrammingTest, TupleElements_subsetSelection) {
  auto x = std::make_tuple(0, "HEY", 2.0);
  auto y = tuple_elements(x, std::index_sequence<0, 2>());
  auto z = std::make_tuple(0, 2.0);
  EXPECT_EQ(y, z);
}

TEST(MetaprogrammingTest, TupleElements_reorderSelection) {
  auto x = std::make_tuple(0, "HEY", 2.0);
  auto y = tuple_elements(x, std::index_sequence<0, 2, 1>());
  auto z = std::make_tuple(0, 2.0, "HEY");
  EXPECT_EQ(y, z);
}
} // namespace test_tuple_elements

namespace test_tuple_take {
// note: not testing empty prefix, see note on empty selection above.

TEST(MetaprogrammingTest, TupleTake_nonemptyPrefix) {
  auto x = std::make_tuple(0, "HEY", 2.0);
  auto y = tuple_take<decltype(x), 2>(x);
  auto z = std::make_tuple(0, "HEY");
  EXPECT_EQ(y, z);
}

TEST(MetaprogrammingTest, TupleTake_fullPrefix) {
  auto x = std::make_tuple(0, "HEY", 2.0);
  auto y = tuple_take<decltype(x), 3>(x);
  EXPECT_EQ(x, y);
}

TEST(MetaprogrammingTest, TupleTake_negative) {
  auto x = std::make_tuple(0, "HEY", 2.0);
  auto y = tuple_take<decltype(x), -2>(x);
  auto z = std::make_tuple("HEY", 2.0);
  EXPECT_EQ(y, z);
}
} // namespace test_tuple_take

namespace test_tuple_slice {
TEST(MetaprogrammingTest, TupleSlice_middle) {
  auto x = std::make_tuple(0, "HEY", 2.0, false);
  auto y = tuple_slice<decltype(x), 1, 2>(x);
  auto z = std::make_tuple("HEY", 2.0);
  EXPECT_EQ(y, z);
}

TEST(MetaprogrammingTest, TupleSlice_full) {
  auto x = std::make_tuple(0, "HEY", 2.0);
  auto y = tuple_slice<decltype(x), 0, 3>(x);
  EXPECT_EQ(x, y);
}
} // namespace test_tuple_slice

namespace test_tuple_map {
TEST(MetaprogrammingTest, TupleMap_simple) {
  auto result = tuple_map(
      std::tuple<int32_t, int32_t, int32_t>(3, 4, 5),
      [](int32_t a) -> int16_t { return a + 1; });
  static_assert(
      std::is_same<std::tuple<int16_t, int16_t, int16_t>, decltype(result)>::
          value,
      "");
  EXPECT_EQ(4, std::get<0>(result));
  EXPECT_EQ(5, std::get<1>(result));
  EXPECT_EQ(6, std::get<2>(result));
}

TEST(MetaprogrammingTest, TupleMap_mapperTakesDifferentButConvertibleType) {
  auto result = tuple_map(
      std::tuple<int32_t, int32_t, int32_t>(3, 4, 5),
      [](int64_t a) -> int16_t { return a + 1; });
  static_assert(
      std::is_same<std::tuple<int16_t, int16_t, int16_t>, decltype(result)>::
          value,
      "");
  EXPECT_EQ(4, std::get<0>(result));
  EXPECT_EQ(5, std::get<1>(result));
  EXPECT_EQ(6, std::get<2>(result));
}

TEST(MetaprogrammingTest, TupleMap_mapperTakesConstRef) {
  auto result = tuple_map(
      std::tuple<int32_t, int32_t, int32_t>(3, 4, 5),
      [](const int32_t& a) -> int16_t { return a + 1; });
  static_assert(
      std::is_same<std::tuple<int16_t, int16_t, int16_t>, decltype(result)>::
          value,
      "");
  EXPECT_EQ(4, std::get<0>(result));
  EXPECT_EQ(5, std::get<1>(result));
  EXPECT_EQ(6, std::get<2>(result));
}

TEST(MetaprogrammingTest, TupleMap_mapsToDifferentTypes) {
  struct Mapper {
    std::string operator()(int32_t a) const {
      return std::to_string(a);
    }
    int32_t operator()(const std::string& a) const {
      return atoi(a.c_str());
    }
  };
  auto result = tuple_map(std::tuple<int32_t, std::string>(3, "4"), Mapper());
  static_assert(
      std::is_same<std::tuple<std::string, int32_t>, decltype(result)>::value,
      "");
  EXPECT_EQ("3", std::get<0>(result));
  EXPECT_EQ(4, std::get<1>(result));
}

TEST(MetaprogrammingTest, TupleMap_differentiatesLRValueReferences) {
  struct Mapper {
    std::string operator()(std::string&& a) const {
      return "moved";
    }
    std::string operator()(const std::string& a) const {
      return "copied";
    }
  };
  std::string str1, str2;
  auto result = tuple_map(
      std::tuple<const std::string&, std::string&&>(str1, std::move(str2)),
      Mapper());
  static_assert(
      std::is_same<std::tuple<std::string, std::string>, decltype(result)>::
          value,
      "");
  EXPECT_EQ("copied", std::get<0>(result));
  EXPECT_EQ("moved", std::get<1>(result));
}

TEST(MetaprogrammingTest, TupleMap_canWorkWithMovableOnlyType) {
  auto result = tuple_map(
      std::tuple<MovableOnly>(MovableOnly(7)), [](MovableOnly a) { return a; });
  static_assert(
      std::is_same<std::tuple<MovableOnly>, decltype(result)>::value, "");
  EXPECT_EQ(MovableOnly(7), std::get<0>(result));
}

TEST(MetaprogrammingTest, TupleMap_doesntUnecessarilyCopyValues) {
  auto result = tuple_map(
      std::tuple<CopyCounting>(CopyCounting()),
      [](CopyCounting a) { return a; });
  static_assert(
      std::is_same<std::tuple<CopyCounting>, decltype(result)>::value, "");
  EXPECT_EQ(4, std::get<0>(result).move_count);
  EXPECT_EQ(0, std::get<0>(result).copy_count);
}

TEST(MetaprogrammingTest, TupleMap_doesntUnecessarilyMoveValues) {
  CopyCounting a;
  auto result = tuple_map(
      std::tuple<CopyCounting&&>(std::move(a)),
      [](CopyCounting&& a) -> CopyCounting&& { return std::move(a); });
  static_assert(
      std::is_same<std::tuple<CopyCounting&&>, decltype(result)>::value, "");
  EXPECT_EQ(&a, &std::get<0>(result));
  EXPECT_EQ(0, std::get<0>(result).move_count);
  EXPECT_EQ(0, std::get<0>(result).copy_count);
}

TEST(MetaprogrammingTest, TupleMap_canBeUsedWithAutoLambdas) {
  struct A final {
    int32_t func() {
      return 5;
    }
  };
  struct B final {
    std::string func() {
      return "5";
    }
  };
  auto result =
      tuple_map(std::make_tuple(A(), B()), [](auto a) { return a.func(); });
  static_assert(
      std::is_same<std::tuple<int32_t, std::string>, decltype(result)>::value,
      "");
  EXPECT_EQ(5, std::get<0>(result));
  EXPECT_EQ("5", std::get<1>(result));
}
} // namespace test_tuple_map

} // namespace
// NOLINTEND(modernize*)
