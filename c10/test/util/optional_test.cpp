#include <c10/util/Optional.h>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <string>

#include <c10/util/ArrayRef.h>

namespace {

using testing::Eq;
using testing::Ge;
using testing::Gt;
using testing::Le;
using testing::Lt;
using testing::Ne;
using testing::Not;

template <typename T>
class OptionalTest : public ::testing::Test {
 public:
  using optional = c10::optional<T>;
};

template <typename T>
T getSampleValue();

template <>
bool getSampleValue() {
  return true;
}

template <>
uint64_t getSampleValue() {
  return 42;
}

template <>
c10::IntArrayRef getSampleValue() {
  return {};
}

template <>
std::string getSampleValue() {
  return "hello";
}

using OptionalTypes = ::testing::Types<
    // 32-bit scalar optimization.
    bool,
    // Trivially destructible but not 32-bit scalar.
    uint64_t,
    // ArrayRef optimization.
    c10::IntArrayRef,
    // Non-trivial destructor.
    std::string>;

TYPED_TEST_SUITE(OptionalTest, OptionalTypes);

TYPED_TEST(OptionalTest, Empty) {
  typename TestFixture::optional empty;

  EXPECT_FALSE((bool)empty);
  EXPECT_FALSE(empty.has_value());

  // NOLINTNEXTLINE(bugprone-unchecked-optional-access,hicpp-avoid-goto,cppcoreguidelines-avoid-goto)
  EXPECT_THROW(empty.value(), c10::bad_optional_access);
}

TYPED_TEST(OptionalTest, Initialized) {
  using optional = typename TestFixture::optional;

  const auto val = getSampleValue<TypeParam>();
  optional opt((val));
  auto copy(opt), moveFrom1(opt), moveFrom2(opt);
  optional move(std::move(moveFrom1));
  optional copyAssign;
  copyAssign = opt;
  optional moveAssign;
  moveAssign = std::move(moveFrom2);

  std::array<typename TestFixture::optional*, 5> opts = {
      &opt, &copy, &copyAssign, &move, &moveAssign};
  for (auto* popt : opts) {
    auto& opt = *popt;
    EXPECT_TRUE((bool)opt);
    EXPECT_TRUE(opt.has_value());

    // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
    EXPECT_EQ(opt.value(), val);
    // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
    EXPECT_EQ(*opt, val);
  }
}

class SelfCompareTest : public testing::TestWithParam<c10::optional<int>> {};

TEST_P(SelfCompareTest, SelfCompare) {
  c10::optional<int> x = GetParam();
  EXPECT_THAT(x, Eq(x));
  EXPECT_THAT(x, Le(x));
  EXPECT_THAT(x, Ge(x));
  EXPECT_THAT(x, Not(Ne(x)));
  EXPECT_THAT(x, Not(Lt(x)));
  EXPECT_THAT(x, Not(Gt(x)));
}

INSTANTIATE_TEST_SUITE_P(
    nullopt,
    SelfCompareTest,
    testing::Values(c10::nullopt));
INSTANTIATE_TEST_SUITE_P(
    int,
    SelfCompareTest,
    testing::Values(c10::make_optional(2)));

TEST(OptionalTest, Nullopt) {
  c10::optional<int> x = 2;

  EXPECT_THAT(c10::nullopt, Not(Eq(x)));
  EXPECT_THAT(x, Not(Eq(c10::nullopt)));

  EXPECT_THAT(x, Ne(c10::nullopt));
  EXPECT_THAT(c10::nullopt, Ne(x));

  EXPECT_THAT(x, Not(Lt(c10::nullopt)));
  EXPECT_THAT(c10::nullopt, Lt(x));

  EXPECT_THAT(x, Not(Le(c10::nullopt)));
  EXPECT_THAT(c10::nullopt, Le(x));

  EXPECT_THAT(x, Gt(c10::nullopt));
  EXPECT_THAT(c10::nullopt, Not(Gt(x)));

  EXPECT_THAT(x, Ge(c10::nullopt));
  EXPECT_THAT(c10::nullopt, Not(Ge(x)));
}

// Ensure comparisons work...
using CmpTestTypes = testing::Types<
    // between two optionals
    std::pair<c10::optional<int>, c10::optional<int>>,

    // between an optional and a value
    std::pair<c10::optional<int>, int>,
    // between a value and an optional
    std::pair<int, c10::optional<int>>,

    // between an optional and a differently typed value
    std::pair<c10::optional<int>, long>,
    // between a differently typed value and an optional
    std::pair<long, c10::optional<int>>>;
template <typename T>
class CmpTest : public testing::Test {};
TYPED_TEST_SUITE(CmpTest, CmpTestTypes);

TYPED_TEST(CmpTest, Cmp) {
  TypeParam pair = {2, 3};
  auto x = pair.first;
  auto y = pair.second;

  EXPECT_THAT(x, Not(Eq(y)));

  EXPECT_THAT(x, Ne(y));

  EXPECT_THAT(x, Lt(y));
  EXPECT_THAT(y, Not(Lt(x)));

  EXPECT_THAT(x, Le(y));
  EXPECT_THAT(y, Not(Le(x)));

  EXPECT_THAT(x, Not(Gt(y)));
  EXPECT_THAT(y, Gt(x));

  EXPECT_THAT(x, Not(Ge(y)));
  EXPECT_THAT(y, Ge(x));
}

} // namespace
