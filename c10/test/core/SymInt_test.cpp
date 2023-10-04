#include <gtest/gtest.h>

#include <c10/core/SingletonSymNodeImpl.h>
#include <c10/core/SymInt.h>
#include <c10/core/SymNodeImpl.h>

using namespace c10;
#ifndef C10_MOBILE
static void check(int64_t value) {
  const auto i = SymInt(value);
  EXPECT_EQ(i.maybe_as_int(), c10::make_optional(value));
}

TEST(SymIntTest, ConcreteInts) {
  check(INT64_MAX);
  check(0);
  check(-1);
  check(-4611686018427387904LL);
  check(INT64_MIN);
}

TEST(SymIntTest, CheckRange) {
  EXPECT_FALSE(SymInt::check_range(INT64_MIN));
}

TEST(SymIntTest, SingletonSymNode) {
  auto a = c10::SymInt(
      c10::SymNode(c10::make_intrusive<c10::SingletonSymNodeImpl>(1)));
  auto b = c10::SymInt(
      c10::SymNode(c10::make_intrusive<c10::SingletonSymNodeImpl>(1)));
  auto c = c10::SymInt(
      c10::SymNode(c10::make_intrusive<c10::SingletonSymNodeImpl>(2)));
  auto d = c10::SymInt(3);

  ASSERT_TRUE(a == a);
  ASSERT_TRUE(a == b);
  ASSERT_FALSE(a != a);
  ASSERT_FALSE(a != b);
  ASSERT_FALSE(a == c);
  ASSERT_TRUE(a != c);

  ASSERT_FALSE(a == d);
  ASSERT_TRUE(a != d);
  ASSERT_FALSE(d == a);
  ASSERT_TRUE(d != a);

  // ge
  ASSERT_TRUE(a >= a);
  ASSERT_TRUE(a >= b);
  ASSERT_TRUE(b >= a);
  // NOLINTNEXTLINE(hicpp-avoid-goto,cppcoreguidelines-avoid-goto)
  EXPECT_THROW((void)(a >= c), c10::Error);
  // NOLINTNEXTLINE(hicpp-avoid-goto,cppcoreguidelines-avoid-goto)
  EXPECT_THROW((void)(c >= a), c10::Error);
  // NOLINTNEXTLINE(hicpp-avoid-goto,cppcoreguidelines-avoid-goto)
  EXPECT_THROW((void)(c >= 3), c10::Error);
  ASSERT_TRUE(c >= 2);
  ASSERT_TRUE(c >= 1);
  ASSERT_FALSE(1 >= c);

  // lt
  ASSERT_FALSE(a < a);
  ASSERT_FALSE(a < b);
  ASSERT_FALSE(b < a);
  // NOLINTNEXTLINE(hicpp-avoid-goto,cppcoreguidelines-avoid-goto)
  EXPECT_THROW((void)(a < c), c10::Error);
  // NOLINTNEXTLINE(hicpp-avoid-goto,cppcoreguidelines-avoid-goto)
  EXPECT_THROW((void)(c < a), c10::Error);
  // NOLINTNEXTLINE(hicpp-avoid-goto,cppcoreguidelines-avoid-goto)
  EXPECT_THROW((void)(3 < a), c10::Error);
  // NOLINTNEXTLINE(hicpp-avoid-goto,cppcoreguidelines-avoid-goto)
  EXPECT_THROW((void)(2 < a), c10::Error);
  ASSERT_TRUE(1 < a);

  // le
  ASSERT_TRUE(a <= a);
  ASSERT_TRUE(b <= a);
  ASSERT_TRUE(a <= b);
  // NOLINTNEXTLINE(hicpp-avoid-goto,cppcoreguidelines-avoid-goto)
  EXPECT_THROW((void)(a <= c), c10::Error);
  // NOLINTNEXTLINE(hicpp-avoid-goto,cppcoreguidelines-avoid-goto)
  EXPECT_THROW((void)(c <= a), c10::Error);
  // NOLINTNEXTLINE(hicpp-avoid-goto,cppcoreguidelines-avoid-goto)
  EXPECT_THROW((void)(3 <= c), c10::Error);
  ASSERT_TRUE(2 <= c);
  ASSERT_TRUE(1 <= c);
  ASSERT_FALSE(c <= 1);

  // gt
  ASSERT_FALSE(a > a);
  ASSERT_FALSE(b > a);
  ASSERT_FALSE(a > b);
  // NOLINTNEXTLINE(hicpp-avoid-goto,cppcoreguidelines-avoid-goto)
  EXPECT_THROW((void)(a > c), c10::Error);
  // NOLINTNEXTLINE(hicpp-avoid-goto,cppcoreguidelines-avoid-goto)
  EXPECT_THROW((void)(c > a), c10::Error);
  // NOLINTNEXTLINE(hicpp-avoid-goto,cppcoreguidelines-avoid-goto)
  EXPECT_THROW((void)(a > 3), c10::Error);
  // NOLINTNEXTLINE(hicpp-avoid-goto,cppcoreguidelines-avoid-goto)
  EXPECT_THROW((void)(a > 2), c10::Error);
  ASSERT_TRUE(a > 1);
}
#endif
