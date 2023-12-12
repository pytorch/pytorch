#include <gtest/gtest.h>

#include <ATen/core/SingletonSymNodeImpl.h>
#include <c10/core/SymInt.h>
#include <c10/core/SymNodeImpl.h>
#include <torch/torch.h>

#include <test/cpp/api/support.h>
#include "c10/util/intrusive_ptr.h"

TEST(SingletonIntTest, Comparisons) {
  auto x = torch::randn({2, 2});

  auto a = c10::SymInt(c10::SymNode(
      c10::make_intrusive<c10::SingletonSymNodeImpl>(1, 1, x, x, 1)));
  auto b = c10::SymInt(c10::SymNode(
      c10::make_intrusive<c10::SingletonSymNodeImpl>(1, 1, x, x, 1)));
  auto c = c10::SymInt(c10::SymNode(
      c10::make_intrusive<c10::SingletonSymNodeImpl>(2, 1, x, x, 1)));
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

TEST(SingletonIntTest, WiithFactor) {
  auto x = torch::randn({2, 2});

  auto a = c10::SymInt(c10::SymNode(
      c10::make_intrusive<c10::SingletonSymNodeImpl>(1, 5, x, x, 1)));
  auto b = c10::SymInt(c10::SymNode(
      c10::make_intrusive<c10::SingletonSymNodeImpl>(1, 10, x, x, 1)));
  // eq
  ASSERT_FALSE(a == b);
  ASSERT_FALSE(a >= b);
  ASSERT_TRUE(b >= a);
  ASSERT_TRUE(a <= b);
  ASSERT_FALSE(b <= a);
  // ne
  ASSERT_TRUE(a != b);
  // mul
  ASSERT_TRUE(a * 2 == b);
  ASSERT_TRUE(a * 3 >= b);
  ASSERT_TRUE(a * 2 == 2 * a);
}

TEST(SingletonIntTest, AccessToTensor) {
  auto x = torch::randn({2, 2});
  auto y = torch::randn({2, 2});

  auto a = c10::SymInt(c10::SymNode(
      c10::make_intrusive<c10::SingletonSymNodeImpl>(1, 5, x, y, 1)));
  auto b = c10::SymInt(c10::SymNode(
      c10::make_intrusive<c10::SingletonSymNodeImpl>(1, 10, x, y, 1)));

  auto p = dynamic_cast<c10::SingletonSymNodeImpl*>(a.toSymNode().get());
  ASSERT_TRUE(p != nullptr);

  ASSERT_TRUE(torch::allclose(p->singleton_data(), x));
}
