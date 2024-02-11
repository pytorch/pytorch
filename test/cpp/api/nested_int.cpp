#include <gtest/gtest.h>

#include <ATen/core/NestedIntSymNodeImpl.h>
#include <c10/core/SymInt.h>
#include <c10/core/SymNodeImpl.h>
#include <torch/torch.h>

#include <test/cpp/api/support.h>

TEST(NestedIntTest, Comparisons) {
  auto x = torch::randn({2, 2});

  // WARNING: Make sure these SymInts to not make their way into the dispatcher.
  // The naming is kind of unfortunate, but we're using PYTHON variant NTs in
  // C++ because only python variant NTs support comparisons.
  auto a =
      c10::SymInt(c10::SymNode(c10::make_intrusive<c10::NestedIntSymNodeImpl>(
          1, 1, x, c10::NestedTensorVariant::PYTHON)));
  auto b =
      c10::SymInt(c10::SymNode(c10::make_intrusive<c10::NestedIntSymNodeImpl>(
          1, 1, x, c10::NestedTensorVariant::PYTHON)));
  auto c =
      c10::SymInt(c10::SymNode(c10::make_intrusive<c10::NestedIntSymNodeImpl>(
          2, 1, x, c10::NestedTensorVariant::PYTHON)));
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

TEST(NestedIntTest, WithFactor) {
  auto x = torch::randn({2, 2});

  auto a =
      c10::SymInt(c10::SymNode(c10::make_intrusive<c10::NestedIntSymNodeImpl>(
          1, 5, x, c10::NestedTensorVariant::PYTHON)));
  auto b =
      c10::SymInt(c10::SymNode(c10::make_intrusive<c10::NestedIntSymNodeImpl>(
          1, 10, x, c10::NestedTensorVariant::PYTHON)));
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

TEST(NestedIntTest, CppNestedIntErrorsOnComparison) {
  auto x = torch::randn({2, 2});

  auto c =
      c10::SymInt(c10::SymNode(c10::make_intrusive<c10::NestedIntSymNodeImpl>(
          -1, -1, x, c10::NestedTensorVariant::CPP)));
  // WARNING: Make sure these SymInts to not make their way into the dispatcher.
  // See note in "Comparisons" test above.
  auto p =
      c10::SymInt(c10::SymNode(c10::make_intrusive<c10::NestedIntSymNodeImpl>(
          -1, -1, x, c10::NestedTensorVariant::PYTHON)));

  // NOLINTNEXTLINE(hicpp-avoid-goto,cppcoreguidelines-avoid-goto)
  EXPECT_THROW((void)(c == p), c10::Error);
  // NOLINTNEXTLINE(hicpp-avoid-goto,cppcoreguidelines-avoid-goto)
  EXPECT_THROW((void)(c == c), c10::Error);
  // NOLINTNEXTLINE(hicpp-avoid-goto,cppcoreguidelines-avoid-goto)
  EXPECT_THROW((void)(p == c), c10::Error);
}
