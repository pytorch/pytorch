#include <gtest/gtest.h>

#include <torch/csrc/utils/memory.h>

#include <c10/util/Optional.h>

struct TestValue {
  explicit TestValue(const int& x) : lvalue_(x) {}
  explicit TestValue(int&& x) : rvalue_(x) {}

  c10::optional<int> lvalue_;
  c10::optional<int> rvalue_;
};

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(MakeUniqueTest, ForwardRvaluesCorrectly) {
  auto ptr = torch::make_unique<TestValue>(123);
  ASSERT_FALSE(ptr->lvalue_.has_value());
  ASSERT_TRUE(ptr->rvalue_.has_value());
  ASSERT_EQ(*ptr->rvalue_, 123);
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(MakeUniqueTest, ForwardLvaluesCorrectly) {
  int x = 5;
  auto ptr = torch::make_unique<TestValue>(x);
  ASSERT_TRUE(ptr->lvalue_.has_value());
  ASSERT_EQ(*ptr->lvalue_, 5);
  ASSERT_FALSE(ptr->rvalue_.has_value());
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(MakeUniqueTest, CanConstructUniquePtrOfArray) {
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
  auto ptr = torch::make_unique<int[]>(3);
  // Value initialization is required by the standard.
  ASSERT_EQ(ptr[0], 0);
  ASSERT_EQ(ptr[1], 0);
  ASSERT_EQ(ptr[2], 0);
}
