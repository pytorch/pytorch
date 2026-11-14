#include <gtest/gtest.h>

#include <torch/headeronly/core/TransposeType.h>

#include <type_traits>

TEST(TestTransposeType, Values) {
  using torch::headeronly::TransposeType;
  static_assert(std::is_same_v<TransposeType, at::native::TransposeType>);
  EXPECT_EQ(static_cast<int>(TransposeType::NoTranspose), 0);
  EXPECT_EQ(static_cast<int>(TransposeType::Transpose), 1);
  EXPECT_EQ(static_cast<int>(TransposeType::ConjTranspose), 2);
}
