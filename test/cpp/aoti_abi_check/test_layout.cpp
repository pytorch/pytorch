#include <gtest/gtest.h>

#include <torch/headeronly/core/Layout.h>

TEST(TestLayout, TestLayout) {
  using torch::headeronly::Layout;
  constexpr Layout expected_layouts[] = {
      torch::headeronly::kStrided,
      torch::headeronly::kSparse,
      torch::headeronly::kSparseCsr,
      torch::headeronly::kMkldnn,
      torch::headeronly::kSparseCsc,
      torch::headeronly::kSparseBsr,
      torch::headeronly::kSparseBsc,
      torch::headeronly::kJagged,
  };
  for (int8_t i = 0; i < static_cast<int8_t>(Layout::NumOptions); i++) {
    EXPECT_EQ(static_cast<Layout>(i), expected_layouts[i]);
  }
}
