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

TEST(TestLayout, operator_left_shift) {
  using torch::headeronly::Layout;

  {
    std::stringstream ss;
    ss << Layout::Strided;
    EXPECT_EQ(ss.str(), "Strided");
  }
  {
    std::stringstream ss;
    ss << Layout::Sparse;
    EXPECT_EQ(ss.str(), "Sparse");
  }
  {
    std::stringstream ss;
    ss << Layout::SparseCsr;
    EXPECT_EQ(ss.str(), "SparseCsr");
  }
  {
    std::stringstream ss;
    ss << Layout::SparseCsc;
    EXPECT_EQ(ss.str(), "SparseCsc");
  }
  {
    std::stringstream ss;
    ss << Layout::SparseBsr;
    EXPECT_EQ(ss.str(), "SparseBsr");
  }
  {
    std::stringstream ss;
    ss << Layout::SparseBsc;
    EXPECT_EQ(ss.str(), "SparseBsc");
  }
  {
    std::stringstream ss;
    ss << Layout::Mkldnn;
    EXPECT_EQ(ss.str(), "Mkldnn");
  }
  {
    std::stringstream ss;
    ss << Layout::Jagged;
    EXPECT_EQ(ss.str(), "Jagged");
  }
}
