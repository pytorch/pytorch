#include <gtest/gtest.h>

#include <torch/headeronly/macros/Export.h>

namespace torch {
namespace aot_inductor {

C10_API bool equal(int a, int b) {
  return a == b;
}

TEST(TestMacros, TestC10API) {
  EXPECT_TRUE(equal(1, 1));
  EXPECT_FALSE(equal(1, 2));
}

} // namespace aot_inductor
} // namespace torch
