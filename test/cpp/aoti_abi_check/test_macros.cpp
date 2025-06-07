#include <gtest/gtest.h>

#include <torch/standalone/macros/Export.h>
#include <torch/standalone/macros/Macros.h>

namespace torch {
namespace aot_inductor {

C10_API bool equal(int a, int b) {
  return a == b;
}

TEST(TestMacros, TestC10API) {
  EXPECT_TRUE(equal(1, 1));
  EXPECT_FALSE(equal(1, 2));
}

TEST(TestMacros, TestC10Macros) {
  EXPECT_TRUE(C10_STRINGIZE("whatever") == "whatever");
  EXPECT_TRUE(C10_LIKELY(true));
  EXPECT_TRUE(C10_UNLIKELY(false));
}

} // namespace aot_inductor
} // namespace torch
