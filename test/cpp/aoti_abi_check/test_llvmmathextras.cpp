#include <gtest/gtest.h>

#include <torch/headeronly/util/llvmMathExtras.h>

#include <cstdint>

TEST(TestLlvmMathExtras, TestLlvmMathExtras) {
  EXPECT_EQ(c10::llvm::findFirstSet(uint64_t(0b1000)), uint64_t(3));
  EXPECT_TRUE(c10::llvm::isPowerOf2_64(64));
  EXPECT_FALSE(c10::llvm::isPowerOf2_64(63));
  EXPECT_EQ(c10::llvm::Log2_64(64), 6u);
}
