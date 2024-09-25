#include "../../torchinductor/codegen/cpp_prefix.h"
#include <gtest/gtest.h>

TEST(testCppPrefix, testAtomicAddInt) {
  int x = 0;
  atomic_add(&x, 100);
  EXPECT_EQ(x, 100);
}

TEST(testCppPrefix, testAtomicAddFloat) {
  float x = 0.0f;
  atomic_add(&x, 100.0f);
  EXPECT_EQ(x, 100.0f);
}

TEST(testCppPrefix, testAtomicAddI64) {
  int64_t x = 0.0;
  int64_t y = 100.0;
  atomic_add(&x, y);
  EXPECT_EQ(x, 100);
}
