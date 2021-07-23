#include <gtest/gtest.h>
#include "caffe2/utils/cpuid.h"

namespace caffe2 {

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(CpuIdTest, ShouldAlwaysHaveMMX) {
  EXPECT_TRUE(GetCpuId().mmx());
}

} // namespace caffe2
