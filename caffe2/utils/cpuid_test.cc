#include <gtest/gtest.h>
#include "caffe2/utils/cpuid.h"

namespace caffe2 {

TEST(CpuIdTest, ShouldAlwaysHaveMMX) {
  EXPECT_TRUE(GetCpuId().mmx());
}

} // namespace caffe2
