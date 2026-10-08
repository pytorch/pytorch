#include <gtest/gtest.h>

#include <ATen/native/cuda/ScanUtils.cuh>

TEST(ScanUtilsTest, ShortRowsUseMultipleRowsPerBlock) {
  EXPECT_EQ(at::native::get_log_num_threads_x_inner_scan(4096, 8), 4);
  EXPECT_EQ(at::native::get_log_num_threads_x_inner_scan(4097, 8), 4);
  EXPECT_EQ(at::native::get_log_num_threads_x_inner_scan(95760, 8), 4);
}
