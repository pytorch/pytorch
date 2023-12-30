#include <gtest/gtest.h>
#include <stdlib.h>
#include "simple_kernel.hpp"

TEST(SYCLBuildingSysTest, SimpleKernelExecution) {
  int numel = 1024;
  float a[1024];

  // a simple sycl kernel
  itoa(a, numel);

  bool success = true;
  for (int i = 0; i < numel; i++) {
    if (a[i] != i) {
      success = false;
      break;
    }
  }

  ASSERT_TRUE(success);
}
