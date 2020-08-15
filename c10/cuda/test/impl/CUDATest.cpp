#include <gtest/gtest.h>

#include <c10/cuda/impl/CUDATest.h>

using namespace c10::cuda::impl;

TEST(CUDATest, SmokeTest) {
  c10_cuda_test();
}
