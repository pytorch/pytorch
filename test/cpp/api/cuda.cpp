#include <gtest/gtest.h>

#include <torch/cuda.h>

TEST(CUDA, device_count) {
  // Shouldn't throw an error
  torch::cuda::device_count();
}

TEST(CUDA, is_available) {
  // Shouldn't throw an error
  torch::cuda::is_available();
}
