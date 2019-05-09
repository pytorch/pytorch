#include <gtest/gtest.h>

#include <torch/torch.h>

// NOTE: This test suite exists to make sure that common `torch::` functions
// can be used without additional includes beyond `torch/torch.h`.

TEST(TorchIncludeTest, GetSetNumThreads) {
  torch::init_num_threads();
  torch::set_num_threads(2);
  ASSERT_EQ(torch::get_num_threads(), 2);
}
