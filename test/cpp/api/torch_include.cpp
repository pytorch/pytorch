#include <gtest/gtest.h>

#include <torch/torch.h>

// NOTE: This test suite exists to make sure that common `torch::` functions
// can be used without additional includes beyond `torch/torch.h`.

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(TorchIncludeTest, GetSetNumThreads) {
  torch::init_num_threads();
  torch::set_num_threads(2);
  torch::set_num_interop_threads(2);
  torch::get_num_threads();
  torch::get_num_interop_threads();
}
