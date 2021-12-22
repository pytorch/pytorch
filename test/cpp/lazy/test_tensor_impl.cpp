#include <gtest/gtest.h>

#include <torch/csrc/lazy/core/tensor_impl.h>
#include <torch/torch.h>

namespace torch {
namespace lazy {

// TODO(alanwaketan): Update the following unit tests once the TorchScript backend is merged.
TEST(LazyTensorImplTest, BasicThrow) {
  EXPECT_THROW({
    auto input = torch::rand({0, 1, 3, 0}, torch::TensorOptions(torch::kFloat).device("lazy"));
  }, ::c10::Error);
}

}  // namespace lazy
}  // namespace torch
