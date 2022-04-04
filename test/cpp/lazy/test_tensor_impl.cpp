#include <gtest/gtest.h>

#include <torch/csrc/lazy/core/tensor_impl.h>
#include <torch/torch.h>

namespace torch {
namespace lazy {

#ifndef USE_LAZY_TS_BACKEND
// Lazy Tensor is disabled in FBCODE until addressing non-virtual methods (e.g. sizes) in TensorImpl
TEST(LazyTensorImplTest, BasicThrow) {
  EXPECT_THROW({
    auto input = torch::rand({0, 1, 3, 0}, torch::TensorOptions(torch::kFloat).device("lazy"));
  }, ::c10::Error);
}
#endif // USE_LAZY_TS_BACKEND

}  // namespace lazy
}  // namespace torch
