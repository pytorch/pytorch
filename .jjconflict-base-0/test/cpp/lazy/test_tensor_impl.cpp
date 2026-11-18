#include <gtest/gtest.h>

#include <torch/csrc/lazy/core/tensor_impl.h>
#include <torch/torch.h>

namespace torch {
namespace lazy {

#ifdef FBCODE_CAFFE2
// Lazy Tensor is disabled in FBCODE until addressing non-virtual methods (e.g.
// sizes) in TensorImpl
TEST(LazyTensorImplTest, BasicThrow) {
  EXPECT_THROW(
      {
        auto input = torch::rand(
            {0, 1, 3, 0}, torch::TensorOptions(torch::kFloat).device("lazy"));
      },
      ::c10::Error);
}
#endif // FBCODE_CAFFE2

} // namespace lazy
} // namespace torch
