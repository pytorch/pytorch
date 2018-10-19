#include <gtest/gtest.h>

#include <ATen/ATen.h>

#include <cmath>

TEST(Tensor, AllocatesTensorOnTheCorrectDevice_MultiCUDA) {
  auto tensor = at::tensor({1, 2, 3}, at::device({at::kCUDA, 1}));
  ASSERT_EQ(tensor.device().type(), at::Device::Type::CUDA);
  ASSERT_EQ(tensor.device().index(), 1);
}
