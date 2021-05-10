#include <caffe2/core/tensor.h>
#include <gtest/gtest.h>

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(TensorImplTest, Caffe2Constructor) {
  caffe2::Tensor tensor(caffe2::CPU);
  ASSERT_EQ(tensor.strides()[0], 1);
}
