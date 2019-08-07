#include <gtest/gtest.h>
#include <caffe2/core/tensor.h>
#include <ATen/core/Tensor.h>

TEST(TensorImplTest, Caffe2Constructor) {
  caffe2::Tensor tensor(caffe2::CPU);
  ASSERT_EQ(tensor.strides()[0], 1);
}
