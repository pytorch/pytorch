#include <gtest/gtest.h>

#include <ATen/ATen.h>
#include <ATen/test/test_assert.h>
#include <ATen/native/Pool.h>

using namespace at;

template<typename T>
static inline float outputSize(
    T inputSize, T kernelSize, T pad, T stride, T dilation) {
  return (inputSize + 2 * pad - dilation * (kernelSize - 1) - 1) / stride;
}

TEST(TestPooling, CeilModeWithNegative) {
  // Let's make pad == 0 and test ceil mode
  int64_t pad = 0;
  int64_t inputSize = 6;
  int64_t dilation = 2;
  int64_t kernelSize = 4;
  int64_t stride = 2;
  ASSERT_EQ(-0.5, outputSize(inputSize, kernelSize, pad, stride, dilation));
  // floor(-0.5) + 1 == 0
  ASSERT_EQ(0, pooling_output_shape(inputSize, kernelSize, pad, stride, dilation, false));
  //ceil(-0.5) + 1 == 1
  ASSERT_EQ(1, pooling_output_shape(inputSize, kernelSize, pad, stride, dilation, true));
}

TEST(TestPooling, CeilModeWithNegative) {
  // Let's make pad == 0 and test ceil mode
  int64_t pad = 0;
  int64_t inputSize = 7;
  int64_t dilation = 2;
  int64_t kernelSize = 4;
  int64_t stride = 2;
  ASSERT_EQ(0.5, outputSize(inputSize, kernelSize, pad, stride, dilation));
  // floor(0.5) + 1 == 1
  ASSERT_EQ(1, pooling_output_shape(inputSize, kernelSize, pad, stride, dilation, false));
  // ceil(0.5) + 1 == 2
  ASSERT_EQ(2, pooling_output_shape(inputSize, kernelSize, pad, stride, dilation, true));
}