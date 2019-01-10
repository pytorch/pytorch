#include <random>

#include <gtest/gtest.h>
#include "caffe2/core/context.h"
#include "caffe2/proto/caffe2.pb.h"

namespace caffe2 {

TEST(CPUContextTest, TestAllocAlignment) {
  for (int i = 1; i < 10; ++i) {
    auto data = CPUContext::New(i);
    EXPECT_EQ((reinterpret_cast<size_t>(data.first) % gCaffe2Alignment), 0);
    data.second(data.first);
  }
}

TEST(CPUContextTest, TestAllocDealloc) {
  auto data_and_deleter = CPUContext::New(10 * sizeof(float));
  float* data = static_cast<float*>(data_and_deleter.first);
  EXPECT_NE(data, nullptr);
  auto dst_data_and_deleter = CPUContext::New(10 * sizeof(float));
  float* dst_data = static_cast<float*>(dst_data_and_deleter.first);
  EXPECT_NE(dst_data, nullptr);
  for (int i = 0; i < 10; ++i) {
    data[i] = i;
  }
  DeviceOption option;
  CPUContext context(option);
  context.Copy<float, CPUContext, CPUContext>(10, data, dst_data);
  for (int i = 0; i < 10; ++i) {
    EXPECT_FLOAT_EQ(dst_data[i], i);
  }
  data_and_deleter.second(data);
  dst_data_and_deleter.second(dst_data);
}

}  // namespace caffe2
