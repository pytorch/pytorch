#include <random>

#include "caffe2/proto/caffe2.pb.h"
#include "caffe2/core/context.h"
#include <gtest/gtest.h>

namespace caffe2 {

TEST(CPUContextTest, TestAllocAlignment) {
  for (int i = 1; i < 10; ++i) {
    void* data = CPUContext::New(i);
    EXPECT_EQ((reinterpret_cast<size_t>(data) % gCaffe2Alignment), 0);
    CPUContext::Delete(data);
  }
}

TEST(CPUContextTest, TestAllocDealloc) {
  float* data = static_cast<float*>(CPUContext::New(10 * sizeof(float)));
  EXPECT_NE(data, nullptr);
  float* dst_data = static_cast<float*>(CPUContext::New(10 * sizeof(float)));
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
  CPUContext::Delete(data);
  CPUContext::Delete(dst_data);
}

}  // namespace caffe2
