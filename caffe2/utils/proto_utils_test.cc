#include "caffe2/utils/proto_utils.h"
#include <gtest/gtest.h>

namespace caffe2 {

TEST(ProtoUtilsTest, IsSameDevice) {
  DeviceOption a;
  DeviceOption b;
  EXPECT_TRUE(IsSameDevice(a, b));
  a.set_node_name("my_node");
  EXPECT_FALSE(IsSameDevice(a, b));
  b.set_node_name("my_node");
  EXPECT_TRUE(IsSameDevice(a, b));
  b.set_cuda_gpu_id(2);
  EXPECT_FALSE(IsSameDevice(a, b));
  a.set_cuda_gpu_id(2);
  EXPECT_TRUE(IsSameDevice(a, b));
  a.set_device_type(DeviceType::CUDA);
  b.set_device_type(DeviceType::CPU);
  EXPECT_FALSE(IsSameDevice(a, b));
}

TEST(ProtoUtilsTest, SimpleReadWrite) {
  string content("The quick brown fox jumps over the lazy dog.");
  string name = std::tmpnam(nullptr);
  EXPECT_TRUE(WriteStringToFile(content, name.c_str()));
  string read_back;
  EXPECT_TRUE(ReadStringFromFile(name.c_str(), &read_back));
  EXPECT_EQ(content, read_back);
}

}  // namespace caffe2
