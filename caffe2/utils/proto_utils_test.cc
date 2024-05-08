#include <gtest/gtest.h>

#include "caffe2/core/test_utils.h"
#include "caffe2/utils/proto_utils.h"

namespace caffe2 {

TEST(ProtoUtilsTest, IsSameDevice) {
  DeviceOption a;
  DeviceOption b;
  EXPECT_TRUE(IsSameDevice(a, b));
  a.set_node_name("my_node");
  EXPECT_FALSE(IsSameDevice(a, b));
  b.set_node_name("my_node");
  EXPECT_TRUE(IsSameDevice(a, b));
  b.set_device_id(2);
  EXPECT_FALSE(IsSameDevice(a, b));
  a.set_device_id(2);
  EXPECT_TRUE(IsSameDevice(a, b));
  a.set_device_type(DeviceTypeProto::PROTO_CUDA);
  b.set_device_type(DeviceTypeProto::PROTO_CPU);
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

TEST(ProtoUtilsTest, CleanupExternalInputsAndOutputs) {
  caffe2::NetDef net;
  caffe2::testing::NetMutator(&net)
      .newOp("op1", {"X1", "X2"}, {"Y"})
      .newOp("op2", {"W", "Y"}, {"Z1", "Z2"})
      .newOp("op3", {"Z2", "W"}, {"O"})
      .externalInputs({"X1", "X3", "X1", "W"})
      .externalOutputs({"O", "Z2", "Z3", "O", "X3"});
  cleanupExternalInputsAndOutputs(&net);

  std::vector<std::string> externalInputs;
  for (const auto& inputName : net.external_input()) {
    externalInputs.emplace_back(inputName);
  }
  // The 2nd X1 is removed because of duplication.
  // X2 is added because it should be a missing external input.
  std::vector<std::string> expectedExternalInputs{"X1", "X3", "W", "X2"};
  EXPECT_EQ(externalInputs, expectedExternalInputs);

  std::vector<std::string> externalOutputs;
  for (const auto& outputName : net.external_output()) {
    externalOutputs.emplace_back(outputName);
  }
  // Z3 is removed because it's not an output of any operator in the net.
  // The 2nd O is removed because of duplication.
  std::vector<std::string> expectedexternalOutputs{"O", "Z2", "X3"};
  EXPECT_EQ(externalOutputs, expectedexternalOutputs);
}

} // namespace caffe2
