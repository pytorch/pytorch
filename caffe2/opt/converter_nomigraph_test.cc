#include "caffe2/core/test_utils.h"
#include "caffe2/opt/converter.h"

#include <gtest/gtest.h>

TEST(Converter, Basic) {
  using namespace caffe2::testing;
  caffe2::NetDef net;
  for (auto i = 0; i < 10; ++i) {
    // NOLINTNEXTLINE(clang-analyzer-security.insecureAPI.rand)
    if (rand() % 2) {
      NetMutator(&net)
          .newOp("Conv", {"X", "W" + c10::to_string(i)}, {"X"})
          .addArgument("kernel", 3)
          .addArgument("stride", 1)
          .addArgument("pad", 0)
          .addArgument("order", std::string("NCHW"))
          .setDeviceOptionName("conv_runner");
    } else {
      NetMutator(&net)
          .newOp("Relu", {"X"}, {"X"})
          .setDeviceOptionName("relu_runner");
    }
  }
  auto nn = caffe2::convertToNNModule(net);
  auto new_netdef = caffe2::convertToCaffe2Proto(nn);
}

TEST(Converter, UnknownType) {
  using namespace caffe2::testing;
  caffe2::NetDef net;
  NetMutator(&net)
      .newOp("NeverSeen", {"X"}, {"X"})
      .setDeviceOptionName("device_" + c10::to_string(rand() % 2));
  auto nn = caffe2::convertToNNModule(net);
  auto new_netdef = caffe2::convertToCaffe2Proto(nn);
}

TEST(Converter, SpecializeConverter) {
  using namespace caffe2::testing;
  caffe2::NetDef net;
  NetMutator(&net).newOp("Slice", {"X"}, {"X"}).setDeviceOptionName("abc");
  EXPECT_EQ(net.op(0).device_option().node_name(), "abc");
  auto nn = caffe2::convertToNNModule(net);
  auto new_netdef = caffe2::convertToCaffe2Proto(nn);
  EXPECT_EQ(new_netdef.op(0).device_option().node_name(), "abc");
}

caffe2::NetDef fakeNet() {
  using namespace caffe2::testing;
  caffe2::NetDef net;
  NetMutator(&net)
      .newOp("Fake", {"X"}, {"Y"})
      .newOp("Fake", {"Y"}, {"Z"})
      .newOp("Fake", {"Z", "X"}, {"W"})
      .externalInputs({"X"})
      .externalOutputs({"Y", "W"});
  return net;
}

TEST(Converter, ExternalInputs) {
  auto net = fakeNet();

  auto nn = caffe2::convertToNNModule(net);
  auto new_netdef = caffe2::convertToCaffe2Proto(nn);
  EXPECT_EQ(new_netdef.external_input().size(), net.external_input().size());
  for (auto i = 0; i < net.external_input().size(); ++i) {
    EXPECT_EQ(new_netdef.external_input(i), net.external_input(i));
  }
}

TEST(Converter, ExternalOutputs) {
  auto net = fakeNet();

  auto nn = caffe2::convertToNNModule(net);
  auto new_netdef = caffe2::convertToCaffe2Proto(nn);
  EXPECT_EQ(new_netdef.external_output().size(), net.external_output().size());
  for (auto i = 0; i < net.external_output().size(); ++i) {
    EXPECT_EQ(new_netdef.external_output(i), net.external_output(i));
  }
}

TEST(Converter, InjectDataEdgeIndicators) {
  auto net = fakeNet();
  caffe2::injectDataEdgeIndicators(&net);

  EXPECT_EQ(net.op_size(), 3 + 1 + 2); // Inserted 1 Declare and 2 Export

  auto declare_count = 0;
  auto export_count = 0;
  for (const auto& op : net.op()) {
    declare_count += op.type() == "Declare";
    export_count += op.type() == "Export";
  }
  EXPECT_EQ(declare_count, 1);
  EXPECT_EQ(export_count, 2);

  // Remove them from the network
  EXPECT_EQ(net.external_input_size(), 0);
  EXPECT_EQ(net.external_output_size(), 0);

  // Ensure nomnigraph can handle this change
  auto nn = caffe2::convertToNNModule(net);
  auto new_net = caffe2::convertToCaffe2Proto(nn);

  caffe2::removeDataEdgeIndicators(&new_net);

  for (const auto& op : new_net.op()) {
    EXPECT_NE(op.type(), "Declare");
    EXPECT_NE(op.type(), "Export");
  }

  EXPECT_EQ(new_net.external_input_size(), 1);
  EXPECT_EQ(new_net.external_output_size(), 2);
}
