#include "caffe2/core/common.h"
#include "caffe2/core/logging.h"
#include "caffe2/onnx/onnx_exporter.h"

#include <gtest/gtest.h>
#include <string>
#include <tuple>
#include <unordered_map>

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(SsaTest, ConvReluInplace) {
  caffe2::NetDef net;
  auto* op = net.add_op();
  op->set_type("Conv");
  op->add_input("X");
  op->add_input("W");
  op->add_input("b");
  op->add_output("Y");
  op = net.add_op();
  op->set_type("Relu");
  op->add_input("Y");
  op->add_output("Y");
  net.add_external_input("X");
  net.add_external_output("Y");

  std::unordered_map<std::string, std::string> input_mapping =
      caffe2::onnx::SsaRewrite(nullptr, &net);
  for (const auto& net_op : net.op()) {
    std::unordered_set<std::string> inputs;
    for (const auto& i : net_op.input()) {
      inputs.emplace(i);
    }
    for (const auto& o : net_op.output()) {
      EXPECT_TRUE(inputs.count(o) == 0);
    }
  }
  EXPECT_EQ(net.op(0).output(0), net.op(1).input(0));
  EXPECT_EQ("X", input_mapping.at(net.external_input(0)));
  EXPECT_EQ("Y", net.external_output(0));
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(SsaTest, FC_Relu_FC_InPlace_Output) {
  caffe2::NetDef net;
  auto* op = net.add_op();
  op->set_type("FC");
  op->add_input("X");
  op->add_input("W0");
  op->add_input("b0");
  op->add_output("Y");
  op = net.add_op();
  op->set_type("Relu");
  op->add_input("Y");
  op->add_output("Y");
  op = net.add_op();
  op->set_type("FC");
  op->add_input("Y");
  op->add_input("W2");
  op->add_input("b2");
  op->add_output("Z");
  net.add_external_input("X");
  net.add_external_output("Y");
  net.add_external_output("Z");

  std::unordered_map<std::string, std::string> input_mapping =
      caffe2::onnx::SsaRewrite(nullptr, &net);
  for (const auto& net_op : net.op()) {
    std::unordered_set<std::string> inputs;
    for (const auto& i : net_op.input()) {
      inputs.emplace(i);
    }
    for (const auto& o : net_op.output()) {
      EXPECT_TRUE(inputs.count(o) == 0);
    }
  }
  EXPECT_EQ(net.op(0).output(0), net.op(1).input(0));
  EXPECT_EQ("Y", net.op(2).input(0));
  EXPECT_EQ("Y_0", net.op(1).input(0));
  EXPECT_EQ("X", input_mapping.at(net.external_input(0)));
  EXPECT_EQ("Y", net.external_output(0));
  EXPECT_EQ("Z", net.external_output(1));
}
