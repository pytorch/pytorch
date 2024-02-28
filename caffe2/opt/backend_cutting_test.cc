#include "caffe2/core/common.h"
#include "caffe2/core/logging.h"
#include "caffe2/opt/backend_cutting.h"
#include "caffe2/utils/string_utils.h"

#include <gtest/gtest.h>

namespace {
using caffe2::StartsWith;

void AddConv(caffe2::NetDef* net, int tick) {
  auto* op = net->add_op();
  op->set_type("MyConv");
  op->add_input("N" + c10::to_string(tick));
  op->add_input("W" + c10::to_string(tick));
  op->add_input("b" + c10::to_string(tick));
  op->add_output("N" + c10::to_string(tick + 1));
}

bool Supports(const caffe2::OperatorDef& op) {
  return StartsWith(op.type(), "MyConv") || StartsWith(op.type(), "MyRelu") ||
      StartsWith(op.type(), "Concat");
}

caffe2::NetDef Transform(const caffe2::NetDef& net) {
  caffe2::NetDef net_opt;
  auto* op = net_opt.add_op();
  op->set_type("BigOpt");

  for (const auto& i : net.external_input()) {
    // Absorb the weights and bias
    if (!StartsWith(i, "W") && !StartsWith(i, "b")) {
      net_opt.add_external_input(i);
      op->add_input(i);
    }
  }
  for (const auto& i : net.external_output()) {
    net_opt.add_external_output(i);
    op->add_output(i);
  }
  return net_opt;
}
} // namespace

// N0 -> MyConv -> N1
TEST(BackendCuttingTest, unit) {
  caffe2::NetDef net;
  AddConv(&net, 0);
  net.add_external_input("N0");
  net.add_external_input("W0");
  net.add_external_input("b0");
  net.add_external_output("N1");
  auto cutResult = caffe2::opt::OptimizeForBackend(net, Supports, Transform);
  auto net_opt = cutResult.net;
  EXPECT_EQ(0, cutResult.numberOfSubnets);
  EXPECT_EQ(1, net_opt.op_size());
  EXPECT_EQ(1, net_opt.external_input_size());
  EXPECT_EQ(1, net_opt.external_output_size());
}

// X -> CopyIn -> MyConv -> MyConv -> CopyOut -> Y
TEST(BackendCuttingTest, line) {
  caffe2::NetDef net;
  net.add_external_input("X");
  // Adding weights as external inputs to test weight absorption
  net.add_external_input("W0");
  net.add_external_input("W1");
  net.add_external_input("b0");
  net.add_external_input("b1");
  net.add_external_output("Y");
  auto* op = net.add_op();
  op->set_type("CopyIn");
  op->add_input("X");
  op->add_output("N0");
  for (int i = 0; i < 2; ++i) {
    AddConv(&net, i);
  }
  op = net.add_op();
  op->set_type("CopyOut");
  op->add_input("N2");
  op->add_output("Y");
  auto cutResult = caffe2::opt::OptimizeForBackend(net, Supports, Transform);
  auto net_opt = cutResult.net;
  EXPECT_EQ(0, cutResult.numberOfSubnets);
  EXPECT_EQ(3, net_opt.op_size());
}

//  X0 -> CopyIn -> MyConv -|
//                           > Concat -> CopyOut -> Y
//  N2 -> MyConv -> MyRelu -|
TEST(BackendCuttingTest, convergedPaths) {
  caffe2::NetDef net;
  net.add_external_input("X0");
  net.add_external_input("X1");
  net.add_external_input("N2");
  net.add_external_output("Y");
  auto* op = net.add_op();
  op->set_type("CopyIn");
  op->add_input("X0");
  op->add_output("N0");
  AddConv(&net, 0);
  AddConv(&net, 2);
  op = net.add_op();
  op->set_type("MyRelu");
  op->add_input("N3");
  op->add_output("N4");
  op = net.add_op();
  op->set_type("Concat");
  op->add_input("X1");
  op->add_input("N1");
  op->add_input("N4");
  op->add_output("N5");
  op = net.add_op();
  op->set_type("CopyOut");
  op->add_input("N5");
  op->add_output("Y");

  auto cutResult = caffe2::opt::OptimizeForBackend(net, Supports, Transform);
  auto net_opt = cutResult.net;
  EXPECT_EQ(0, cutResult.numberOfSubnets);
  EXPECT_EQ(3, net_opt.op_size());
};

//                -> Random -> Relu -> MyConv4
//                |                           |
// N0 -> MyConv -> MyRelu -> MyConv2 ----------> Concat -> CopyOut -> Y
TEST(BackendCuttingTest, skipPath) {
  caffe2::NetDef net;
  net.add_external_input("N0");
  net.add_external_output("Y");
  AddConv(&net, 0);
  auto* op = net.add_op();
  op->set_type("MyRelu");
  op->add_input("N1");
  op->add_output("N2");
  op = net.add_op();
  op->set_type("Random");
  op->add_input("N1");
  op->add_output("N4");
  op = net.add_op();
  op->set_type("MyRelu");
  op->add_input("N4");
  op->add_output("N5");
  AddConv(&net, 2);
  AddConv(&net, 5);
  op = net.add_op();
  op->set_type("Concat");
  op->add_input("N3");
  op->add_input("N6");
  op->add_output("N7");
  op = net.add_op();
  op->set_type("CopyOut");
  op->add_input("N7");
  op->add_output("Y");

  auto cutResult = caffe2::opt::OptimizeForBackend(net, Supports, Transform);
  auto net_opt = cutResult.net;
  EXPECT_EQ(0, cutResult.numberOfSubnets);
  EXPECT_EQ(4, net_opt.op_size());
}
