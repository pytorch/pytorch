#include "caffe2/core/common.h"
#include "caffe2/opt/converter.h"
#include "mobile.h"

#include <gtest/gtest.h>

#define ADD_ARG(_op, _name, _type, _val)    \
  {                                         \
    caffe2::Argument* arg = _op->add_arg(); \
    arg->set_name(_name);                   \
    arg->set_##_type(_val);                 \
  }

TEST(MobileTest, Convolution) {
  caffe2::NetDef net;
  for (auto i = 0; i < 10; ++i) {
    if (i % 3) {
      caffe2::OperatorDef* def = net.add_op();
      def->set_type("Conv");
      def->add_input("X");
      def->add_input("W" + c10::to_string(i));
      def->add_input("b" + c10::to_string(i));
      ADD_ARG(def, "kernel", i, 3);
      ADD_ARG(def, "stride", i, 1);
      ADD_ARG(def, "pad", i, 0);
      ADD_ARG(def, "order", s, "NCHW");
      def->add_output("X");
      def->mutable_device_option()->set_node_name("conv_runner");
    } else {
      caffe2::OperatorDef* def = net.add_op();
      def->set_type("Relu");
      def->add_input("X");
      def->add_output("X");
      def->mutable_device_option()->set_node_name("relu_runner");
    }
  }
  auto nn = caffe2::convertToNNModule(net);
  caffe2::opt::addNNPACK(&nn);
  auto optimized_net = caffe2::convertToCaffe2Proto(nn, net);
  // NOLINTNEXTLINE(performance-for-range-copy)
  for (auto op : optimized_net.op()) {
    if (op.type() == "Conv") {
      assert(op.engine() == "NNPACK");
    }
  }
}
