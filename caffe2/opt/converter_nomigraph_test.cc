#include "caffe2/opt/converter.h"

#include <gtest/gtest.h>

#define ADD_ARG(_op, _name, _type, _val)                                       \
{                                                                            \
  caffe2::Argument *arg = _op->add_arg();                                    \
  arg->set_name(_name);                                                      \
  arg->set_##_type(_val);                                                    \
}

TEST(Converter, Basic) {
  caffe2::NetDef net;
  for (auto i = 0; i < 10; ++i) {
    if (rand() % 2) {
      caffe2::OperatorDef *def = net.add_op();
      def->set_type("Conv");
      def->add_input("X");
      def->add_input("W" + std::to_string(i)); // different weights
      ADD_ARG(def, "kernel", i, 3);
      ADD_ARG(def, "stride", i, 1);
      ADD_ARG(def, "pad", i, 0);
      ADD_ARG(def, "order", s, "NCHW");
      def->add_output("X");
      def->mutable_device_option()->set_node_name("conv_runner");
    } else {
      caffe2::OperatorDef *def = net.add_op();
      def->set_type("Relu");
      def->add_input("X");
      def->add_output("X");
      def->mutable_device_option()->set_node_name("relu_runner");
    }
  }
  auto nn = caffe2::convertToNNModule(net);
  auto new_netdef = caffe2::convertToCaffe2Proto(nn);
}

TEST(Converter, UnknownType) {
  caffe2::NetDef net;

  caffe2::OperatorDef *def = net.add_op();
  def->set_type("NeverSeen");
  def->add_input("X");
  def->add_output("X");
  def->mutable_device_option()->set_node_name("device_" +
      std::to_string(rand() % 2));
  auto nn = caffe2::convertToNNModule(net);
  auto new_netdef = caffe2::convertToCaffe2Proto(nn);
}

/* Temporarily disabled While conversion tests
TEST(Converter, While) {
  caffe2::NetDef net;

  caffe2::OperatorDef *def = net.add_op();
  def->set_type("While");
  def->add_input("X");

  caffe2::NetDef body_net;
  {
    caffe2::OperatorDef *rdef = body_net.add_op();
    rdef->set_type("Relu");
    rdef->add_input("X");
    rdef->add_output("X");
  }
  std::string body_net_serialized;
  assert(body_net.SerializeToString(&body_net_serialized));
  ADD_ARG(def, "body", s, body_net_serialized);

  auto nn = caffe2::convertToNNModule(net);
}

TEST(Converter, ComplexWhile) {
  caffe2::NetDef net;

  {
    caffe2::OperatorDef *rdef = net.add_op();
    rdef->set_type("Relu");
    rdef->add_input("X");
    rdef->add_output("X");
  }

  caffe2::OperatorDef *def = net.add_op();
  def->set_type("While");
  def->add_input("X");

  caffe2::NetDef body_net;
  {
    caffe2::OperatorDef *rdef = body_net.add_op();
    rdef->set_type("Instr1");
    rdef->add_input("X");
    rdef->add_output("X");
  }
  {
    caffe2::OperatorDef *rdef = body_net.add_op();
    rdef->set_type("Instr2");
    rdef->add_input("X");
    rdef->add_output("X");
  }
  {
    caffe2::OperatorDef *rdef = body_net.add_op();
    rdef->set_type("Instr3");
    rdef->add_input("X");
    rdef->add_output("X");
  }
  std::string body_net_serialized;
  assert(body_net.SerializeToString(&body_net_serialized));
  ADD_ARG(def, "body", s, body_net_serialized);

  auto nn = caffe2::convertToNNModule(net);
}
*/
