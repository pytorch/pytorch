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
      def->add_input("W" + caffe2::to_string(i)); // different weights
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
      caffe2::to_string(rand() % 2));
  auto nn = caffe2::convertToNNModule(net);
  auto new_netdef = caffe2::convertToCaffe2Proto(nn);
}

caffe2::NetDef fakeNet() {
  caffe2::NetDef net;

  {
    caffe2::OperatorDef* def = net.add_op();
    def->set_type("Fake");
    def->add_input("X");
    def->add_output("Y");
  }
  {
    caffe2::OperatorDef* def = net.add_op();
    def->set_type("Fake");
    def->add_input("Y");
    def->add_output("Z");
  }
  {
    caffe2::OperatorDef* def = net.add_op();
    def->set_type("Fake");
    def->add_input("Z");
    def->add_input("X");
    def->add_output("W");
  }
  net.add_external_input("X");
  net.add_external_output("Y");
  net.add_external_output("W");

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
