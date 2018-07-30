#include "caffe2/opt/converter.h"

#include <gtest/gtest.h>

using namespace nom::repr;

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

TEST(Converter, AddNewOpChain) {
  caffe2::NetDef net;
  // create net of Op1->Op2
  caffe2::OperatorDef* def = net.add_op();
  def->set_type("Op1");
  def->add_input("X");
  def->add_output("Y");

  def = net.add_op();
  def->set_type("Op2");
  def->add_input("Y");
  def->add_output("Z");
  auto nn = caffe2::convertToNNModule(net);
  auto& dfg = nn.dataFlow;
  auto& cfg = nn.controlFlow;
  for (auto data_pair : nn::dataIterator<NeuralNetData>(dfg)) {
    NNGraph::NodeRef node;
    NeuralNetData* node_data;
    std::tie(node_data, node) = data_pair;
    if (node_data->getName() != "X") {
      continue;
    }
    // insert two new ops so the correct op order should be
    // Softmax -> Relu -> Op1 -> Op2
    auto consumers = nn::getConsumers(node);
    ASSERT_EQ(consumers.size(), 1);
    NNGraph::NodeRef op1_node = consumers.front();
    auto outputs = nn::getOutputs(op1_node);
    ASSERT_EQ(outputs.size(), 1);
    consumers = nn::getConsumers(outputs.front());
    ASSERT_EQ(consumers.size(), 1);
    NNGraph::NodeRef op2_node = consumers.front();

    dfg.deleteEdge(dfg.getEdge(node, op1_node));
    auto softmax_output = dfg.createNode(
        nom::util::make_unique<nom::repr::Tensor>("softmax_output"));
    auto relu_output = dfg.createNode(
        nom::util::make_unique<nom::repr::Tensor>("relu_output"));
    auto softmax_node = dfg.createNode(nom::util::make_unique<Softmax>());
    auto relu_node = dfg.createNode(nom::util::make_unique<Relu>());
    dfg.createEdge(node, softmax_node);
    dfg.createEdge(softmax_node, softmax_output);
    dfg.createEdge(softmax_output, relu_node);
    dfg.createEdge(relu_node, relu_output);
    dfg.createEdge(relu_output, op1_node);
    nn::coalesceInsertedDataDependencies(&nn);
    // change the op order in cfg to be
    // Op1 -> Op2 -> Softmax -> Relu
    for (auto& bbNode : cfg.getMutableNodes()) {
      auto bb = bbNode->mutableData()->get();
      if (bb->getInstructions().empty()) {
        continue;
      }
      ASSERT_EQ(bb->getInstructions().size(), 4);
      bb->moveInstructionBefore(op1_node, softmax_node);
      bb->moveInstructionBefore(op2_node, softmax_node);
      break;
    }
    break;
  }
  auto new_netdef = caffe2::convertToCaffe2Proto(nn);
  std::vector<std::string> ops;
  for (auto op : new_netdef.op()) {
    ops.emplace_back(op.type());
  }
  std::vector<std::string> result = {"Softmax", "Relu", "Op1", "Op2"};
  ASSERT_EQ(ops.size(), result.size());
  for (auto idx = 0; idx < result.size(); ++idx) {
    ASSERT_EQ(ops.at(idx), result.at(idx));
  }
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
