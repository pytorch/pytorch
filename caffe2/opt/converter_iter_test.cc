#include "caffe2/opt/converter.h"
#include "nomnigraph/Graph/Algorithms.h"

#include <gtest/gtest.h>
#include <random>
using namespace nom::repr;

#define ADD_ARG(_op, _name, _type, _val)    \
  {                                         \
    caffe2::Argument* arg = _op->add_arg(); \
    arg->set_name(_name);                   \
    arg->set_##_type(_val);                 \
  }

#define ASSERT_VECTOR_EQUAL(_vec1, _vec2)             \
  {                                                   \
    ASSERT_EQ(_vec1.size(), _vec2.size());            \
    for (size_t idx = 0; idx < _vec1.size(); idx++) { \
      ASSERT_EQ(_vec1[idx], _vec2[idx]);              \
    }                                                 \
  }

caffe2::NetDef generateComplexNet(int iterations) {
  // create a graph from a complex network that consists from 100 edges
  // connecting from 200 potential nodes, and delete loops from it.
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> dis(0, 200);
  caffe2::NetDef net;
  do {
    net.clear_op();
    for (int iter = 0; iter < iterations; iter++) {
      auto op = net.add_op();
      op->set_type(string("op_") + caffe2::to_string(iter));
      op->add_input(caffe2::to_string(char(dis(gen))));
      op->add_output(caffe2::to_string(char(dis(gen))));
    }
    auto nn = caffe2::convertToNNModule(net);
    auto ssc = nom::algorithm::tarjans(&nn.controlFlow);
    for (auto iter = ssc.rbegin(); iter != ssc.rend(); ++iter) {
      if (iter->getNodes().size() > 1) {
        // loop detected, delete all ops related to this loop.
        for (auto& bbNode : iter->getNodes()) {
          auto instr_copy = bbNode->data().get()->getInstructions();
          for (auto instr : instr_copy) {
            nn.dataFlow.deleteNode(instr);
          }
        }
      }
    }
    auto prune_net = caffe2::convertToCaffe2Proto(nn);
    vector<caffe2::OperatorDef> ops_to_keep;
    for (auto op : prune_net.op()) {
      if (op.input_size() != 0 || op.output_size() != 0) {
        ops_to_keep.emplace_back(op);
      }
    }
    prune_net.clear_op();
    for (auto op : ops_to_keep) {
      prune_net.add_op()->CopyFrom(op);
    }
    net = prune_net;
  } while (net.op().size() == 0);
  return net;
}

// tests for nn::iter() functionality of nomnigraph. Put it in here Because
// it is mostly used in conjunction with convertToNNModule().
TEST(ConverterIter, dataflow) {
  caffe2::NetDef net;
  caffe2::OperatorDef* def = net.add_op();
  def->set_type("op_A");
  def->add_input("X");
  def->add_output("Y");

  caffe2::OperatorDef* def2 = net.add_op();
  def2->set_type("op_B");
  def2->add_input("Y");
  def2->add_output("Z");
  auto nn = caffe2::convertToNNModule(net);
  auto& dfg = nn.dataFlow;
  std::vector<std::string> names;
  auto all_noderefs = dfg.getMutableNodes();
  for (auto& node : nn::iterate(nn.dataFlow)) {
    // ensure that the node are passed by reference
    ASSERT_TRUE(
        std::find(all_noderefs.begin(), all_noderefs.end(), &node) !=
        all_noderefs.end());
    if (nn::is<NeuralNetData>(&node)) {
      auto data = nn::get<NeuralNetData>(&node);
      names.emplace_back(data->getName());
    } else if (nn::is<NeuralNetOperator>(&node)) {
      auto op = nn::get<NeuralNetOperator>(&node);
      names.emplace_back(op->getName());
    }
  }
  std::vector<std::string> result = {"op_A", "X", "Y", "op_B", "Z"};
  ASSERT_VECTOR_EQUAL(names, result);

  names.clear();
  // switch op_B to Relu when visiting op_A. Thus we should not print out
  // name "op_B" since it is deleted, neither should we print out "Relu" since
  // it is inserted during iterate() traversal.
  for (auto& node : nn::iterate(nn.dataFlow)) {
    if (nn::is<NeuralNetData>(&node)) {
      auto data = nn::get<NeuralNetData>(&node);
      names.emplace_back(data->getName());
    } else if (nn::is<NeuralNetOperator>(&node)) {
      auto op = nn::get<NeuralNetOperator>(&node);
      if (op->getName() == "op_A") {
        auto outputs = nn::getOutputs(&node);
        assert(outputs.size() == 1);
        auto Y = outputs[0];
        assert(nn::hasConsumer(Y));
        auto consumers = nn::getConsumers(Y);
        assert(consumers.size() == 1);
        auto op_B = consumers[0];
        auto relu_node =
            dfg.createNode(nom::util::make_unique<nom::repr::Relu>());
        dfg.swapNodes(relu_node, op_B);
        dfg.deleteNode(op_B);
      }
      names.emplace_back(op->getName());
    }
  }

  result = {"op_A", "X", "Y", "Z"};
  ASSERT_VECTOR_EQUAL(names, result);
}

TEST(ConverterIter, large_graph) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> dis(50, 200);

  for (int iter = 0; iter < 10; ++iter) {
    auto net = generateComplexNet(dis(gen));
    ASSERT_GT(net.op().size(), 0);
    auto nn = caffe2::convertToNNModule(net);
    // traverse the data flow to see it works
    vector<string> names;
    for (auto& node : nn::iterate(nn.dataFlow)) {
      if (nn::is<NeuralNetData>(&node)) {
        auto data = nn::get<NeuralNetData>(&node);
        names.emplace_back(data->getName());
      } else if (nn::is<NeuralNetOperator>(&node)) {
        auto op = nn::get<NeuralNetOperator>(&node);
        names.emplace_back(op->getName());
      }
    }

    // traverse the control flow to check the op names match with ops in NetDef
    vector<string> op_names;
    for (auto& node : nn::iterate(nn.controlFlow, nn.dataFlow)) {
      assert(nn::is<NeuralNetOperator>(&node));
      auto op = nn::get<NeuralNetOperator>(&node);
      op_names.emplace_back(op->getName());
    }
    ASSERT_EQ(op_names.size(), net.op().size());
    for (auto op : net.op()) {
      auto name = op.type();
      ASSERT_TRUE(
          std::find(op_names.begin(), op_names.end(), name) != op_names.end());
    }
  }
}

TEST(ConverterIter, controlFlow_circle_net) {
  caffe2::NetDef net;

  caffe2::OperatorDef* def1 = net.add_op();
  def1->set_type("op_A");
  def1->add_input("X");
  def1->add_output("Y");

  caffe2::OperatorDef* def2 = net.add_op();
  def2->set_type("op_B");
  def2->add_input("X");
  def2->add_input("Y");
  def2->add_output("Z");

  caffe2::OperatorDef* def3 = net.add_op();
  def3->set_type("op_C");
  def3->add_input("X");
  def3->add_input("Y");
  def3->add_input("Z");
  def3->add_output("A");

  auto nn = caffe2::convertToNNModule(net);

  std::vector<std::string> names;
  for (auto& node : nn::iterate(nn.controlFlow, nn.dataFlow)) {
    assert(nn::is<NeuralNetOperator>(&node));
    auto op = nn::get<NeuralNetOperator>(&node);
    names.emplace_back(op->getName());
  }
  // make sure that op_C is at a last position;
  std::vector<std::string> result = {"op_A", "op_B", "op_C"};
  ASSERT_VECTOR_EQUAL(names, result);
}
