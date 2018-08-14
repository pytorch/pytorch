#include "caffe2/ir/nomni_ir.h"
#include "nomnigraph/Converters/Dot.h"

#include "torch/csrc/jit/script/compiler.h"
#include "torch/csrc/jit/script/module.h"

#include "torch/csrc/jit/symbolic_variable.h"

#include <gtest/gtest.h>

TEST(Types, VerifyTypes) {
  torch::jit::Node node(torch::jit::Symbol::fromQualString("test::test"));
  torch::jit::Value value;

  nom::DFGraph dd;
  auto operatorNode = dd.createNode(std::move(node));
  auto valueNode = dd.createNode(torch::jit::Value());

  ASSERT_NO_THROW({
    auto& testOp = operatorNode->data().getOperator();
    LOG(INFO) << (uint32_t)testOp.kind();
  });
  ASSERT_NO_THROW({
    auto& testVal = valueNode->data().getValue();
    LOG(INFO) << (uint32_t)testVal.isTensor();
  });
}

TEST(Types, Errors) {
  torch::jit::Node node(torch::jit::Symbol::fromQualString("test::test"));
  torch::jit::Value value;

  nom::DFGraph dd;
  auto operatorNode = dd.createNode(std::move(node));
  auto valueNode = dd.createNode(torch::jit::Value());

  ASSERT_ANY_THROW(auto& testOp = operatorNode->data().getValue());
  ASSERT_ANY_THROW(auto& testVal = valueNode->data().getOperator());
}

std::string getDotString(nom::NeuralNet& nn) {
  return nom::converters::convertToDotString(
      &nn.dataFlow, [](nom::DFGraph::NodeRef node) {
        std::map<std::string, std::string> labels;
        if (node->data().getKind() == IRNodeKind::Operator) {
          std::ostringstream os;
          // os << node->data().getOperator();
          os << node->data().getOperator().kind().toDisplayString();
          labels["label"] = os.str();
        } else {
          std::ostringstream os;
          os << *node->data().getValue().type();
          labels["label"] = os.str();
        }
        return labels;
      });
}

const static auto jit_example = R"JIT(
  def simple_test(a, b):
      c = a * b
      d = c * c
      return d
)JIT";

TEST(GraphInjestion, ConvertScript) {
  torch::jit::script::Module cu;
  torch::jit::script::defineMethodsInModule(
      cu, jit_example, torch::jit::script::Resolver(), nullptr);

  auto graph = cu.get_method("simple_test").graph();

  nom::NeuralNet nn;
  nom::convert(*graph, &nn);

  LOG(ERROR) << getDotString(nn);
  EXPECT_EQ(nn.dataFlow.getMutableNodes().size(), 6);
}

using namespace torch::jit;
using Var = SymbolicVariable;

std::tuple<Var, Var>
build_lstm_body(Graph& g, Var input, Var hx, Var cx, Var w_ih, Var w_hh) {
  auto gates = input.mm(w_ih);
  gates = gates + hx.mm(w_hh);
  auto outputs = gates.chunk(4, 1);
  auto ingate = outputs[0];
  auto forgetgate = outputs[1];
  auto cellgate = outputs[2];
  auto outgate = outputs[3];
  ingate = ingate.sigmoid();
  outgate = outgate.sigmoid();
  cellgate = cellgate.tanh();
  forgetgate = forgetgate.sigmoid();

  auto cy = forgetgate * cx;
  cy = cy + ingate * cellgate;
  auto hy = outgate * cy.tanh();

  return std::make_tuple(hy, cy);
}

std::shared_ptr<Graph> build_lstm() {
  auto r = std::make_shared<Graph>();
  auto& g = *r;
  Value* input = g.addInput();
  Value* hx = g.addInput();
  Value* cx = g.addInput();
  Value* w_ih = g.addInput();
  Value* w_hh = g.addInput();

  Var hy;
  Var cy;
  std::tie(hy, cy) = build_lstm_body(g, input, hx, cx, w_ih, w_hh);

  hy.addAsOutput();
  cy.addAsOutput();
  g.lint();

  return r;
}

TEST(GraphInjestion, ConvertCpp) {
  auto graph = build_lstm();
  nom::NeuralNet nn;
  nom::convert(*graph, &nn);

  LOG(ERROR) << getDotString(nn);
  EXPECT_EQ(nn.dataFlow.getMutableNodes().size(), 42);
}
