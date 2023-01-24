#include "test_util.h"

#include <string>
#include <sstream>

namespace {

template <typename T>
std::string to_string(T value) {
    std::ostringstream os;
    os << value;
    return os.str();
}

}

nom::Graph<std::string> createGraph() {
  nom::Graph<std::string> graph;
  auto entry = graph.createNode(std::string("entry"));
  auto n1 = graph.createNode(std::string("1"));
  auto n2 = graph.createNode(std::string("2"));
  auto n3 = graph.createNode(std::string("3"));
  auto n4 = graph.createNode(std::string("4"));
  auto n5 = graph.createNode(std::string("5"));
  auto n6 = graph.createNode(std::string("6"));
  auto n7 = graph.createNode(std::string("7"));
  auto exit = graph.createNode(std::string("exit"));
  graph.createEdge(entry, n1);
  graph.createEdge(n1, n2);
  graph.createEdge(n2, n3);
  graph.createEdge(n2, n4);
  graph.createEdge(n3, n6);
  graph.createEdge(n4, n6);
  graph.createEdge(n6, n7);
  graph.createEdge(n5, n7);
  graph.createEdge(n7, exit);
  return graph;
}

nom::Graph<std::string> createGraphWithCycle() {
  nom::Graph<std::string> graph;
  auto entry = graph.createNode(std::string("entry"));
  auto n1 = graph.createNode(std::string("1"));
  auto n2 = graph.createNode(std::string("2"));
  auto n3 = graph.createNode(std::string("3"));
  auto n4 = graph.createNode(std::string("4"));
  auto n5 = graph.createNode(std::string("5"));
  auto n6 = graph.createNode(std::string("6"));
  auto n7 = graph.createNode(std::string("7"));
  auto exit = graph.createNode(std::string("exit"));
  graph.createEdge(entry, n1);
  graph.createEdge(n1, n2);
  graph.createEdge(n2, n3);
  graph.createEdge(n2, n4);
  graph.createEdge(n3, n6);
  graph.createEdge(n6, n3); // Cycle
  graph.createEdge(n4, n6);
  graph.createEdge(n6, n7);
  graph.createEdge(n5, n7);
  graph.createEdge(n7, exit);
  return graph;
}

std::map<std::string, std::string> BBPrinter(typename nom::repr::NNCFGraph::NodeRef node) {
  std::map<std::string, std::string> labelMap;
  auto& bb = node->data();
  labelMap["label"] = to_string((unsigned long long)node) + "\\n";
  for (const auto& instr : bb.getInstructions()) {
    assert(isa<nom::repr::NeuralNetOperator>(instr->data()) &&
           "Invalid instruction.");
    auto *op = dyn_cast<nom::repr::NeuralNetOperator>(instr->data().get());
    bool hasOutput = false;
    for (const auto &outEdge : instr->getOutEdges()) {
      auto *output =
          dyn_cast<nom::repr::NeuralNetData>(outEdge->head()->data().get());
      labelMap["label"] += " " + output->getName();
      hasOutput = true;
    }
    if (hasOutput) {
      labelMap["label"] += " = ";
    }
    labelMap["label"] += op->getName();
    for (const auto &inEdge : instr->getInEdges()) {
      auto *arg =
          dyn_cast<nom::repr::NeuralNetData>(inEdge->tail()->data().get());
      labelMap["label"] += " " + arg->getName();
    }
    labelMap["label"] += "\\l";
  }
  labelMap["shape"] = "box";
  return labelMap;
};

std::map<std::string, std::string> cfgEdgePrinter(typename nom::repr::NNCFGraph::EdgeRef edge) {
  std::map<std::string, std::string> labelMap;
  if (edge->data() == -1) {
    labelMap["label"] = "F";
  } else if (edge->data() == 1) {
    labelMap["label"] = "T";
  }
  return labelMap;
};

std::map<std::string, std::string> NNPrinter(typename nom::repr::NNGraph::NodeRef node) {
  std::map<std::string, std::string> labelMap;
  assert(node->data() && "Node doesn't have data, can't render it");
  if (isa<nom::repr::NeuralNetOperator>(node->data())) {
    auto *op = dyn_cast<nom::repr::NeuralNetOperator>(node->data().get());
    labelMap["label"] =
        op->getName() + " (" + to_string((unsigned long long)node) + ")";
    labelMap["shape"] = "box";
  } else if (isa<nom::repr::Data>(node->data())) {
    auto tensor = dyn_cast<nom::repr::NeuralNetData>(node->data().get());
    labelMap["label"] = tensor->getName();
    labelMap["label"] += "_" + to_string(tensor->getVersion()) + " " + to_string((unsigned long long)node);
  }
  return labelMap;
};

nom::Graph<TestClass>::NodeRef createTestNode(nom::Graph<TestClass>& g) {
  return g.createNode(TestClass());
}

std::map<std::string, std::string> TestNodePrinter(
    nom::Graph<TestClass>::NodeRef /* unused */) {
  std::map<std::string, std::string> labelMap;
  labelMap["label"] = "Node";
  return labelMap;
}
