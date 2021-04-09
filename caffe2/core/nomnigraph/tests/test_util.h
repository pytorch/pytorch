#ifndef NOM_TESTS_TEST_UTIL_H
#define NOM_TESTS_TEST_UTIL_H

#include "caffe2/core/common.h"
#include "nomnigraph/Graph/Graph.h"
#include "nomnigraph/Graph/Algorithms.h"
#include "nomnigraph/Representations/NeuralNet.h"
#include "nomnigraph/Converters/Dot.h"

#include <map>

class TestClass {
public:
  TestClass() {}
  ~TestClass() {}
};

struct NNEquality {
  static bool equal(
      const typename nom::repr::NNGraph::NodeRef& a,
      const typename nom::repr::NNGraph::NodeRef& b) {
    if (
        !nom::repr::nn::is<nom::repr::NeuralNetOperator>(a) ||
        !nom::repr::nn::is<nom::repr::NeuralNetOperator>(b)) {
      return false;
    }
    auto a_ = nom::repr::nn::get<nom::repr::NeuralNetOperator>(a);
    auto b_ = nom::repr::nn::get<nom::repr::NeuralNetOperator>(b);

    bool sameKind = a_->getKind() == b_->getKind();
    if (sameKind && a_->getKind() == nom::repr::NeuralNetOperator::NNKind::GenericOperator) {
      return a_->getName() == b_->getName();
    }
    return sameKind;
  }
};

// Very simple random number generator used to generate platform independent
// random test data.
class TestRandom {
 public:
  TestRandom(unsigned int seed) : seed_(seed){};

  unsigned int nextInt() {
    seed_ = A * seed_ + C;
    return seed_;
  }

 private:
  static const unsigned int A = 1103515245;
  static const unsigned int C = 12345;
  unsigned int seed_;
};

/** Our test graph looks like this:
 *           +-------+
 *           | entry |
 *           +-------+
 *             |
 *             |
 *             v
 *           +-------+
 *           |   1   |
 *           +-------+
 *             |
 *             |
 *             v
 * +---+     +-------+
 * | 4 | <-- |   2   |
 * +---+     +-------+
 *   |         |
 *   |         |
 *   |         v
 *   |       +-------+
 *   |       |   3   |
 *   |       +-------+
 *   |         |
 *   |         |
 *   |         v
 *   |       +-------+
 *   +-----> |   6   |
 *           +-------+
 *             |
 *             |
 *             v
 * +---+     +-------+
 * | 5 | --> |   7   |
 * +---+     +-------+
 *             |
 *             |
 *             v
 *           +-------+
 *           | exit  |
 *           +-------+
 *
 * Here is the code used to generate the dot file for it:
 *
 *  auto str = nom::converters::convertToDotString(&graph,
 *    [](nom::Graph<std::string>::NodeRef node) {
 *      std::map<std::string, std::string> labelMap;
 *      labelMap["label"] = node->data();
 *      return labelMap;
 *    });
 */
TORCH_API nom::Graph<std::string> createGraph();

TORCH_API nom::Graph<std::string> createGraphWithCycle();

std::map<std::string, std::string> BBPrinter(typename nom::repr::NNCFGraph::NodeRef node);

std::map<std::string, std::string> cfgEdgePrinter(typename nom::repr::NNCFGraph::EdgeRef edge);

std::map<std::string, std::string> NNPrinter(typename nom::repr::NNGraph::NodeRef node);

TORCH_API nom::Graph<TestClass>::NodeRef createTestNode(
    nom::Graph<TestClass>& g);

TORCH_API std::map<std::string, std::string> TestNodePrinter(
    nom::Graph<TestClass>::NodeRef node);
#endif // NOM_TESTS_TEST_UTIL_H
