#include <algorithm>

#include "test_util.h"

#include "nomnigraph/Representations/NeuralNet.h"
#include "nomnigraph/Support/Pointer.h"
#include "nomnigraph/Transformations/SubgraphMatcher.h"

#include <gtest/gtest.h>

using namespace nom;
using namespace nom::repr;
using namespace nom::repr::nn;

// Test for the NNGraph subgraph matching APIs.
TEST(NeuralNetGraph, ReplaceGraph) {
  NNGraph graph;

  auto input1 = graph.createNode(util::make_unique<Tensor>("input1"));
  auto input2 = graph.createNode(util::make_unique<Tensor>("input2"));
  auto sum = graph.createNode(util::make_unique<Sum>());
  auto sumOutput = graph.createNode(util::make_unique<Tensor>("sumOutput"));
  auto relu = graph.createNode(util::make_unique<Relu>());
  auto reluOutput = graph.createNode(util::make_unique<Tensor>("reluOutput"));

  graph.createEdge(input1, sum);
  graph.createEdge(input2, sum);
  graph.createEdge(sum, sumOutput);
  graph.createEdge(sumOutput, relu);
  graph.createEdge(relu, reluOutput);

  /* input1       input2
         \        /
          \      /
            sum
             |
             |
        sumOutput
             |
           relu
             |
        reluOutput
  */

  auto mg = NNMatchGraph();
  // clang-format off
  auto pattern = subgraph(mg,
      matchNodeType<Relu>(), {
          operatorSubgraph(mg,
              matchNodeType<Sum>(), {
                subgraph(mg, matchNodeType<Tensor>(), {}, 2, true)
              }),
      });
  // clang-format on

  EXPECT_FALSE(NNSubgraphMatcher::isSubgraphMatch(sum, pattern).isMatch());
  EXPECT_FALSE(
      NNSubgraphMatcher::isSubgraphMatch(reluOutput, pattern).isMatch());
  EXPECT_FALSE(NNSubgraphMatcher::isSubgraphMatch(input1, pattern).isMatch());

  EXPECT_TRUE(NNSubgraphMatcher::isSubgraphMatch(relu, pattern).isMatch());

  NNSubgraphMatcher::replaceSubgraph(
      graph, pattern, [](NNGraph& g, NNGraph::NodeRef relu) {
        auto sumOutput = getInputs(relu)[0];
        auto sum = getProducer(sumOutput);

        auto fusedNode = g.createNode(util::make_unique<SumRelu>());
        g.deleteNode(sumOutput);
        g.replaceNode(relu, fusedNode);
        g.replaceNode(sum, fusedNode);

        g.deleteNode(sum);
        g.deleteNode(relu);

        return true;
      });

  /*
      Fused graph:

      input1       input2
         \        /
          \      /
          sumRelu
             |
             |
           output
  */
  EXPECT_EQ(graph.getNodesCount(), 4);
  auto fusedNode = getProducer(reluOutput);
  EXPECT_TRUE(is<SumRelu>(fusedNode));
  EXPECT_EQ(getInputs(fusedNode).size(), 2);
}
