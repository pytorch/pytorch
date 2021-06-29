#include <algorithm>
#include <memory>

#include "test_util.h"

#include "nomnigraph/Representations/NeuralNet.h"
#include "nomnigraph/Transformations/SubgraphMatcher.h"

#include <gtest/gtest.h>

using namespace nom;
using namespace nom::repr;
using namespace nom::repr::nn;

// Test for the NNGraph subgraph matching APIs.
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(NeuralNetGraph, ReplaceGraph) {
  NNGraph graph;

  auto input1 = graph.createNode(std::make_unique<Tensor>("input1"));
  auto input2 = graph.createNode(std::make_unique<Tensor>("input2"));
  // Test renaming blob
  nn::get<Tensor>(input2)->setName("input2_renamed");
  auto sum = graph.createNode(std::make_unique<Sum>());
  auto sumOutput = graph.createNode(std::make_unique<Tensor>("sumOutput"));
  auto relu = graph.createNode(std::make_unique<Relu>());
  auto reluOutput = graph.createNode(std::make_unique<Tensor>("reluOutput"));

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
  auto matchSumInput =
      mg.createNode(std::move(matchExternalTensorNode().count(2)));
  auto matchSum = mg.createNode(nn::is<Sum>);
  mg.createEdge(matchSumInput, matchSum);

  auto matchSumOutput = mg.createNode(nn::is<Tensor>);
  mg.createEdge(matchSum, matchSumOutput);

  auto matchRelu = mg.createNode(nn::is<Relu>);
  mg.createEdge(matchSumOutput, matchRelu);

  auto matchRoot = matchRelu;
  EXPECT_FALSE(mg.isSubgraphMatch(sum, matchRoot).isMatch());
  EXPECT_FALSE(mg.isSubgraphMatch(reluOutput, matchRoot).isMatch());
  EXPECT_FALSE(mg.isSubgraphMatch(input1, matchRoot).isMatch());

  EXPECT_TRUE(mg.isSubgraphMatch(relu, matchRoot).isMatch());

  mg.replaceSubgraph(
      graph,
      matchRoot,
      [&matchSumOutput](
          NNGraph& g,
          NNGraph::NodeRef relu,
          const NNMatchGraph::SubgraphMatchResultType& matchResult) {
        auto fusedNode = g.createNode(std::make_unique<SumRelu>());
        auto sumNode =
            getProducer(matchResult.getMatchNodeMap()->at(matchSumOutput));
        g.replaceOutEdges(relu, fusedNode);
        g.replaceInEdges(sumNode, fusedNode);
        g.deleteNodes(matchResult.getMatchedSubgraph()->getNodes());
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
