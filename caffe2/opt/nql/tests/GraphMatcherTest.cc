#include "caffe2/opt/nql/graphmatcher.h"
#include <gtest/gtest.h>

using namespace nom::nql;
using namespace nom::repr;

/// \brief Create tensor-nodes in \param graph with names specified in \param
/// names and \return a name->NodeRef map.
std::unordered_map<std::string, NNGraph::NodeRef> genTensors(
    NNGraph& graph,
    std::vector<std::string> names) {
  std::unordered_map<std::string, NNGraph::NodeRef> result;
  for (auto& name : names) {
    result[name] = graph.createNode(caffe2::make_unique<Tensor>(name));
  }
  return result;
}

TEST(Basic, MatchSingleNode) {
  NNGraph graph;
  auto reluInput = graph.createNode(caffe2::make_unique<Tensor>("reluInput"));
  auto relu = graph.createNode(caffe2::make_unique<Relu>());
  auto reluOutput = graph.createNode(caffe2::make_unique<Tensor>("reluOutput"));
  graph.createEdge(reluInput, relu);
  graph.createEdge(relu, reluOutput);

  GraphMatcher gm;
  gm.initFromString(R"NQL(
    def my_nn {
      %x = Relu(%y)
    })NQL");
  EXPECT_TRUE(gm.findSubgraph(graph));

  GraphMatcher gmMismatch;
  gmMismatch.initFromString(R"NQL(
    def my_nn {
      %x = Foo(%y)
    })NQL");
  EXPECT_FALSE(gmMismatch.findSubgraph(graph));
}

TEST(Basic, SyntaxError) {
  NNGraph graph;
  auto reluInput = graph.createNode(caffe2::make_unique<Tensor>("reluInput"));
  auto relu = graph.createNode(caffe2::make_unique<Relu>());
  auto reluOutput = graph.createNode(caffe2::make_unique<Tensor>("reluOutput"));
  graph.createEdge(reluInput, relu);
  graph.createEdge(relu, reluOutput);

  GraphMatcher gm;
  gm.initFromString(R"NQL(
    def my_nn {
      %x =
    })NQL");
  EXPECT_FALSE(gm.findSubgraph(graph));
}

TEST(Basic, Diamond) {
  NNGraph graph;
  /*
   The graph we're building looks like this:

       a        b
        \      /
         Concat
        /      \
       c        d
       |        |
     Relu1     Relu2
       |        |
       e        f
        \      /
          Sum
           |
           x
   */
  auto tensors = genTensors(graph, {"a", "b", "c", "d", "e", "f", "x"});
  auto relu1 = graph.createNode(caffe2::make_unique<Relu>());
  auto relu2 = graph.createNode(caffe2::make_unique<Relu>());
  auto concat = graph.createNode(caffe2::make_unique<Concat>());
  auto sum = graph.createNode(caffe2::make_unique<Sum>());

  graph.createEdge(tensors["a"], concat);
  graph.createEdge(tensors["b"], concat);
  graph.createEdge(concat, tensors["c"]);
  graph.createEdge(concat, tensors["d"]);

  graph.createEdge(tensors["c"], relu1);
  graph.createEdge(relu1, tensors["e"]);

  graph.createEdge(tensors["d"], relu2);
  graph.createEdge(relu2, tensors["f"]);

  graph.createEdge(tensors["e"], sum);
  graph.createEdge(tensors["f"], sum);
  graph.createEdge(sum, tensors["x"]);

  GraphMatcher gm1;
  gm1.initFromString(R"NQL(
    def my_nn {
      %c, %d = Concat(%a, %b)
      %e = Relu(%c)
      %f = Relu(%d)
      %x = Sum(%e, %f)
    })NQL");
  EXPECT_TRUE(gm1.findSubgraph(graph));

  // Check that syntax with inlining works too.
  GraphMatcher gm2;
  gm2.initFromString(R"NQL(
    def my_nn {
      %c, %d = Concat(%a, %b)
      %x = Sum(Relu(%c), Relu(%d))
    })NQL");
  EXPECT_TRUE(gm2.findSubgraph(graph));

  // Check that we understand that the Relu nodes should use output from the
  // same Concat node.
  GraphMatcher gm3;
  gm3.initFromString(R"NQL(
    def my_nn {
      %c = Concat(%a)
      %d = Concat(%b)
      %x = Sum(Relu(%c), Relu(%d))
    })NQL");
  EXPECT_FALSE(gm3.findSubgraph(graph));
}

TEST(Basic, BadDiamond) {
  NNGraph graph;
  /*
   The graph we're building looks like this:

       a         b
       |         |
    Concat1   Concat2
       |         |
       c         d
       |         |
     Relu1      Relu2
       |         |
       e         f
        \       /
           Sum
            |
            x
   */
  auto tensors = genTensors(graph, {"a", "b", "c", "d", "e", "f", "x"});
  auto relu1 = graph.createNode(caffe2::make_unique<Relu>());
  auto relu2 = graph.createNode(caffe2::make_unique<Relu>());
  auto concat1 = graph.createNode(caffe2::make_unique<Concat>());
  auto concat2 = graph.createNode(caffe2::make_unique<Concat>());
  auto sum = graph.createNode(caffe2::make_unique<Sum>());

  graph.createEdge(tensors["a"], concat1);
  graph.createEdge(tensors["b"], concat2);
  graph.createEdge(concat1, tensors["c"]);
  graph.createEdge(concat2, tensors["d"]);

  graph.createEdge(tensors["c"], relu1);
  graph.createEdge(relu1, tensors["e"]);

  graph.createEdge(tensors["d"], relu2);
  graph.createEdge(relu2, tensors["f"]);

  graph.createEdge(tensors["e"], sum);
  graph.createEdge(tensors["f"], sum);
  graph.createEdge(sum, tensors["x"]);

  // Check that we don't match this graph when looking for a diamond shape.
  GraphMatcher gm;
  gm.initFromString(R"NQL(
    def my_nn {
      %c, %d = Concat(%a, %b)
      %x = Sum(Relu(%c), Relu(%d))
    })NQL");
  EXPECT_FALSE(gm.findSubgraph(graph));
  EXPECT_EQ(gm.getMatchMap().size(), 0);

  GraphMatcher gm2;
  gm2.initFromString(R"NQL(
    def my_nn {
      %c = Concat(%a)
      %d = Concat(%b)
      %x = Sum(Relu(%c), Relu(%d))
    })NQL");
  EXPECT_TRUE(gm2.findSubgraph(graph));

  auto matchMap = gm2.getMatchMap();
  EXPECT_EQ(matchMap["%a"], tensors["a"]);
  EXPECT_EQ(matchMap["%b"], tensors["b"]);
  EXPECT_EQ(matchMap["%c"], tensors["c"]);
  EXPECT_EQ(matchMap["%d"], tensors["d"]);
  EXPECT_EQ(matchMap["%x"], tensors["x"]);
  EXPECT_EQ(matchMap["Sum"], sum);
  EXPECT_TRUE(
      (matchMap["Concat"] == concat1) || (matchMap["Concat"] == concat2));
  EXPECT_TRUE((matchMap["Relu"] == relu1) || (matchMap["Relu"] == relu2));
}

TEST(Basic, StarInputs) {
  NNGraph graph;
  /*
   The graph we're building looks like this:

       a       b     c      d
       |       |     |      |
     Relu   Flatten  FC    Sum
       |       |     |      |
       e       f     g      h
        \      |     |     /
          \    |     |   /
            \  |     | /
               Concat
                 |
                 x
   */
  auto tensors =
      genTensors(graph, {"a", "b", "c", "d", "e", "f", "g", "h", "x"});
  auto concat = graph.createNode(caffe2::make_unique<Concat>());
  auto relu = graph.createNode(caffe2::make_unique<Relu>());
  auto flat = graph.createNode(caffe2::make_unique<Flatten>());
  auto fc = graph.createNode(caffe2::make_unique<FC>());
  auto sum = graph.createNode(caffe2::make_unique<Sum>());

  graph.createEdge(tensors["a"], relu);
  graph.createEdge(relu, tensors["e"]);
  graph.createEdge(tensors["b"], flat);
  graph.createEdge(flat, tensors["f"]);
  graph.createEdge(tensors["c"], fc);
  graph.createEdge(fc, tensors["g"]);
  graph.createEdge(tensors["d"], sum);
  graph.createEdge(sum, tensors["h"]);

  graph.createEdge(tensors["e"], concat);
  graph.createEdge(tensors["f"], concat);
  graph.createEdge(tensors["g"], concat);
  graph.createEdge(tensors["h"], concat);
  graph.createEdge(concat, tensors["x"]);

  GraphMatcher gm1;
  gm1.initFromString(R"NQL(
    def my_nn {
      %e = Relu(%a)
      %f = Flatten(%b)
      %g = FC(%c)
      %h = Sum(%d)
      %x = Concat(%e, %f, %g, %h)
    })NQL");
  EXPECT_TRUE(gm1.findSubgraph(graph));
  EXPECT_EQ(gm1.getMatchMap()["Concat"], concat);
  EXPECT_EQ(gm1.getMatchMap()["Relu"], relu);
  EXPECT_EQ(gm1.getMatchMap()["Flatten"], flat);
  EXPECT_EQ(gm1.getMatchMap()["FC"], fc);
  EXPECT_EQ(gm1.getMatchMap()["Sum"], sum);
  EXPECT_EQ(gm1.getMatchMap()["%e"], tensors["e"]);
  EXPECT_EQ(gm1.getMatchMap()["%f"], tensors["f"]);
  EXPECT_EQ(gm1.getMatchMap()["%g"], tensors["g"]);
  EXPECT_EQ(gm1.getMatchMap()["%h"], tensors["h"]);

  GraphMatcher gm2;
  gm2.initFromString(R"NQL(
    def my_nn {
      %x = Concat(*)
    })NQL");
  EXPECT_TRUE(gm2.findSubgraph(graph));
  EXPECT_EQ(gm2.getMatchMap()["Concat"], concat);

  GraphMatcher gm3;
  gm3.initFromString(R"NQL(
    def my_nn {
      %e = Relu(%a)
      %x = Concat(%e, *)
    })NQL");
  EXPECT_TRUE(gm3.findSubgraph(graph));
  EXPECT_EQ(gm3.getMatchMap()["Concat"], concat);
  EXPECT_EQ(gm3.getMatchMap()["Relu"], relu);
  EXPECT_EQ(gm3.getMatchMap()["%e"], tensors["e"]);

  GraphMatcher gm4;
  gm4.initFromString(R"NQL(
    def my_nn {
      %x = Concat(Sum(%a), *)
    })NQL");
  EXPECT_FALSE(gm4.findSubgraph(graph));

  GraphMatcher gm5;
  gm5.initFromString(R"NQL(
    def my_nn {
      %x = Concat(*, Sum(%d))
    })NQL");
  // '*' greedily matches all inputs, and then we fail to match an extra Sum
  // input.
  EXPECT_FALSE(gm5.findSubgraph(graph));
}

TEST(Basic, StarOutputs) {
  NNGraph graph;
  /*
   The graph we're building looks like this:

       a    b    c
        \   |   /
         Concat
        /   |   \
       d    e    f
   */

  auto tensors = genTensors(graph, {"a", "b", "c", "d", "e", "f"});
  auto concat = graph.createNode(caffe2::make_unique<Concat>());

  graph.createEdge(tensors["a"], concat);
  graph.createEdge(tensors["b"], concat);
  graph.createEdge(tensors["c"], concat);
  graph.createEdge(concat, tensors["d"]);
  graph.createEdge(concat, tensors["e"]);
  graph.createEdge(concat, tensors["f"]);

  GraphMatcher gm1;
  gm1.initFromString(R"NQL(
    def my_nn {
      %a, %b, %c = Concat(%d, %e, %f)
    })NQL");
  EXPECT_TRUE(gm1.findSubgraph(graph));
  EXPECT_EQ(gm1.getMatchMap()["Concat"], concat);

  GraphMatcher gm2;
  gm2.initFromString(R"NQL(
    def my_nn {
      * = Concat(*)
    })NQL");
  EXPECT_TRUE(gm2.findSubgraph(graph));
  EXPECT_EQ(gm2.getMatchMap()["Concat"], concat);

  GraphMatcher gm3;
  gm3.initFromString(R"NQL(
    def my_nn {
      %a, * = Concat(*)
    })NQL");
  EXPECT_TRUE(gm3.findSubgraph(graph));
  EXPECT_EQ(gm3.getMatchMap()["Concat"], concat);

  GraphMatcher gm4;
  gm4.initFromString(R"NQL(
    def my_nn {
      %a, %b, * = Concat(%d, %e, *)
    })NQL");
  EXPECT_TRUE(gm4.findSubgraph(graph));
  EXPECT_EQ(gm4.getMatchMap()["Concat"], concat);

  GraphMatcher gm5;
  gm5.initFromString(R"NQL(
    def my_nn {
      %a = Concat(%d, %e, *)
    })NQL");
  // We ignore mismatches in outputs
  EXPECT_TRUE(gm5.findSubgraph(graph));
  EXPECT_EQ(gm5.getMatchMap()["Concat"], concat);

  GraphMatcher gm6;
  gm6.initFromString(R"NQL(
    def my_nn {
      %a, %b, %c, %x = Concat(%d, %e, %f)
    })NQL");
  // We ignore mismatches in outputs
  EXPECT_TRUE(gm6.findSubgraph(graph));
  EXPECT_EQ(gm6.getMatchMap()["Concat"], concat);

  GraphMatcher gm7;
  gm7.initFromString(R"NQL(
    def my_nn {
      %a, %b, %c = Concat(%d, %e)
    })NQL");
  // We don't ignore mismatches in inputs
  EXPECT_FALSE(gm7.findSubgraph(graph));
}

TEST(Caffe2ToNQL, Basic) {
  NNGraph graph;
  /*
   The graph we're building looks like this:

       a
       |
    Concat
       |
       b
       |
     Relu
       |
       c
   */
  auto tensors = genTensors(graph, {"a", "b", "c"});
  auto relu = graph.createNode(caffe2::make_unique<Relu>());
  auto concat = graph.createNode(caffe2::make_unique<Concat>());

  graph.createEdge(tensors["a"], concat);
  graph.createEdge(concat, tensors["b"]);

  graph.createEdge(tensors["b"], relu);
  graph.createEdge(relu, tensors["c"]);

  EXPECT_EQ(convertToNQLString(graph), R"NQL(def nn {
  %b = Concat(%a)
  %c = Relu(%b)
}
)NQL");
}

TEST(Caffe2ToNQL, TensorsNameDeduplication) {
  NNGraph graph;
  /*
   The graph we're building looks like this:

       a
       |
    Concat
       |
       b
       |
     Relu
       |
       c
   */
  std::unordered_map<std::string, NNGraph::NodeRef> tensors;
  // Manually create tensors with the same names. NQL will have to disambiguate
  // the names by adding a suffix.
  tensors["a"] = graph.createNode(caffe2::make_unique<Tensor>("tensor"));
  tensors["b"] = graph.createNode(caffe2::make_unique<Tensor>("tensor"));
  tensors["c"] = graph.createNode(caffe2::make_unique<Tensor>("tensor"));

  auto relu = graph.createNode(caffe2::make_unique<Relu>());
  auto concat = graph.createNode(caffe2::make_unique<Concat>());

  graph.createEdge(tensors["a"], concat);
  graph.createEdge(concat, tensors["b"]);

  graph.createEdge(tensors["b"], relu);
  graph.createEdge(relu, tensors["c"]);

  EXPECT_EQ(convertToNQLString(graph), R"NQL(def nn {
  %tensor_0 = Concat(%tensor)
  %tensor_1 = Relu(%tensor_0)
}
)NQL");
}
