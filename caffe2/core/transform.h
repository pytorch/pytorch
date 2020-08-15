#pragma once

#include "caffe2/core/common.h"
#include "caffe2/core/graph.h"
#include "caffe2/core/workspace.h"
#include "caffe2/proto/caffe2_pb.h"
#include "caffe2/utils/proto_utils.h"

namespace caffe2 {

/**
 * The Transform Base Object
 *
 * A Transform is an operation which manipulates a Caffe2 NetDef.
 * You can consider it as a function: Transform.ApplyTo(NetDef) -> NetDef
 *
 * A Transform Operation does 4 things:
 *    1) Creates a Graph object from a NetDef, which stores connections.
 *    2) Pattern Matches on the Graph, to find subgraphs it wants to change.
 *    3) Replaces the subgraphs that it's matched with new operators.
 *    4) Creates a NetDef from the changed Graph, and returns it.
 *
 * The effect of a Transform is defined by its 3 protected virtual functions.
 *    1) PatternRule determines for an ordered subgraph and a node, whether to
 *        consider adding the node to the subgraph.
 *    2) ValidatorRule determines, for an ordered subgraph, whether it is a
 *        match.
 *    3) ReplaceRule mutates the graph, based on a matched subgraph.
 *
 * This is the base class for all derived classes to base off. To create your
 * own transform, write your implementations for PatternRule, ValidatorRule, and
 * ReplaceRule.
 */
class CAFFE2_API Transform {
 public:
  Transform() {}

  /**
   * Apply a Transform onto a NetDef.
   * Returns the transformed NetDef.
   */
  NetDef ApplyTo(const NetDef& orig_net_def);

  virtual ~Transform() {}

  /**
   * Determines the type of subgraphs that PatternMatch will find.
   *
   * CONNECTED_SUBGRAPH will only match subgraphs that are connected.
   * These subgraphs satisfy that every node of the match is connected to the
   * subgraph of the nodes that come before it.
   * For example, in the graph (1) --> (2) --> (3) --> (4),
   *    This is capable of matching the subgraph [2, 3] and [4, 3]
   *    This is not capable of matching the subgraph [2, 4].
   *
   *
   * SORTED_WRT_EXECUTION_ORDER will match subgraphs that guarantee
   * sorted execution order.
   * The nodes don't have to be connected. It is faster than General.
   * For example, in the graph (1) --> (2) --> (3) --> (4),
   *    This is capable of matching the subgraph [2, 4], [3, 4].
   *    This is not capable of matching the subgraph [3, 1], [4, 3].
   *
   *
   * GENERAL can match any subgraph.
   * For example, in the graph (1) --> (2) --> (3) --> (4),
   *    This is capable of matching subgraphs [2, 4], [3, 4], [4, 2, 1].
   *    There is no ordered subgraph of G that cannot be matched by this.
   */
  enum PatternMatchType {
    CONNECTED_SUBGRAPH,
    SORTED_WRT_EXECUTION_ORDER,
    GENERAL
  };

  /**
   * Generates all matches (stored as ordered subgraphs) and returns them.
   *
   * A match is stored as vector<int>, which is a mapping to OperatorDefs
   * in Graph. The order matters.
   */
  std::vector<std::vector<int>> PatternMatch(const transform::Graph& graph);

  /**
   * Applies the replace rule onto each of the matches found.
   */
  void ReplacePattern(
      const std::vector<std::vector<int>>& matches,
      transform::Graph* graph);

 protected:
  /**
   * The PatternRule essentially answers:
   * Given the current subgraph (ordered), should we append the new node at idx?
   */
  virtual bool PatternRule(
      const transform::Graph& g,
      const std::vector<int>& subgraph,
      int /*idx*/) {
    CAFFE_NOT_IMPLEMENTED;
  }

  /**
   * The ValidatorRule essentially answers:
   * Given a subgraph, can we accept it?
   */
  virtual bool ValidatorRule(
      const transform::Graph& g,
      const std::vector<int>& subgraph) {
    CAFFE_NOT_IMPLEMENTED;
  }

  /**
   * The ReplaceRule actually mutates the graph, and applies the transformation
   * upon the subgraph.
   */
  virtual bool ReplaceRule(
      const std::vector<int>& subgraph,
      transform::Graph* g_ptr) {
    CAFFE_NOT_IMPLEMENTED;
  }

  void SetPatternMatchType(PatternMatchType type) {
    pattern_match_type_ = type;
  }

 private:
  /**
   * A helper function for PatternMatch, which keeps track of the best subgraph
   * so far.
   */
  void PatternMatchHelper(
      const transform::Graph& graph,
      const std::vector<bool>& matched,
      std::vector<int>* subgraph_ptr,
      std::vector<int>* best_subgraph_ptr);
  /**
   * Attempts to append each neighbor to the end of the subgraph.
   */
  void TryNeighbors(
      const transform::Graph& graph,
      const std::map<int, std::vector<string>>& neighbors,
      const std::vector<bool>& matched,
      std::vector<int>* subgraph_ptr,
      std::vector<int>* best_subgraph_ptr);

  PatternMatchType pattern_match_type_ = CONNECTED_SUBGRAPH;
};

// Creates a Transform based on a key, which should be defined in registry.
CAFFE2_API unique_ptr<Transform> CreateTransform(string key);

C10_DECLARE_REGISTRY(TransformRegistry, Transform);
#define REGISTER_TRANSFORM(name, ...) \
  C10_REGISTER_CLASS(TransformRegistry, name, __VA_ARGS__)

// Create a Transform object from registry,
// and immediately apply it to a Netdef.
CAFFE2_API NetDef ApplyTransform(const string& key, const NetDef& netdef);

// Create a Transform object from registry, apply it to a NetDef.
// Will only return the transformed net if it is faster than the old net.
// This will run the init net first, will run the two nets warmup_runs times.
// Then, we will take the average time of main_runs runs, and only keep the
// transformed net if it is faster by a factor of improvement_threshold.
CAFFE2_API NetDef ApplyTransformIfFaster(
    const string& key,
    const NetDef& netdef,
    const NetDef& init_netdef,
    const int warmup_runs,
    const int main_runs,
    const double improvement_threshold);

} // namespace
