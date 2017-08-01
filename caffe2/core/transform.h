#pragma once

#include "caffe2/core/common.h"
#include "caffe2/core/graph.h"
#include "caffe2/proto/caffe2.pb.h"
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
class Transform {
 public:
  Transform() {}

  /**
   * Apply a Transform onto a NetDef.
   * Returns the transformed NetDef.
   */
  NetDef ApplyTo(const NetDef& orig_net_def);

  virtual ~Transform() {}

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

 private:
  /**
   * A helper function for PatternMatch, which keeps track of the best subgraph
   * so far.
   */
  void PatternMatchHelper(
      const transform::Graph& graph,
      std::vector<int>* subgraph_ptr,
      std::vector<int>* best_subgraph_ptr);
  /**
   * Attempts to append each neighbor to the end of the subgraph.
   */
  void TryNeighbors(
      const transform::Graph& graph,
      const std::map<int, std::vector<string>>& neighbors,
      std::vector<int>* subgraph_ptr,
      std::vector<int>* best_subgraph_ptr);
};

// Creates a Transform based on a key, which should be defined in registry.
unique_ptr<Transform> CreateTransform(string key);

CAFFE_DECLARE_REGISTRY(TransformRegistry, Transform);
#define REGISTER_TRANSFORM(name, ...) \
  CAFFE_REGISTER_CLASS(TransformRegistry, name, __VA_ARGS__)

} // namespace
