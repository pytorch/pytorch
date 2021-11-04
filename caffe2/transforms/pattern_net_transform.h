#pragma once

#include "caffe2/core/common.h"
#include "caffe2/core/transform.h"
#include "caffe2/proto/caffe2_pb.h"
#include "caffe2/utils/proto_utils.h"

namespace caffe2 {

/**
 * PatternNetTransform allows you to create transforms using a simple
 * interface.
 *
 * Simply provide a Pattern NetDef and a Replace NetDef,
 * and this Transform will find subgraphs which fit the pattern net,
 * and replace it with the replace net.
 */
class TORCH_API PatternNetTransform : public Transform {
 public:
  PatternNetTransform(const NetDef& pattern_net, const NetDef& replace_net)
      : p_(transform::Graph(pattern_net)), r_(transform::Graph(replace_net)) {
    // external input and output must match!
    CAFFE_ENFORCE(
        p_.external_input() == r_.external_input(),
        "External inputs do not match!");
    CAFFE_ENFORCE(
        p_.external_output() == r_.external_output(),
        "External outputs do not match!");
    ordered_ops_ = GetPatternTraversalOrder(p_);
    inverse_ops_.resize(ordered_ops_.size());
    for (size_t i = 0; i < ordered_ops_.size(); i++) {
      inverse_ops_[ordered_ops_[i]] = i;
    }
  }

  void EnableArgumentMatching() {
    argument_match_ = true;
  }

  void DisableArgumentMatching() {
    argument_match_ = false;
  }

 protected:
  /**
   * We want to the final result of subgraph to match the PatternNet in the
   * order of ordered_ops, operator by operator.
   *
   * [[[ ie. g.node(subgraph[i]) should match p.node(ordered_ops[i]) ]]]
   *
   * PatternRule for PatternNetTransform does the following:
   *
   * When trying to insert node idx into subgraph[p_idx],
   * we need to see if the edges between index and the
   * subgraph match the edges between p[ordered_ops[idx]]
   * and p[ordered_ops[0]...ordered_ops[p_idx-1]].
   */
  bool PatternRule(
      const transform::Graph& g,
      const std::vector<int>& subgraph,
      int idx) override;
  /**
   * ValidatorRule for PatternNetTransform does the following:
   *
   * Checks if the size of subgraph and p.size() are the same. That's it!
   */
  bool ValidatorRule(
      const transform::Graph& g,
      const std::vector<int>& subgraph) override;
  /**
   * ReplaceRule for PatternNet Transform does the following:
   *
   * 1) Figure out edge renamings for edges going into/out of the subgraph.
   * That is, for each blob in the pattern graph, what is it called in the
   * matched subgraph?
   *
   * 2) Remove the matched subgraph.
   *
   * 3) Append the replace graph's operators to the graph's operators, and use
   *    the renamings to rename the blob names.
   *
   * 4) Create all the children/parent relationships within the replaced graph,
   *    and stitch together the inputs and outputs into the rest of the graph,
   *    matching the removed subgraph.
   */
  bool ReplaceRule(const std::vector<int>& subgraph, transform::Graph* g_ptr)
      override;

 private:
  /**
   * This returns a permutation of the Pattern Net's operators.
   * The permutation satisfies this property:
   *    - For any index i, order(i) is a neighbor of some node from
   *      {order(1), ..., order(i-1)}.
   *
   * Why is this important? Consider the following case:
   * PatternNet: 0 ---> 2 <--- 1
   *
   * When we have matched onto [0], and trying to add [1] to our subgraph,
   * we cannot, since PatternMatch only considers neighbors of the current
   * subgraph as a candidate next node.
   *
   * Therefore, we must present the subgraph in an order such that each node is
   * a neighbor of its prefix subgraph. One ordering for the above example is
   * [0, 2, 1].
   */
  std::vector<int> GetPatternTraversalOrder(const transform::Graph& g);

  // Graph of Pattern NetDef
  transform::Graph p_;

  // The Traversal Order of the Pattern Net's Operators
  // This is a permutation of the numbers from {0, ..., p.size()-1}
  std::vector<int> ordered_ops_;

  // The Inverse of the Traversal Order of the Pattern Net's Operators
  // That is, inverse_ops[ordered_ops[i]] == i is always true.
  std::vector<int> inverse_ops_;

  // Graph of Replace NetDef
  transform::Graph r_;

  // This flag determines if the transform will match operator arguments.
  bool argument_match_ = false;

  const string TransformBlobWrapper(const string& blob_name) {
    return "transform/" + blob_name + "_" + c10::to_string(ssa_id_);
  }

  int ssa_id_ = 0;
};

} // namespace caffe2
