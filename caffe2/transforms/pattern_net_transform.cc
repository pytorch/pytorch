#include "caffe2/transforms/pattern_net_transform.h"

#include "caffe2/core/common.h"
#include "caffe2/core/logging.h"
#include "caffe2/core/net.h"
#include "caffe2/proto/caffe2_pb.h"

#include <c10/util/irange.h>

namespace caffe2 {

// First, single source traverse through the netdef.
// This ensures all newly ordered are reachable from their prefix subset
// Outputs a permutation of the operators.
std::vector<int> PatternNetTransform::GetPatternTraversalOrder(
    const transform::Graph& graph) {
  std::vector<bool> visited(graph.size(), false);
  std::vector<int> ordered_ops;
  std::queue<int> q;
  if (graph.size() > 0) {
    q.push(0);
    ordered_ops.push_back(0);
    visited[0] = true;
  }
  while (!q.empty()) {
    int idx = q.front();
    q.pop();
    for (const auto& edge : graph.node(idx).children) {
      int x = edge.first;
      if (!visited[x]) {
        q.push(x);
        ordered_ops.push_back(x);
        visited[x] = true;
      }
    }
    for (const auto& edge : graph.node(idx).parents) {
      int x = edge.first;
      if (!visited[x]) {
        q.push(x);
        ordered_ops.push_back(x);
        visited[x] = true;
      }
    }
  }
  CAFFE_ENFORCE(
      ordered_ops.size() == graph.size(), "Pattern graph must be connected.");
  return ordered_ops;
}

bool compare_ops(
    const OperatorDef& p_op,
    const OperatorDef& g_op,
    bool arg_match) {
  // must specify a type for pattern operators
  CAFFE_ENFORCE(
      p_op.has_type(), "Types must be specified for all pattern operators.");
  if (!MatchStrings(p_op.type(), g_op.type())) {
    return false;
  }
  // ensure number of inputs are the same
  if (p_op.input().size() != g_op.input().size()) {
    return false;
  }

  // ensure number of outputs are the same
  if (p_op.output().size() != g_op.output().size()) {
    return false;
  }

  if (p_op.has_device_option()) {
    if (!g_op.has_device_option() ||
        p_op.device_option().device_type() !=
            g_op.device_option().device_type()) {
      return false;
    }
  }

  // make sure engine is the same (if specified in pattern)
  if (p_op.has_engine() && !MatchStrings(p_op.engine(), g_op.engine())) {
    return false;
  }
  // If argument_match is specified, make sure those are the same.
  if (arg_match) {
    if (!MatchArguments(p_op, g_op)) {
      return false;
    }
  }
  return true;
}

// g.node(subgraph[i]) should match p_.node(ordered_ops_[i])
// g.node(g_idx) should match p_.node(p_idx)
bool PatternNetTransform::PatternRule(
    const transform::Graph& g,
    const std::vector<int>& subgraph,
    int g_idx) {
  if (subgraph.size() >= ordered_ops_.size()) {
    return false;
  }
  int p_idx = ordered_ops_[subgraph.size()];

  if (!compare_ops(p_.node(p_idx).op, g.node(g_idx).op, argument_match_)) {
    return false;
  }

  // Let's say ordered_ops_ is [0, 2, 1], with 0 -> 2 being an edge
  // When we try to match onto the second element, let's say our
  // subgraph so far is [4], with it trying to become [4, 5].
  // Then, we need to show that since 0 -> 2 is an edge is ordered_ops_,
  // 4 must be a direct parent of 5 in the subgraph
  // (the indices must match).
  // Similarly, assume there is an edge from 1 -> 2 in p_.
  // When trying to match [4, 5] to [4, 5, 7], we must verify that
  // there exists an edge from 7 -> 5 in G.
  for (const auto& edge : p_.node(p_idx).parents) {
    int parent = edge.first;
    // g_idx doesn't have parent in subgraph that p_[p_idx] has
    // inverse_ops_ gets the index of a p_idx inside of ordered_ops_.
    if (inverse_ops_[parent] < subgraph.size() &&
        g.node(g_idx).parents.count(subgraph[inverse_ops_[parent]]) == 0) {
      return false;
    }
  }

  for (const auto& edge : p_.node(p_idx).children) {
    int child = edge.first;
    if (inverse_ops_[child] < subgraph.size() &&
        g.node(g_idx).children.count(subgraph[inverse_ops_[child]]) == 0) {
      return false;
    }
  }
  return true;
}

bool PatternNetTransform::ValidatorRule(
    const transform::Graph& /*g*/,
    const std::vector<int>& subgraph) {
  // Due to strict PatternRule, it suffices to simply check for size
  return subgraph.size() == p_.size();
}

bool PatternNetTransform::ReplaceRule(
    const std::vector<int>& match,
    transform::Graph* g_ptr) {
  CHECK(g_ptr);
  auto& g = *g_ptr;

  ssa_id_++;

  // Map of PatternNet blob name to Matched blob name.
  // Figures out how to rename the pattern_net to make the replacement fit.
  std::unordered_map<string, string> external_renaming;

  // Figure out blob renamings
  for (const auto i : c10::irange(match.size())) {
    int g_idx = match[i];
    int p_idx = ordered_ops_[i];
    for (int j = 0; j < p_.node(p_idx).op.input().size(); j++) {
      string p_blob = p_.node(p_idx).op.input(j);
      string g_blob = g.node(g_idx).op.input(j);
      if (p_.external_input().count(p_blob)) {
        external_renaming[p_blob] = g_blob;
      }
    }
    for (int j = 0; j < p_.node(p_idx).op.output().size(); j++) {
      string p_blob = p_.node(p_idx).op.output(j);
      string g_blob = g.node(g_idx).op.output(j);
      if (p_.external_output().count(p_blob)) {
        external_renaming[p_blob] = g_blob;
      }
    }
  }

  auto input_list = g.GetSubgraphInput(match);
  auto output_list = g.GetSubgraphOutput(match);

  g.DeactivateSubgraph(match);

  int offset = g.size();

  g.resize_nodes(offset + r_.size());

  // Append all the new operators.
  for (const auto i : c10::irange(r_.size())) {
    int new_node_idx = offset + i;

    OperatorDef new_op = r_.node(i).op;

    new_op.clear_input();
    new_op.clear_output();
    // Stitch Input from external graph into replaced subgraph
    for (const auto& blob : r_.node(i).op.input()) {
      if (external_renaming.count(blob)) {
        string new_blob = external_renaming[blob];
        new_op.add_input(new_blob);

        // binary searches for new_blob amongst input list.
        auto it = std::lower_bound(
            input_list.begin(), input_list.end(), std::make_pair(new_blob, -1));

        // if the input came from the graph (instead of G's external input)
        for (; it < input_list.end() && it->first == new_blob; it++) {
          int parent = it->second;
          g.node(parent).children[new_node_idx].push_back(new_blob);
          g.node(new_node_idx).parents[parent].push_back(new_blob);
        }
      } else {
        new_op.add_input(TransformBlobWrapper(blob));
      }
    }
    // Stitch Output from replaced subgraph to external graph.
    for (const auto& blob : r_.node(i).op.output()) {
      if (external_renaming.count(blob)) {
        string new_blob = external_renaming[blob];
        new_op.add_output(new_blob);

        // binary searches for new_blob amongst input list.
        auto it = std::lower_bound(
            output_list.begin(),
            output_list.end(),
            std::make_pair(new_blob, -1));

        // if the output goes to the graph (instead of G's external output)
        for (; it < output_list.end() && it->first == new_blob; it++) {
          int child = it->second;
          g.node(child).parents[new_node_idx].push_back(new_blob);
          g.node(new_node_idx).children[child].push_back(new_blob);
        }
      } else {
        new_op.add_output(TransformBlobWrapper(blob));
      }
    }

    // Connect all internal edges within replace graph
    for (const auto& edge : r_.node(i).parents) {
      int parent = edge.first;
      int new_node_parent = offset + parent;
      const auto& blobs = edge.second;
      for (const string& blob : blobs) {
        g.node(new_node_idx)
            .parents[new_node_parent]
            .push_back(TransformBlobWrapper(blob));
      }
    }

    for (const auto& edge : r_.node(i).children) {
      int child = edge.first;
      int new_node_child = offset + child;
      const auto& blobs = edge.second;
      for (const string& blob : blobs) {
        g.node(offset + i)
            .children[new_node_child]
            .push_back(TransformBlobWrapper(blob));
      }
    }

    g.node(new_node_idx).op = new_op;
    g.node(new_node_idx).active = true;
  }
  return true;
}

} // namespace Caffe2
