#include "caffe2/core/transform.h"

#include "caffe2/core/common.h"
#include "caffe2/core/logging.h"
#include "caffe2/core/net.h"
#include "caffe2/core/timer.h"
#include "caffe2/proto/caffe2_pb.h"

namespace caffe2 {

using transform::Graph;

C10_DEFINE_REGISTRY(TransformRegistry, Transform);

std::vector<std::vector<int>> Transform::PatternMatch(const Graph& graph) {
  // checks if the node at index i is matched already or not
  std::vector<bool> matched(graph.size(), false);

  // stores matches, which are ordered subgraphs of G
  std::vector<std::vector<int>> matches;

  // Consider every possible node as the starting point.
  for (int idx = 0; idx < (int)graph.size(); ++idx) {
    // The current working subgraph. We will try to add new nodes to this,
    // when invoking the PatternRule.
    std::vector<int> subgraph;

    // The largest "validated" subgraph found so far.
    // This will be mutated by PatternMatchHelper.
    std::vector<int> best_subgraph;

    // Only begin to match if the start node is accepted.
    if (!matched.at(idx) && PatternRule(graph, subgraph, idx)) {
      subgraph.push_back(idx);
      PatternMatchHelper(graph, matched, &subgraph, &best_subgraph);
      subgraph.pop_back();
    }
    if (best_subgraph.size() > 0) { // match found
      matches.push_back(best_subgraph);
      for (const auto& x : best_subgraph) {
        matched[x] = true;
      }
    }
  }
  return matches;
}

void Transform::TryNeighbors(
    const Graph& graph,
    const std::map<int, std::vector<string>>& neighbors,
    const std::vector<bool>& matched,
    std::vector<int>* subgraph_ptr,
    std::vector<int>* best_subgraph_ptr) {
  auto& subgraph = *subgraph_ptr;
  for (const auto& edge : neighbors) {
    int j = edge.first;
    if (std::find(subgraph.begin(), subgraph.end(), j) == subgraph.end()) {
      if (!matched.at(j) && PatternRule(graph, subgraph, j)) {
        subgraph.push_back(j);
        PatternMatchHelper(graph, matched, subgraph_ptr, best_subgraph_ptr);
        subgraph.pop_back();
      }
    }
  }
}

void Transform::PatternMatchHelper(
    const Graph& graph,
    const std::vector<bool>& matched,
    std::vector<int>* subgraph_ptr,
    std::vector<int>* best_subgraph_ptr) {
  CHECK(subgraph_ptr);
  auto& subgraph = *subgraph_ptr;
  CHECK(best_subgraph_ptr);
  auto& best_subgraph = *best_subgraph_ptr;

  // If the current subgraph is valid, and the largest we've seen so far,
  // make it the best_subgraph.
  if (ValidatorRule(graph, subgraph) &&
      subgraph.size() > best_subgraph.size()) {
    best_subgraph = subgraph;
  }

  size_t size_before = subgraph.size();

  if (pattern_match_type_ == CONNECTED_SUBGRAPH) {
    // Connected Component Order Pattern Matching
    // We want to match subgraphs which are connected ConnectedComponents

    // Try adding each parent and child of every node in the subgraph,
    // and see if we can accept it.
    for (size_t i = 0; i < subgraph.size(); i++) {
      int x = subgraph[i];
      TryNeighbors(
          graph,
          graph.node(x).children,
          matched,
          subgraph_ptr,
          best_subgraph_ptr);
      CAFFE_ENFORCE(
          size_before == subgraph.size(),
          "Subgraph size should not change after returning from recursive call.");
      TryNeighbors(
          graph,
          graph.node(x).parents,
          matched,
          subgraph_ptr,
          best_subgraph_ptr);
      CAFFE_ENFORCE(
          size_before == subgraph.size(),
          "Subgraph size should not change after returning from recursive call.");
    }
  } else if (pattern_match_type_ == SORTED_WRT_EXECUTION_ORDER) {
    // Sorted Execution Order Pattern matching
    // We want to be able to match subgraphs in sorted execution order

    // We can safely assume our subgraph is already sorted.
    // This means, we only need to consider nodes that come after the LAST
    // node in our current subgraph.
    // Thus, we simply iterate over the nodes that come AFTER the last node of
    // our current subgraph.
    size_t start_idx = 0;
    if (subgraph.size() > 0) {
      start_idx = subgraph.back() + 1;
    }
    for (size_t i = start_idx; i < graph.size(); i++) {
      if (!matched.at(i) && PatternRule(graph, subgraph, i)) {
        subgraph.push_back(i);
        PatternMatchHelper(graph, matched, subgraph_ptr, best_subgraph_ptr);
        subgraph.pop_back();
      }
    }
  } else if (pattern_match_type_ == GENERAL) {
    // General Pattern matching
    // We want to be able to match any ordered subgraph

    // For every current subgraph, we consider all nodes to be
    // the next candidate node, as long as it isn't already matched.
    for (size_t i = 0; i < graph.size(); i++) {
      if (std::find(subgraph.begin(), subgraph.end(), i) == subgraph.end()) {
        // Then we try appending it to the subgraph.
        if (!matched.at(i) && PatternRule(graph, subgraph, i)) {
          subgraph.push_back(i);
          PatternMatchHelper(graph, matched, subgraph_ptr, best_subgraph_ptr);
          subgraph.pop_back();
        }
      }
    }
  } else {
    CAFFE_NOT_IMPLEMENTED;
  }
}

void Transform::ReplacePattern(
    const std::vector<vector<int>>& matches,
    Graph* graph) {
  for (const auto& match : matches) {
    // Make sure each matched node is still active (not overwritten)
    bool is_match_active = true;
    for (int idx : match) {
      if (!graph->is_node_active(idx)) {
        is_match_active = false;
      }
    }

    // Simply try to apply the replace rule upon every match.
    if (is_match_active && !ReplaceRule(match, graph)) {
      CAFFE_THROW("Replace failed!");
    }
  }
}

// The simple interface - performs the transformation upon a NetDef, and returns
// the result.
NetDef Transform::ApplyTo(const NetDef& orig_net) {
  Graph g(orig_net);
  const auto matches = PatternMatch(g);
  ReplacePattern(matches, &g);
  return g.GetNetDef();
}

// Create a Transform object
unique_ptr<Transform> CreateTransform(string key) {
  auto t = TransformRegistry()->Create(key);
  CAFFE_ENFORCE(t != nullptr, "Transform not found in registry: ", key);
  return t;
}

// Create a Transform object from registry,
// and immediately apply it to a Netdef.
NetDef ApplyTransform(const string& key, const NetDef& netdef) {
  auto t = CreateTransform(key);
  return t->ApplyTo(netdef);
}

static double average_net_run_duration(
    const NetDef& netdef,
    const NetDef& init_netdef,
    const int warmup_runs,
    const int main_runs) {
  Workspace ws;
  if (init_netdef.op_size() > 0) {
    std::unique_ptr<NetBase> init_net(CreateNet(init_netdef, &ws));
    CHECK(init_net);
    CAFFE_ENFORCE(init_net->Run(), "Init run has failed!");
  } else {
    // If a proper init_net is not provided, then this is the best we can do.
    // NOLINTNEXTLINE(performance-for-range-copy)
    for (auto inp : netdef.external_input()) {
      ws.CreateBlob(inp);
    }
  }
  std::unique_ptr<NetBase> net(CreateNet(netdef, &ws));
  CHECK(net);
  CAFFE_ENFORCE(
      warmup_runs >= 0,
      "Number of warm up runs should be non negative, provided ",
      warmup_runs,
      ".");

  for (int i = 0; i < warmup_runs; i++) {
    CAFFE_ENFORCE(net->Run(), "Warmup run ", i, " has failed.");
  }

  CAFFE_ENFORCE(
      main_runs > 0,
      "Number of main runs should be positive, provided ",
      main_runs,
      ".");
  Timer timer;
  for (int i = 0; i < main_runs; i++) {
    CAFFE_ENFORCE(net->Run(), "Main run ", i, " has failed.");
  }
  return timer.MilliSeconds();
}

// Create a Transform object from registry, apply it to a NetDef.
// Will only return the transformed net if it is faster than the old net.
// This will run the init net first, will run the two nets warmup_runs times.
// Then, we will take the average time of main_runs runs, and only keep the
// transformed net if it is faster by a factor of improvement_threshold.
NetDef ApplyTransformIfFaster(
    const string& key,
    const NetDef& netdef,
    const NetDef& init_netdef,
    const int warmup_runs,
    const int main_runs,
    const double improvement_threshold) {
  NetDef transformed_netdef = ApplyTransform(key, netdef);
  double original_net_time =
      average_net_run_duration(netdef, init_netdef, warmup_runs, main_runs);
  double new_net_time = average_net_run_duration(
      transformed_netdef, init_netdef, warmup_runs, main_runs);
  if (original_net_time > improvement_threshold * new_net_time) {
    return transformed_netdef;
  }
  return netdef;
}

} // namespace Caffe2
