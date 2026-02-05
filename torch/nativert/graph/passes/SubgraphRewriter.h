#pragma once

#include <c10/util/FbcodeMaps.h>
#include <torch/nativert/graph/Graph.h>

#include <utility>

namespace torch::nativert {

/*
 * node_map: A map from nodes in the pattern to nodes in the actual graph.
 * value_map : A map between values in the pattern to values in the actual
 * graph.
 * dummy_input_to_attribute_map: A map between the actual dummy input values to
 * constant attributes in the actual graph that should replace the dummy nodes
 */
struct Match {
  std::unordered_map<const Node*, Node*> node_map;
  std::unordered_map<const Value*, Value*> value_map;
  std::unordered_map<Value*, const Constant*>
      dummy_input_to_attribute_map; // For constant attrs matching graph inputs
};

using MatchFilter = std::function<
    bool(const Match&, const c10::FastMap<std::string, const Value*>&)>;

inline std::ostream& operator<<(std::ostream& out, const Match& match) {
  out << "\nNode mapping:\n";
  for (const auto& kv : match.node_map) {
    const Node* patternNode = kv.first;
    Node* targetNode = kv.second;
    out << "  Pattern Node: " << *patternNode
        << " -> Target Node: " << *targetNode << "\n";
  }

  out << "Value mapping:\n";
  for (const auto& kv : match.value_map) {
    const Value* patternValue = kv.first;
    Value* targetValue = kv.second;
    out << "  Pattern Value: " << *patternValue
        << " -> Target Value: " << *targetValue << "\n";
  }

  return out;
}

/**
 * A helper class for matching a subgraph pattern within a larger graph.
 * It attempts to match a given `pattern` graph inside a target `graph`,
 * starting from a single "root" output node in the pattern graph. The
 * matching process works backward through the graph, comparing each node
 * in the pattern to corresponding nodes in the candidate graph.
 *
 * Note: This implementation currently only supports deterministic matching
 * for patterns with one output node. It also only matches nodes connecting to
 * output nodes
 *
 * Constraints for Patterns with Multiple Output Nodes:
 * To avoid an exponential increase in the search space, this implementation
 * starts searching from the first output node as an anchor as an heuristic. It
 * assumes that all other output nodes in the pattern are interconnected through
 * the graph from this anchor node, allowing the matcher to traverse from the
 * anchor to other outputs.
 *
 * Important: The order of output nodes in the pattern matters. For example:
 *
 *   graph(%x):
 *       %a = a.aaa(input=%x)
 *       %b = b.bbb(input=%a)
 *       return (%a, %b)
 *
 * If the search starts from %a, it will not explore the portion of the graph
 * connected to %b. However, if the order is switched:
 *
 *   graph(%x):
 *       %a = a.aaa(input=%x)
 *       %b = b.bbb(input=%a)
 *       return (%b, %a)
 *
 * The search will start from %b and successfully explore both %b and %a.
 */
class SubgraphMatcher {
 public:
  explicit SubgraphMatcher(const Graph* pattern);

  /// Attempt to match the pattern at a given node in the target graph.
  /// If successful, returns a Match, otherwise std::nullopt.
  std::optional<Match> match(Node* target_node);

  std::vector<Match> matchAll(Graph* target_graph);

 private:
  const Graph* pattern_;
  const Node* pattern_root_;

  /**
   * Finds the root output node of a Graph g to start a match from
   * Note that graphs with multiple output nodes, this will pick the first
   * output node in the order provided.
   **/
  const Node* findRootNode(const Graph* g);

  /**
   * Tries to match nodes in the pattern_ graph with the target graph, starting
   * from pattern_node and target_node. Nodes are considered to match if they
   * have the same target type, and all input and output values to the nodes
   * match. Matching nodes are stored to `match`
   **/
  bool tryMatchNode(const Node* pattern_node, Node* target_node, Match& match);

  /**
   * Match inputs of pattern_node w/ target_node. Store matching values to
   *`match`
   **/
  bool tryMatchNodeInputs(
      const Node* pattern_node,
      Node* target_node,
      Match& match);

  /**
   * Tries to match values in the pattern_ graph with the target graph, starting
   * from pval and tval. Matching values are stored to `match`.
   **/
  bool tryMatchValue(const Value* pval, Value* tval, Match& match);

  /**
   * Returns true of val is an output of its graph, and false otherwise
   **/
  bool isOutputValue(const Value* val);
};

struct RewriteRule {
  std::string pattern;
  std::string replacement;
};

/**
 * Rewrite subgraphs in a given graph.
 * TODO: Write more detailed documentation
 **/
class SubgraphRewriter {
 public:
  SubgraphRewriter(std::string name) : name_(std::move(name)) {}

  /**
   * Registers the rewrite pattern.
   * @param patternA The subgraph str to match.
   * @param patternB The subgraph str to replace with.
   */
  void registerRewritePattern(
      const std::string& pattern,
      const std::string& replacement);

  /**
   * Runs the subgraph rewrite process on a graph.
   * @param graph The graph on which the rewrite is applied.
   * @param pattern The subgraph to match.
   * @param replacement The subgraph to replace with.
   * @param filters A list of filters to apply to the match. If any filter
   * predicate returns true, the match will not be considered.
   */
  bool /* mutated? */ runForPattern(
      Graph* graph,
      const Graph& pattern,
      const Graph& replacement,
      const std::vector<MatchFilter>& filters);

  bool /* mutated? */ run(
      Graph* graph,
      const MatchFilter& filter =
          [](const Match&, const c10::FastMap<std::string, const Value*>&) {
            return true;
          }) {
    return run(graph, std::vector<MatchFilter>({filter}));
  }

  bool /* mutated? */ run(
      Graph* graph,
      const std::vector<MatchFilter>& filters);

 private:
  std::string name_;
  std::vector<RewriteRule> patterns_; // The subgraph pattern to match
  std::unordered_set<Node*> replacedNodes_;
  std::vector<Value*> valuesToRewrite_;
  std::unordered_map<const Value*, Value*> valueRewrites_;

  // Helper methods
  bool overlapsWithUsedNodes(
      const Match& match,
      const std::unordered_set<Node*>& replacedNodes);
  void rewriteMatch(
      Graph* graph,
      const Match& match,
      const Graph& pattern,
      const Graph& replacement);

  c10::FastMap<std::string, const Value*> getVmap(const Graph& pattern);
};
} // namespace torch::nativert
