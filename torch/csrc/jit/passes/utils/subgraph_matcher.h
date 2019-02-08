#pragma once

#include <torch/csrc/jit/ir.h>
#include <torch/csrc/jit/passes/alias_analysis.h>

#include <iostream>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace torch {
namespace jit {

/** \brief Instantiate a Graph and use a MatchIterator to iteratively
  * get matches on the underlying graph.
  *
  * The returned `Match`s can
  * be used as maps from the passed in match graph to the graph the
  * match was run on.
  *
  *   for (auto m : MatchIterator(match_graph, graph)) {
  *     AT_ASSERT(m[m0.value()] == a.value());
  *   }
  *
  * Optionally, pass in a std::ostream* to get debug information.
  * e.g. `MatchIterator(match_graph, graph, &std::cout)`.
  */
struct TORCH_API MatchIterator {
  MatchIterator(
      const Graph& match_graph,
      const Graph& graph,
      std::ostream* log_stream = nullptr)
      : match_graph_(match_graph),
        // Arbitrarily choose anchor node
        anchor_(*(match_graph.nodes().begin())),
        graph_(graph),
	aliasDb_(std::move(std::make_shared<AliasDb>(graph.copy()))),
        finished_(false),
        log_stream_(log_stream) {
    AliasDb matchAliasDb(match_graph_.copy());
    for (const auto& n : match_graph_.nodes()) {
      AT_CHECK(!matchAliasDb.hasWriters(n), "Match graph cannot contain aliases");
    }
    // Remove the "Return" and "Param" nodes from being counted
    match_size_ = match_graph_.size();
    if (match_graph_.param_node()) {
      match_size_--;
    }
    if (match_graph_.return_node()) {
      match_size_--;
    }
  }

  // Match is a two typed map that makes it convenient to deal with
  // the matches by overloading the subscript [] operator to be
  // const correct and handle both types (Node*, Value*)
  struct Match {
    std::unordered_map<const Node*, const Node*> node_match;
    std::unordered_map<const Value*, const Value*> value_match;

    size_t size() const;
    void clear();

    void set(const Node* const& m, const Node* n);
    void set(const Value* m, const Value* v);

    const Node* get(const Node* const& m) const;
    const Value* get(const Value* const& m) const;
    const Node* operator[](const Node* const& m) const;
    const Value* operator[](const Value* const& m) const;
  };

  MatchIterator& begin();
  MatchIterator& end();

  MatchIterator& operator++();
  MatchIterator& operator++(int);

  Match operator*();
  bool operator!=(MatchIterator& rhs);

 private:
  using NodeFrontier = std::vector<std::pair<const Node*, const Node*>>;
  using ValueFrontier = std::vector<std::pair<const Value*, const Value*>>;

  bool matchNodes(const Node* n1, const Node* n2) {
    if (aliasDb_->hasWriters(n1) || aliasDb_->hasWriters(n2)) {
      return false;
    }
    auto match = n1->kind() == n2->kind();
    std::stringstream os;
    if (!match) {
      os << "[mismatch] node types don't match: "
         << n1->kind().toDisplayString() << "(" << n1 << ")"
         << " vs " << n2->kind().toDisplayString() << "(" << n2 << ")";
    } else {
      os << "[match] node of type " << n1->kind().toDisplayString() << " " << n1
         << " - " << n2;
    }
    log(os.str());
    return match;
  }
  bool matchValues(const Value* v1, const Value* v2) {
    std::stringstream os;
    os << "[match] value " << v1 << " - " << v2;
    log(os.str());
    return true;
  }

  // Matches values that are either inputs or outputs
  // of nodes in the the node_frontier.
  // If all values are matched, updates the
  // match with the values, stores found values
  // in value_frontier and return true. Otherwise
  // returns false.
  bool matchNodeFrontier(
      NodeFrontier* node_frontier,
      MatchIterator::Match* match);

  bool matchValueFrontier(
      const ValueFrontier& value_frontier,
      Match* match,
      NodeFrontier* node_frontier);

  // Returns a new match.
  //
  // NB: The last match may be "unfilled" (either empty or smaller
  // than the requested match graph) which should be
  // used as the termination condition for matches.
  Match getMatch();

  // Updates the stateful match_ and finished_ variables.
  void updateCurrentMatch();

  // Find a new node to act as the anchor for the match.
  const Node* findNewNode();

  // Initializes passed in elements, calls findNewNode.
  // Returns true if unable to find a new node.
  bool tryResetMatch(Match* m, NodeFrontier* nf);

  // Logs to user provided "log_stream" which is defaulted
  // to nullptr.
  inline void log(std::string s) {
    if (log_stream_) {
      (*log_stream_) << s << "\n";
    }
  }

 private:
  const Graph& match_graph_;
  const Node* anchor_;
  const Graph& graph_;
  const std::shared_ptr<AliasDb> aliasDb_;
  bool finished_;
  std::ostream* log_stream_ = nullptr;

  std::unordered_set<const Node*> tried_nodes_;
  Match match_;
  const Block* current_block_;
  size_t match_size_ = 0;
};

} // namespace jit
} // namespace torch
