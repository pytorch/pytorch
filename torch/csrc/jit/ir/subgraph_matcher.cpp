#include <c10/util/irange.h>
#include <torch/csrc/jit/ir/subgraph_matcher.h>
#include <torch/csrc/jit/jit_log.h>

#include <regex>
#include <stack>

namespace torch::jit {
namespace {

/**
 * \brief A class implementing an API for comparing subgraphs.
 */
class SubgraphMatcher {
 public:
  explicit SubgraphMatcher(const Graph& pattern) : pattern_(pattern) {}

  /**
   * \brief Compare matchGraph with the part of the graph denoted by a node \p
   * ANCHOR.
   *
   * The anchor node would be compared against the deepest node in the
   * match-graph. A node is considered matching if its number of inputs/outputs
   * is the same as in the corresponding matchGraph node, its type is the same,
   * and all nodes producing input-values also match.
   */
  bool matchesSubgraphFromAnchorNode(Node* anchor);

  /** \brief Return match map for nodes. */
  std::unordered_map<const Node*, Node*> nodes_map() const {
    return nodes_map_;
  }

  /** \brief Return match map for values. */
  std::unordered_map<const Value*, Value*> values_map() const {
    return values_map_;
  }

 private:
  bool matchValues(const Value* v1, Value* v2);
  bool matchNodes(const Node* n1, Node* n2);
  bool matchAttributes(const Node* n1, Node* n2);

  static bool isInput(const Value* v);
  static bool isOutput(const Value* v);

  std::unordered_map<const Node*, Node*> nodes_map_;
  std::unordered_map<const Value*, Value*> values_map_;

  const Graph& pattern_;
  const Node* anchor_ = nullptr;
};

/**
 * \brief A function to verify that \p PATTERN is valid. Concrete requirements
 * for validity can be found in subgraph_matcher.h.
 */
bool patternGraphIsValid(const Graph& pattern) {
  // Verify that pattern graph has a single block.
  for (const Node* n : pattern.nodes()) {
    if (!n->blocks().empty()) {
      return false;
    }
  }

  // TODO: Verify that nodes in the pattern don't alias.
  return true;
}

bool SubgraphMatcher::isInput(const Value* v) {
  return v->node()->kind() == prim::Param;
}

bool SubgraphMatcher::isOutput(const Value* v) {
  for (const Value* output : v->owningGraph()->outputs()) {
    if (v == output) {
      return true;
    }
  }
  return false;
}

/**
 * Compare two Values. V1 is from pattern, V2 is from the actual graph.
 *
 * The values are considered matching if:
 * 1) the nodes defining them match
 * 2) they have the same number of uses, except they are entry or exit nodes.
 */
bool SubgraphMatcher::matchValues(const Value* v1, Value* v2) {
  // Check if we've already visited these values.
  if (values_map_.count(v1)) {
    if (values_map_.at(v1) != v2) {
      GRAPH_DEBUG(
          "Values %",
          v1->debugName(),
          " and %",
          v2->debugName(),
          " did not match because %",
          v1->debugName(),
          " has already been matched with %",
          values_map_.at(v1)->debugName(),
          ".\n");
      return false;
    }
    return true;
  }

  // When V2 is ANCHOR, we're comparing exiting values, and when V1->node is
  // PARAM, we're comparing entering values - in these two cases the number of
  // uses don't need to be the same.
  if (v1->uses().size() != v2->uses().size() && !isOutput(v1) && !isInput(v1)) {
    GRAPH_DEBUG(
        "Values %",
        v1->debugName(),
        " and %",
        v2->debugName(),
        " did not match because number of their uses is different.\n");
    return false;
  }

  // Add the values to the map before calling matchNodes to avoid infinite
  // recursion.
  GRAPH_DEBUG(
      "Values %", v1->debugName(), " and %", v2->debugName(), " matched.\n");
  values_map_[v1] = v2;
  return matchNodes(v1->node(), v2->node());
}

bool SubgraphMatcher::matchAttributes(const Node* n1, Node* n2) {
  if (n1->numAttributes() != n2->numAttributes()) {
    GRAPH_DEBUG("Nodes did not match in number attributes:\n", *n1, *n2);
    return false;
  }
  for (const Symbol& attr_name : n1->attributeNames()) {
    if (n1->kindOf(attr_name) != n2->kindOf(attr_name)) {
      GRAPH_DEBUG(
          "Nodes did not match because type of attribute '",
          attr_name.toQualString(),
          "' did not match:\n",
          *n1,
          *n2);
      return false;
    }
    switch (n1->kindOf(attr_name)) {
      case AttributeKind::s:
        if (!std::regex_match(n2->s(attr_name), std::regex(n1->s(attr_name)))) {
          GRAPH_DEBUG(
              "Nodes did not match because attribute '",
              attr_name.toQualString(),
              "' did not match: ",
              n1->s(attr_name),
              " != ",
              n2->s(attr_name),
              " \n",
              *n1,
              *n2);
          return false;
        }
        break;
      case AttributeKind::c:
        if (n1->c(attr_name) != n2->c(attr_name)) {
          GRAPH_DEBUG(
              "Nodes did not match because attribute '",
              attr_name.toQualString(),
              "' did not match:",
              n1->c(attr_name),
              " != ",
              n2->c(attr_name),
              " \n",
              *n1,
              *n2);
          return false;
        }
        break;
      case AttributeKind::f:
        if (n1->f(attr_name) != n2->f(attr_name)) {
          GRAPH_DEBUG(
              "Nodes did not match because attribute '",
              attr_name.toQualString(),
              "' did not match:",
              n1->f(attr_name),
              " != ",
              n2->f(attr_name),
              " \n",
              *n1,
              *n2);
          return false;
        }
        break;
      case AttributeKind::i:
        if (n1->i(attr_name) != n2->i(attr_name)) {
          GRAPH_DEBUG(
              "Nodes did not match because attribute '",
              attr_name.toQualString(),
              "' did not match:",
              n1->i(attr_name),
              " != ",
              n2->i(attr_name),
              " \n",
              *n1,
              *n2);
          return false;
        }
        break;
      default: {
        // Other attributes types not supported yet
        GRAPH_DEBUG(
            "Nodes did not match because type of attribute '",
            attr_name.toQualString(),
            "' is not supported.\n",
            *n1,
            *n2);
        return false;
      }
    }
  }
  return true;
}

static bool endsWith(const std::string& str, const std::string& suffix) {
  return str.size() >= suffix.size() &&
      0 == str.compare(str.size() - suffix.size(), suffix.size(), suffix);
}

/**
 * Compare two Nodes. N1 is from pattern, N2 is from the actual graph.
 *
 * The nodes are considered matching if:
 * 1) N1 and N2 are of the same kind.
 * 2) Number of inputs and outputs is the same.
 * 3) All input and output values match.
 *
 * A special case is when N1 is PARAM - this is considered outside the pattern,
 * so it matches everything.
 */
bool SubgraphMatcher::matchNodes(const Node* n1, Node* n2) {
  // Check if we've already visited these nodes.
  if (nodes_map_.count(n1)) {
    return nodes_map_.at(n1) == n2;
  }

  // Param node in pattern graph matches everything.
  if (n1->kind() == prim::Param) {
    GRAPH_DEBUG("Nodes matched:\n", *n1, *n2);
    return true;
  }

  // We don't allow matches to span across blocks, so check if N2 is in the same
  // block as the first (anchor) node.
  if (n2->owningBlock() != anchor_->owningBlock()) {
    GRAPH_DEBUG(
        "Nodes did not match because it is in the different block:\n",
        *n1,
        *n2);
    return false;
  }

  // Special handling for matching modules
  if (n1->kind() == Symbol::fromQualString("match::module")) {
    if (n2->kind() == prim::GetAttr) {
      if (!n1->hasAttributeS("name")) {
        GRAPH_DEBUG(
            "Nodes did not match because special node match::module does not have 'name' attribute:\n",
            *n1,
            *n2);
        return false;
      }
      auto t = n2->output()->type()->expect<ClassType>();
      auto real_typename = t->name()->qualifiedName();
      auto pattern_typename = n1->s(attr::name);
      if (!endsWith(real_typename, pattern_typename)) {
        GRAPH_DEBUG(
            "Nodes did not match because expected module type is different:\n");
        GRAPH_DEBUG("  actualtype:    ", real_typename, "\n");
        GRAPH_DEBUG("  expected type: ", pattern_typename, "\n");
        GRAPH_DEBUG("Nodes:", *n1, *n2);
        return false;
      }
    }
  } else {
    if (n1->kind() != n2->kind() ||
        n1->outputs().size() != n2->outputs().size() ||
        n1->inputs().size() != n2->inputs().size()) {
      GRAPH_DEBUG(
          "Nodes did not match in their kind or number of inputs/outputs:\n",
          *n1,
          *n2);
      return false;
    }
    if (!matchAttributes(n1, n2)) {
      return false;
    }
  }

  // Add nodes to the map before calling matchValues to avoid infinite
  // recursion.
  nodes_map_[n1] = n2;
  for (const auto i : c10::irange(n1->outputs().size())) {
    if (!matchValues(n1->outputs()[i], n2->outputs()[i])) {
      return false;
    }
  }
  for (const auto i : c10::irange(n1->inputs().size())) {
    if (!matchValues(n1->inputs()[i], n2->inputs()[i])) {
      return false;
    }
  }

  GRAPH_DEBUG("Nodes matched:\n", *n1, *n2);
  return true;
}

/**
 * Recursively try to match pattern with the actual graph starting from the
 * exiting node in the pattern and anchor node in the actual graph.
 */
bool SubgraphMatcher::matchesSubgraphFromAnchorNode(Node* anchor) {
  GRAPH_UPDATE("Starting match from a new anchor: ", *anchor);
  nodes_map_.clear();
  values_map_.clear();
  anchor_ = anchor;

  const Node* bottom_node = *(pattern_.nodes().end());
  bottom_node = bottom_node->input(0)->node();

  if (!matchNodes(bottom_node, anchor)) {
    return false;
  }

  for (const Value* output : pattern_.outputs()) {
    AT_ASSERT(values_map_.count(output));
  }

  GRAPH_UPDATE("Pattern matched!\n");
  return true;
}

} // unnamed namespace

// Main entry point for the subgraph matching.
std::vector<Match> findPatternMatches(const Graph& pattern, Graph& graph) {
  AT_ASSERT(patternGraphIsValid(pattern));
  GRAPH_DUMP("Pattern graph: ", &pattern);
  GRAPH_DUMP("Target graph: ", &graph);

  SubgraphMatcher m(pattern);
  std::vector<Match> matches;
  std::stack<Block*> blocks_to_visit;

  // Iterate over all nodes in the graph (including nodes in subblocks) trying
  // to match the pattern each node.
  blocks_to_visit.push(graph.block());
  while (!blocks_to_visit.empty()) {
    Block* block = blocks_to_visit.top();
    blocks_to_visit.pop();
    for (Node* n : block->nodes()) {
      if (m.matchesSubgraphFromAnchorNode(n)) {
        matches.push_back({n, m.nodes_map(), m.values_map()});
      }
      for (Block* subblock : n->blocks()) {
        blocks_to_visit.push(subblock);
      }
    }
  }
  return matches;
}

} // namespace torch::jit
