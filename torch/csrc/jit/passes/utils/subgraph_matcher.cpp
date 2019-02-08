#include <torch/csrc/jit/passes/utils/subgraph_matcher.h>

namespace torch {
namespace jit {

MatchIterator& MatchIterator::begin() {
  tried_nodes_.clear();
  updateCurrentMatch();
  return *this;
}

MatchIterator& MatchIterator::end() {
  return *this;
}

MatchIterator& MatchIterator::operator++() {
  updateCurrentMatch();
  return *this;
}

MatchIterator& MatchIterator::operator++(int) {
  updateCurrentMatch();
  return *this;
};

MatchIterator::Match MatchIterator::operator*() {
  return match_;
}

bool MatchIterator::operator!=(MatchIterator& rhs) {
  return !finished_;
}

const Node* MatchIterator::findNewNode() {
  for (const auto node : graph_.nodes()) {
    if (tried_nodes_.count(node) || !matchNodes(anchor_, node)) {
      continue;
    }
    tried_nodes_.insert(node);
    current_block_ = node->owningBlock();
    return node;
  }
  return nullptr;
}

// Macro to help with match requirements
#define REQ(cond)                         \
  if (!(cond)) {                          \
    log("[mismatch] cond failed " #cond); \
    return false;                         \
  } else {                                \
    log("[match] cond " #cond);           \
  }

bool MatchIterator::matchNodeFrontier(
    NodeFrontier* node_frontier,
    MatchIterator::Match* match) {
  ValueFrontier value_frontier = {};
  if (!node_frontier->size()) {
    return true;
  }
  for (const auto node_pair : *node_frontier) {
    auto match_inputs = node_pair.first->inputs();
    auto inputs = node_pair.second->inputs();
    REQ(inputs.size() == match_inputs.size());

    for (auto i = 0; i < inputs.size(); ++i) {
      auto match_input = match_inputs[i];
      auto input = inputs[i];
      if (match->get(match_input) == input) {
        std::stringstream ss;
        ss << "[info] already matched input value " << match_input << " to "
           << input;
        log(ss.str());
        continue;
      }
      if (input->node()) {
        REQ(input->node()->owningBlock() == current_block_);
      }
      REQ(matchValues(match_input, input));
      match->set(match_input, input);
      if (match_input->node() && input->node()) {
        value_frontier.emplace_back(std::make_pair(match_input, input));
      }
    }

    auto match_outputs = node_pair.first->outputs();
    auto outputs = node_pair.second->outputs();
    REQ(outputs.size() == match_outputs.size());

    for (auto i = 0; i < outputs.size(); ++i) {
      auto match_output = match_outputs[i];
      auto output = outputs[i];
      if (match->get(match_output) == output) {
        std::stringstream ss;
        ss << "[info] already matched output value " << match_output << " to "
           << output;
        log(ss.str());
        continue;
      }
      REQ(matchValues(match_output, output));
      match->set(match_output, output);
      if (match_output->node() && output->node()) {
        value_frontier.emplace_back(std::make_pair(match_output, output));
      }
    }
  }

  return matchValueFrontier(value_frontier, match, node_frontier);
}

bool MatchIterator::matchValueFrontier(
    const ValueFrontier& value_frontier,
    MatchIterator::Match* match,
    NodeFrontier* node_frontier) {
  node_frontier->clear();
  if (!value_frontier.size()) {
    return true;
  }
  for (const auto& value_pair : value_frontier) {
    auto match_node = value_pair.first->node();
    auto node = value_pair.second->node();
    do {
      if (match->get(match_node) == node) {
        std::stringstream ss;
        ss << "[info] already matched node " << match_node << " to " << node;
        log(ss.str());
        continue;
      }
      if (match_node->kind() == prim::Param) {
        continue;
      }
      REQ(node->owningBlock() == current_block_);
      REQ(matchNodes(match_node, node));
      match->set(match_node, node);
      node_frontier->emplace_back(std::make_pair(match_node, node));
    } while (0);

    auto match_uses = value_pair.first->uses();
    auto uses = value_pair.second->uses();
    for (auto i = 0; i < std::min(uses.size(), match_uses.size()); ++i) {
      auto match_user = match_uses[i].user;
      auto user = uses[i].user;
      if (match->get(match_user) == user) {
        std::stringstream ss;
        ss << "[info] already matched node " << match_user << " to " << user;
        log(ss.str());
        continue;
      }
      REQ(user->owningBlock() == current_block_);
      REQ(matchNodes(match_user, user));
      node_frontier->emplace_back(std::make_pair(match_user, user));
    }
  }
  return true;
}

#undef REQ

bool MatchIterator::tryResetMatch(
    MatchIterator::Match* match,
    NodeFrontier* node_frontier) {
  log("[info] clearing match params");
  match->clear();
  node_frontier->clear();
  auto new_node = findNewNode();
  if (!new_node) {
    return true;
  } else {
    match->set(anchor_, new_node);
  }
  node_frontier->emplace_back(std::make_pair(anchor_, new_node));
  return false;
}

// This greedy Dijkstra-like algorithm works by collecting
// nodes and values into the match map.
//
// Each of these collection passes consists of adding to a frontier by
// matching and adding either the producer of inputs/consumers of outputs
// of every node in the current node_frontier.
// If any match fails, the algorithm restarts on a new seed node.
MatchIterator::Match MatchIterator::getMatch() {
  MatchIterator::Match match;
  NodeFrontier node_frontier;

  if (tryResetMatch(&match, &node_frontier)) {
    return match;
  }

  // Exclude Param and Return nodes from the match graph.
  while (match.size() < match_size_) {
    // If we have a frontier, continue matching
    bool frontier_match = matchNodeFrontier(&node_frontier, &match);
    if (!frontier_match) {
      log("[mismatch] match attempt failed.");
      if (tryResetMatch(&match, &node_frontier)) {
        log("[mismatch] unable to find new anchor for search.");
        return match;
      }
    }
    if (!node_frontier.size()) {
      log("[info] empty frontier");
      break;
    }
  }

  return match;
}

void MatchIterator::updateCurrentMatch() {
  match_ = getMatch();
  if (match_.size() != match_size_) {
    std::stringstream os;
    os << "[mismatch] insufficient size, got match of size " << match_.size()
       << ", expected " << match_size_;
    log(os.str());
    finished_ = !finished_;
  }
}

size_t MatchIterator::Match::size() const {
  return node_match.size() + value_match.size();
}

void MatchIterator::Match::clear() {
  node_match.clear();
  value_match.clear();
}

void MatchIterator::Match::set(const Node* const& m, const Node* n) {
  node_match[m] = n;
}

void MatchIterator::Match::set(const Value* m, const Value* v) {
  value_match[m] = v;
}

const Node* MatchIterator::Match::get(const Node* const& m) const {
  auto iter = node_match.find(m);
  if (iter != node_match.end()) {
    return iter->second;
  }
  return nullptr;
}

const Value* MatchIterator::Match::get(const Value* const& m) const {
  auto iter = value_match.find(m);
  if (iter != value_match.end()) {
    return iter->second;
  }
  return nullptr;
}

const Node* MatchIterator::Match::operator[](const Node* const& m) const {
  return get(m);
}

const Value* MatchIterator::Match::operator[](const Value* const& m) const {
  return get(m);
}

} // namespace jit
} // namespace torch
