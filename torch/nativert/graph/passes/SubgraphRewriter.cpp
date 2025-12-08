#include <variant>

#include <c10/util/Exception.h>
#include <torch/nativert/graph/Graph.h>
#include <torch/nativert/graph/passes/SubgraphRewriter.h>
namespace torch::nativert {

const std::string kDummyTarget = "dummy";

//-------------------------
// SubgraphMatcher
//-------------------------

SubgraphMatcher::SubgraphMatcher(const Graph* pattern)
    : pattern_(pattern), pattern_root_(findRootNode(pattern_)) {}

const Node* SubgraphMatcher::findRootNode(const Graph* g) {
  return g->outputNode()->inputs()[0].value->producer();
}

std::optional<Match> SubgraphMatcher::match(Node* target_node) {
  if (!pattern_root_) {
    return std::nullopt;
  }

  Match current_match;
  if (tryMatchNode(pattern_root_, target_node, current_match)) {
    for (const Value* output : pattern_->outputs()) {
      TORCH_CHECK(
          current_match.value_map.find(output) != current_match.value_map.end(),
          "Not all outputs were matched to the pattern. ",
          "Please check that the first output node suffices ",
          "to traverse all output values in the pattern.");
    }
    return current_match;
  }

  return std::nullopt;
}

std::vector<Match> SubgraphMatcher::matchAll(Graph* graph) {
  std::vector<Match> matches;

  for (auto& node : graph->nodes()) {
    auto maybeMatch = match(&node);
    if (maybeMatch.has_value()) {
      matches.push_back(*maybeMatch);
    }
  }
  return matches;
}

namespace {
bool compareConstants(const Constant& a, const Constant& b) {
  return std::visit(
      [](const auto& lhs, const auto& rhs) -> bool {
        using LType = std::decay_t<decltype(lhs)>;
        using RType = std::decay_t<decltype(rhs)>;

        // Handle directly comparable types
        if constexpr (
            std::is_same_v<LType, RType> &&
            !std::is_same_v<LType, std::unique_ptr<Graph>>) {
          return lhs == rhs;
        }
        // Unsupported types (Graph)
        LOG(ERROR) << "Unsupported Constant types for pattern matching: "
                   << typeid(lhs).name() << " vs " << typeid(rhs).name();
        TORCH_CHECK(
            false,
            "Unsupported Constant types for pattern matching: ",
            typeid(lhs).name(),
            " vs ",
            typeid(rhs).name())
      },
      a,
      b);
}

auto findMatchingAttribute(const Node* target_node, const Attribute& attr) {
  return std::find_if(
      target_node->attributes().begin(),
      target_node->attributes().end(),
      [&](const Attribute& otherAttr) {
        return attr.name == otherAttr.name &&
            compareConstants(attr.value, otherAttr.value);
      });
}

auto findInputByName(const Node* pattern_node, const std::string& inputName) {
  return std::find_if(
      pattern_node->inputs().begin(),
      pattern_node->inputs().end(),
      [&](const NamedArgument& patternInput) {
        return inputName == patternInput.name;
      });
}
} // namespace

bool SubgraphMatcher::tryMatchNodeInputs(
    const Node* pattern_node,
    Node* target_node,
    Match& match) {
  TORCH_CHECK(
      pattern_node->numInputs() + pattern_node->attributes().size() ==
      target_node->numInputs() + target_node->attributes().size());
  TORCH_CHECK(target_node->numInputs() <= pattern_node->numInputs());
  TORCH_CHECK(pattern_node->attributes().size() <= target_node->numInputs());

  // Target node inputs should match pattern node inputs
  for (const auto i : c10::irange(target_node->numInputs())) {
    // Compare input values
    // Current target node input should match a pattern node input
    const auto& inputMatch =
        findInputByName(pattern_node, target_node->inputs()[i].name);
    if (inputMatch == pattern_node->inputs().end()) {
      return false;
    }

    const Value* pval = inputMatch->value;
    Value* tval = target_node->inputs()[i].value;
    if (!tryMatchValue(pval, tval, match)) {
      return false;
    }
  }

  // Pattern node attributes should match target node attributes
  std::unordered_set<std::string> matched_attributes;
  for (const auto i : c10::irange(pattern_node->attributes().size())) {
    // Compare attributes
    const auto& attr = pattern_node->attributes()[i];
    auto it = findMatchingAttribute(target_node, attr);
    if (it == target_node->attributes().end()) {
      return false; // Attribute not found or values differ
    }
    matched_attributes.insert(it->name);
  }

  // Target node attributes that do not match pattern node attributes should
  // match pattern node inputs
  for (const auto i : c10::irange(target_node->attributes().size())) {
    const auto& it = target_node->attributes()[i];
    if (matched_attributes.find(it.name) != matched_attributes.end()) {
      continue; // Skip attributes already matched
    }
    const auto& patternInput = findInputByName(pattern_node, it.name);
    if (patternInput == pattern_node->inputs().end()) {
      return false;
    }
    if (patternInput->value->producer()->target() != "prim.Input" ||
        patternInput->value->users().size() > 1) {
      return false; // Only a pattern graph input should match a constant attr
    }

    // Insert a dummy node to match the pattern input value
    // Record the attribute that should be used to replace the dummy node
    auto* targetGraph = target_node->owningGraph();
    Node* dummyNode = targetGraph->createNode(kDummyTarget);
    Value* dummyOutput = dummyNode->addOutput(
        targetGraph->getUniqueValueName(), Type::Kind::None);
    targetGraph->insertBefore(dummyNode, target_node);
    if (match.value_map.find(patternInput->value) != match.value_map.end()) {
      return match.value_map[patternInput->value]->producer()->target() ==
          kDummyTarget;
    }
    match.value_map[patternInput->value] = dummyOutput;
    match.dummy_input_to_attribute_map[dummyOutput] = &it.value;
  }
  return true;
}

bool SubgraphMatcher::tryMatchNode(
    const Node* pattern_node,
    Node* target_node,
    Match& match) {
  if (match.node_map.find(pattern_node) != match.node_map.end()) {
    return match.node_map[pattern_node] == target_node;
  }

  // If the pattern node is an input, it should match every node
  if (pattern_node->target() == "prim.Input") {
    return true;
  }

  if (pattern_node->target() != target_node->target() ||
      pattern_node->numOutputs() != target_node->numOutputs()) {
    return false;
  }

  int64_t deltaInputCount = static_cast<int64_t>(pattern_node->numInputs()) -
      static_cast<int64_t>(target_node->numInputs());
  int64_t deltaAttributesCount =
      static_cast<int64_t>(pattern_node->attributes().size()) -
      static_cast<int64_t>(target_node->attributes().size());
  // Number of inputs and attributes should match exactly
  // and the pattern should always have >= input count of the target node
  // and the pattern should always have <= attribute count of the target node
  if (deltaInputCount + deltaAttributesCount != 0 ||
      (deltaInputCount < 0 && deltaAttributesCount > 0)) {
    return false;
  }
  match.node_map[pattern_node] = target_node;

  for (const auto i : c10::irange(pattern_node->numOutputs())) {
    const Value* pval = pattern_node->outputs()[i];
    Value* tval = target_node->outputs()[i];
    if (!tryMatchValue(pval, tval, match)) {
      return false;
    }
  }

  return tryMatchNodeInputs(pattern_node, target_node, match);
}

bool SubgraphMatcher::isOutputValue(const Value* val) {
  for (const auto& output : pattern_->outputs()) {
    if (val == output) {
      return true;
    }
  }
  return false;
}

bool SubgraphMatcher::tryMatchValue(
    const Value* pval,
    Value* tval,
    Match& match) {
  if (match.value_map.find(pval) != match.value_map.end()) {
    return match.value_map[pval] == tval;
  }

  const Node* pProducer = pval->producer();
  Node* tProducer = tval->producer();
  // If the value in the pattern is an input, then it could have other uses
  // outside of the subgraph. Similarly, output values can also have uses
  // outside of the matching subgraph.
  if (pval->users().size() != tval->users().size() &&
      pProducer->target() != "prim.Input" && !isOutputValue(pval)) {
    return false;
  }

  if (pval->type().kind() != tval->type().kind()) {
    return false;
  }

  match.value_map[pval] = tval;

  return tryMatchNode(pProducer, tProducer, match);
}

//-------------------------
// SubgraphRewriter
//-------------------------

void SubgraphRewriter::registerRewritePattern(
    const std::string& pattern,
    const std::string& replacement) {
  patterns_.emplace_back(RewriteRule{pattern, replacement});
}

bool SubgraphRewriter::run(
    Graph* graph,
    const std::vector<MatchFilter>& filters) {
  bool mutated = false;
  for (const auto& [pattern, replacement] : patterns_) {
    const auto& pattern_graph = stringToGraph(pattern);
    const auto& replacement_graph = stringToGraph(replacement);
    mutated |=
        runForPattern(graph, *pattern_graph, *replacement_graph, filters);
  }
  return mutated;
}

bool SubgraphRewriter::runForPattern(
    Graph* graph,
    const Graph& pattern,
    const Graph& replacement,
    const std::vector<MatchFilter>& filters) {
  SubgraphMatcher matcher(&pattern);
  std::vector<Match> matches = matcher.matchAll(graph);

  VLOG(1) << "[GraphPasses] Found " << matches.size()
          << " matches for : " << name_;

  for (auto& m : matches) {
    if (!std::all_of(filters.begin(), filters.end(), [&](const MatchFilter& f) {
          return f(m, getVmap(pattern));
        })) {
      continue;
    }
    if (!overlapsWithUsedNodes(m, replacedNodes_)) {
      rewriteMatch(graph, m, pattern, replacement);
    }
  }

  for (auto* v : valuesToRewrite_) {
    graph->replaceAllUses(v, valueRewrites_.at(v));
  }

  for (auto* n : replacedNodes_) {
    for (const auto& input : n->inputs()) {
      input.value->eraseUser(n);
    }
    n->inputs().clear();
  }

  for (auto* n : replacedNodes_) {
    n->destroy();
  }

  bool mutated = (valuesToRewrite_.size() + valueRewrites_.size() +
                  replacedNodes_.size()) > 0;

  valuesToRewrite_.clear();
  valueRewrites_.clear();
  replacedNodes_.clear();

  graph->cleanupDeadNodes();
  graph->finalize();
  graph->lint();

  return mutated;
}

bool SubgraphRewriter::overlapsWithUsedNodes(
    const Match& match,
    const std::unordered_set<Node*>& usedNodes) {
  // If any node or value used by this match is already in usedNodes/usedValues,
  // then this match overlaps with a previously selected match.
  for (auto& kv : match.node_map) {
    Node* target_node = kv.second;
    if (usedNodes.find(target_node) != usedNodes.end()) {
      return true;
    }
  }
  return false;
}

void SubgraphRewriter::rewriteMatch(
    Graph* graph,
    const Match& match,
    const Graph& pattern,
    const Graph& replacement) {
  // TODO: Preserve original node metadata with python source traceback
  std::unordered_map<const Value*, Value*> valueMap;

  // Find the point at which to insert the new subgraph
  // and get pointers to input/output values to insert at
  Node* insertionPoint = nullptr;
  std::vector<Value*> inputs, outputs;
  for (Value* v : pattern.inputs()) {
    if (match.value_map.find(v) == match.value_map.end()) {
      continue;
    }
    Value* input = match.value_map.at(v);
    // We want to insert after latest producer of any input that is not a dummy
    // node
    if (!insertionPoint ||
        (insertionPoint->isBefore(input->producer()) &&
         input->producer()->target() != kDummyTarget)) {
      insertionPoint = input->producer();
    }
    inputs.push_back(input);
  }
  TORCH_CHECK(insertionPoint, "No insertion point found");

  // Check we're not inserting after any of the outputs
  bool insertionPointValid = true;
  for (const auto* v : pattern.outputs()) {
    Value* output = match.value_map.at(v);
    outputs.push_back(match.value_map.at(v));
    for (const auto* user : output->users()) {
      if (user->isBefore(insertionPoint)) {
        insertionPointValid = false;
        break;
      }
    }
  }
  if (!insertionPointValid) {
    return;
  }
  std::vector<Value*> newOutputs;
  {
    InsertingAfter guard(insertionPoint);

    newOutputs = graph->insertGraph(replacement, inputs, valueMap);
  }
  TORCH_CHECK(outputs.size() == newOutputs.size());

  for (auto i : c10::irange(outputs.size())) {
    valuesToRewrite_.push_back(outputs[i]);
    valueRewrites_[outputs[i]] = newOutputs[i];
  }

  for (auto& patternNode : pattern.nodes()) {
    if (match.node_map.find(&patternNode) != match.node_map.end()) {
      Node* n = match.node_map.at(&patternNode);
      replacedNodes_.insert(n);
    }
  }

  // Replace dummy values with constant attributes
  for (const auto& inputToAttr : match.dummy_input_to_attribute_map) {
    auto* dummy = inputToAttr.first;
    // dummy might not be used in rewritten graph
    // e.g., casted_batch_one_hot_lengths
    if (dummy->users().empty()) {
      continue;
    }

    for (auto& userNode : dummy->users()) {
      auto& userInputs = userNode->inputs();
      replacedNodes_.insert(dummy->producer());
      for (auto it = userInputs.begin(); it != userInputs.end(); ++it) {
        if (it->value == dummy) {
          Attribute newAttr;
          std::visit(
              [&](auto&& val) -> void {
                using T = std::decay_t<decltype(val)>;
                if constexpr (std::is_same_v<T, std::unique_ptr<Graph>>) {
                  LOG(ERROR)
                      << "Graph attributes are not supported yet. Skipping attribute";
                } else {
                  newAttr.value = val;
                }
              },
              *inputToAttr.second);
          newAttr.name = it->name;
          userNode->addAttribute(std::move(newAttr));
          dummy->eraseUser(userNode);
          userInputs.erase(it);
          break;
        }
      }
    }
  }
}

c10::FastMap<std::string, const Value*> SubgraphRewriter::getVmap(
    const Graph& pattern) {
  c10::FastMap<std::string, const Value*> vmap;
  for (const auto& v : pattern.inputs()) {
    vmap[std::string(v->name())] = v;
  }
  for (const auto& n : pattern.nodes()) {
    for (const Value* v : n.outputs()) {
      vmap[std::string(v->name())] = v;
    }
  }
  return vmap;
}
} // namespace torch::nativert
