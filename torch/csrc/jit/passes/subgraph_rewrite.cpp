#include <torch/csrc/jit/passes/subgraph_rewrite.h>

#include <torch/csrc/jit/ir/irparser.h>
#include <torch/csrc/jit/ir/subgraph_matcher.h>

#include <c10/util/irange.h>

#include <utility>

namespace torch::jit {

namespace {
void update_source_range_and_cs_ptr(
    const std::set<const Node*>& input_nodes,
    const Match& m,
    std::unordered_map<Node*, Node*>& pattern_node_map) {
  // pattern_node_map, maps nodes of the replacement graph
  // to the nodes of the pattern graph.
  // Now we iterate over each node of the replacement graph
  // and find the corresponding pattern node in the match.
  // The matched's node's source range and callstack is then
  // used to update replacement node's source range and callstack
  for (auto& it : pattern_node_map) {
    Node* replacement_node = it.first;
    Node* pattern_node = it.second;
    if (!input_nodes.count(pattern_node)) {
      Node* orig_node = m.nodes_map.at(pattern_node);
      replacement_node->setSourceRange(orig_node->sourceRange());
      if (orig_node->callstack()) {
        replacement_node->setCallStack(orig_node->callstack().value());
      }
    }
  }
}
} // namespace

void SubgraphRewriter::RegisterDefaultPatterns() {
  // TODO: Add actual patterns (like Conv-Relu).
  RegisterRewritePattern(
      R"IR(
graph(%x, %w, %b):
  %c = aten::conv(%x, %w, %b)
  %r = aten::relu(%c)
  return (%r))IR",
      R"IR(
graph(%x, %w, %b):
  %r = aten::convrelu(%x, %w, %b)
  return (%r))IR",
      {{"r", "c"}});
}

void SubgraphRewriter::RegisterRewritePattern(
    const std::string& pattern,
    const std::string& replacement,
    const std::vector<std::pair<std::string, std::string>>& value_name_pairs) {
  std::unordered_map<std::string, std::string> value_name_map(
      value_name_pairs.begin(), value_name_pairs.end());
  RewritePatternDescr d = {pattern, replacement, std::move(value_name_map)};
  patterns_.push_back(std::move(d));
}

Module SubgraphRewriter::runOnModule(const Module& module) {
  nodes_to_delete_.clear();
  for (const auto& m : module.get_methods()) {
    auto g = toGraphFunction(m.function()).graph();
    runOnGraph(g);
  }
  return module;
}

void SubgraphRewriter::runOnGraph(
    std::shared_ptr<Graph>& graph,
    const std::vector<MatchFilter>& filters) {
  for (const RewritePatternDescr& pattern : patterns_) {
    rewriteSinglePatternOnGraph(graph, pattern, filters);
  }
}

void SubgraphRewriter::rewriteSinglePatternOnGraph(
    std::shared_ptr<Graph>& graph,
    const RewritePatternDescr& pattern,
    const std::vector<MatchFilter>& filters) {
  std::unordered_map<Value*, Value*> rewrite_map;
  std::vector<Value*> values_to_rewrite;

  Graph pattern_graph;
  std::unordered_map<std::string, Value*> vmap;
  parseIR(pattern.pattern, &pattern_graph, vmap);

  Graph replacement_graph;
  std::unordered_map<std::string, Value*> vmap_replacement;
  parseIR(pattern.replacement, &replacement_graph, vmap_replacement);

  // First construct map of Node*-to-Node*
  // This maps Nodes in replacement graph to nodes in pattern graph
  // given the value_name_map, which maps value names from replacement
  // pattern to value name in pattern
  std::unordered_map<Node*, Node*> pattern_node_map;
  std::set<const Node*> pattern_input_nodes;
  for (auto& it : vmap_replacement) {
    const auto& replacement_value_name = it.first;
    Node* replacement_value_node = it.second->node();
    if (pattern.value_name_map.count(replacement_value_name)) {
      const auto& pattern_value_name =
          pattern.value_name_map.at(replacement_value_name);
      TORCH_CHECK(
          vmap.count(pattern_value_name),
          "Value must be found in the replacement graph.");
      Node* pattern_value_node = vmap.at(pattern_value_name)->node();
      pattern_node_map.emplace(replacement_value_node, pattern_value_node);
    }
  }

  const auto& matches = findPatternMatches(pattern_graph, *graph);
  for (const Match& match : matches) {
    if (!std::all_of(filters.begin(), filters.end(), [&](const MatchFilter& f) {
          return f(match, vmap);
        })) {
      continue;
    }
    // Matches might overlap with each other, in that case some of the nodes in
    // the current match might have already been used in another folded pattern.
    // We need to skip such matches.
    if (overlapsWithPreviousMatches(&match)) {
      continue;
    }

    // Figure out what values we need to use as inputs and outputs for the
    // replacement subgraph and where the replacement subgraph needs to be
    // inserted.
    Node* ins_point = nullptr;
    std::vector<Value*> inputs, outputs;
    for (Value* v : pattern_graph.inputs()) {
      Value* input = match.values_map.at(v);
      if (!ins_point || ins_point->isBefore(input->node())) {
        ins_point = input->node();
      }
      inputs.push_back(input);
    }
    AT_ASSERT(ins_point);

    // Check that the insertion point we've chosen precedes all the uses of the
    // outputs - otherwise the replacement is incorrect and we have to skip it.
    bool ins_point_before_uses = true;
    for (Value* v : pattern_graph.outputs()) {
      Value* output = match.values_map.at(v);
      outputs.push_back(match.values_map.at(v));

      for (const Use& u : output->uses()) {
        if (u.user->isBefore(ins_point)) {
          ins_point_before_uses = false;
          break;
        }
      }
    }

    if (!ins_point_before_uses) {
      continue;
    }

    // Before rewriting the graph, update source range and callstack
    // info of the replacement pattern graph so that the rewritten graph
    // has the updated info
    update_source_range_and_cs_ptr(
        pattern_input_nodes, match, pattern_node_map);
    // Insert a clone of replacement subgraph.
    // `inputs` vector holds values that we would use as incoming values to the
    // new subgraph, and we will get `new_outputs` vector containing values
    // produced by this new subgraph - we will then rewrite old outputs with the
    // new ones.
    WithInsertPoint insert_point(ins_point->next());
    std::vector<Value*> new_outputs =
        insertGraph(*graph, replacement_graph, inputs);

    // Record all planned rewritings
    AT_ASSERT(outputs.size() == new_outputs.size());
    for (const auto idx : c10::irange(outputs.size())) {
      values_to_rewrite.push_back(outputs[idx]);
      rewrite_map[outputs[idx]] =
          new_outputs[idx]->setType(outputs[idx]->type());
    }
    // Record all planned deletions
    for (Node* pattern_n : pattern_graph.nodes()) {
      if (match.nodes_map.count(pattern_n)) {
        Node* n = match.nodes_map.at(pattern_n);
        nodes_to_delete_.insert(n);
      }
    }
  }

  // Perform planned rewritings
  for (auto v : values_to_rewrite) {
    v->replaceAllUsesWith(rewrite_map.at(v));
  }

  // Perform planned deletions
  for (auto n : nodes_to_delete_) {
    n->removeAllInputs();
  }
  for (auto n : nodes_to_delete_) {
    n->destroy();
  }
  nodes_to_delete_.clear();
}

bool SubgraphRewriter::overlapsWithPreviousMatches(const Match* match) {
  for (auto n : match->nodes_map) {
    if (nodes_to_delete_.count(n.second)) {
      return true;
    }
  }
  return false;
}

Module PatternBasedRewrite(const Module& module) {
  // TODO: Deep-copy the module
  SubgraphRewriter subgraph_rewriter;
  subgraph_rewriter.RegisterDefaultPatterns();
  return subgraph_rewriter.runOnModule(module);
}

} // namespace torch::jit
