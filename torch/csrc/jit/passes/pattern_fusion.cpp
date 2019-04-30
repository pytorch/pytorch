#include <torch/csrc/jit/passes/pattern_fusion.h>
#include <torch/csrc/jit/irparser.h>
#include <torch/csrc/jit/subgraph_matcher.h>

namespace torch {
namespace jit {

void PatternFuser::RegisterDefaultPatterns() {
  // TODO: Add actual patterns (like Conv-Relu).
  RegisterFusionPattern(
      R"IR(
graph(%x, %w, %b):
  %c = aten::conv(%x, %w, %b)
  %r = aten::relu(%c)
  return (%r))IR",
      /*fused_node_name=*/"aten::convrelu",
      /*inputs=*/{"x", "w", "b"},
      /*outputs=*/{"r"});
}

void PatternFuser::RegisterFusionPattern(
    const std::string& pattern,
    const std::string& fused_node_name,
    std::vector<std::string> inputs,
    std::vector<std::string> outputs) {
  FusionPatternDescr d = {pattern, fused_node_name, inputs, outputs};
  patterns_.push_back(d);
}

std::shared_ptr<script::Module> PatternFuser::runOnModule(
    std::shared_ptr<script::Module> module) {
  nodes_to_delete_.clear();
  const auto& methods = module->get_methods();
  for (const auto& m : methods) {
    auto g = m->function().graph();
    runOnGraph(g);
  }
  return module;
}

void PatternFuser::runOnGraph(std::shared_ptr<Graph>& graph) {
  for (const FusionPatternDescr& pattern : patterns_) {
    fuseSinglePatternOnGraph(graph, pattern);
  }
}

void PatternFuser::fuseSinglePatternOnGraph(
    std::shared_ptr<Graph>& graph,
    FusionPatternDescr pattern) {
  std::unordered_map<Value*, Value*> rewrite_map;
  std::vector<Value*> values_to_rewrite;

  Graph pattern_graph;
  std::unordered_map<std::string, Value*> vmap;
  script::parseIR(pattern.pattern, &pattern_graph, vmap);

  const auto& matches = findPatternMatches(pattern_graph, *graph);
  for (const Match& match : matches) {
    // Matches might overlap with each other, in that case some of the nodes in
    // the current match might have already been used in another folded pattern.
    // We need to skip such matches.
    if (overlapsWithPreviousMatches(&match)) {
      continue;
    }

    // Figure out what values we need to use as inputs and outputs for new fused
    // node. We need to go through two maps: from name in pattern-descriptor
    // (string) to a value in pattern graph (vmap), and from this value to a
    // value in the actual graph (match.values_map).
    std::vector<Value*> inputs, outputs;
    for (const std::string& i : pattern.inputs) {
      Value* v = vmap[i];
      inputs.push_back(const_cast<Value*>(match.values_map.at(v)));
    }
    for (const std::string& o : pattern.outputs) {
      Value* v = vmap[o];
      outputs.push_back(const_cast<Value*>(match.values_map.at(v)));
    }

    // Create a fused node and insert it after the last node in the matched
    // subgraph
    Node* fused_node = graph->create(
        at::Symbol::fromQualString(pattern.fused_node_name),
        inputs,
        outputs.size());
    fused_node->insertAfter(const_cast<Node*>(match.anchor));

    // Record all planned rewritings
    for (size_t idx = 0; idx < outputs.size(); idx++) {
      values_to_rewrite.push_back(outputs[idx]);
      rewrite_map[outputs[idx]] = fused_node->outputs()[idx];
    }
    // Record all planned deletions
    for (Node* pattern_n : pattern_graph.nodes()) {
      if (match.nodes_map.count(pattern_n)) {
        Node* n = const_cast<Node*>(match.nodes_map.at(pattern_n));
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
}

bool PatternFuser::overlapsWithPreviousMatches(const Match* match) {
  for (auto n : match->nodes_map) {
    if (nodes_to_delete_.count(const_cast<Node*>(n.second))) {
      return true;
    }
  }
  return false;
}

std::shared_ptr<script::Module> PatternBasedFusion(
    std::shared_ptr<script::Module> module) {
  // TODO: Deep-copy the module
  PatternFuser pattern_fuser;
  pattern_fuser.RegisterDefaultPatterns();
  return pattern_fuser.runOnModule(module);
}

} // namespace jit
} // namespace torch
