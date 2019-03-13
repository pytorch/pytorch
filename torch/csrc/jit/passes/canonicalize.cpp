#include <torch/csrc/jit/passes/canonicalize.h>

namespace torch {
namespace jit {

// Canonicalize a graph, renumbering it so that all structurally equivalent
// graphs have same numbers.
// keep_unique_names: If false, canonicalizes unique names by removing them
//   and replacing them with normal value names.
//   Otherwise, ignores values with unique names.
std::shared_ptr<Graph> Canonicalize(
    const std::shared_ptr<Graph>& graph,
    bool keep_unique_names) {
  auto r = std::make_shared<Graph>(graph->current_scope());
  std::unordered_map<Value*, Value*> rn_env;
  auto rn_fn = [&](Value* v) { return rn_env.at(v); };
  for (auto* input : graph->inputs()) {
    auto* r_input = r->addInput();
    r_input->copyMetadata(input);
    if (!keep_unique_names)
      r_input->setUniqueName("");
    rn_env[input] = r_input;
  }
  for (auto* node : graph->nodes()) {
    auto* r_node = r->createClone(node, rn_fn);
    if (!keep_unique_names) {
      for (auto* output : r_node->outputs()) {
        output->setUniqueName("");
      }
    }
    r->appendNode(r_node);
    auto outputs = node->outputs();
    auto r_outputs = r_node->outputs();
    for (size_t i = 0; i < outputs.size(); i++) {
      rn_env[outputs.at(i)] = r_outputs.at(i);
    }
    if (node->hasAttribute(attr::Subgraph)) {
      r_node->g_(
          attr::Subgraph,
          Canonicalize(node->g(attr::Subgraph), keep_unique_names));
    }
  }
  for (auto* output : graph->outputs()) {
    r->registerOutput(rn_fn(output));
  }

  return r;
}

} // namespace jit
} // namespace torch
