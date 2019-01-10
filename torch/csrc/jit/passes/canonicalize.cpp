#include "torch/csrc/jit/passes/canonicalize.h"

namespace torch { namespace jit {

// Canonicalize a graph, renumbering it so that all structurally equivalent
// graphs have same numbers.
std::shared_ptr<Graph> Canonicalize(const std::shared_ptr<Graph>& graph) {
  auto r = std::make_shared<Graph>(graph->scope_root());
  std::unordered_map<Value*, Value*> rn_env;
  auto rn_fn = [&](Value* v) { return rn_env.at(v); };
  for (auto* input : graph->inputs()) {
    auto* r_input = r->addInput();
    r_input->copyMetadata(input);
    r_input->setStage(input->stage());
    rn_env[input] = r_input;
  }
  for (auto* node : graph->nodes()) {
    auto* r_node = r->createClone(node, rn_fn);
    r_node->setStage(node->stage());
    r->appendNode(r_node);
    auto outputs = node->outputs();
    auto r_outputs = r_node->outputs();
    for (size_t i = 0; i < outputs.size(); i++) {
      r_outputs.at(i)->setStage(outputs.at(i)->stage());
      rn_env[outputs.at(i)] = r_outputs.at(i);
    }
  }
  for (auto* output : graph->outputs()) {
    r->registerOutput(rn_fn(output));
  }
  r->setStage(graph->stage());

  return r;

}

}}
