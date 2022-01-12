#include <torch/csrc/jit/passes/dbr_quantization/remove_redundant_aliases.h>

#include <torch/csrc/jit/jit_log.h>
#include <torch/csrc/jit/passes/quantization/helper.h>
#include <torch/csrc/jit/ir/alias_analysis.h>

namespace torch {
namespace jit {

namespace {

std::vector<std::string> _alias_func = {"alias"};

}

Module DBRQuantRemoveRedundantAliases(Module& module) {

  for (auto& method : module.get_methods()) {

    auto g = method.graph();
    AliasDb alias_db(g);

    std::vector<Node*> alias_nodes;
    std::stack<Block*> blocks_to_visit;
    blocks_to_visit.push(g->block());
    while (!blocks_to_visit.empty()) {
      Block* b = blocks_to_visit.top();
      blocks_to_visit.pop();
      for (auto* node : b->nodes()) {
        if (isAtenFunc(node, _alias_func)) {
          alias_nodes.push_back(node);
        }
        for (auto* subblock : node->blocks()) {
          blocks_to_visit.push(subblock);
        }
      }
    }

    for (auto* node : alias_nodes) {

      GRAPH_DEBUG(*node);

      Value* input_value = node->inputs()[0];
      Value* output_value = node->outputs()[0];

      bool always_safe_to_mutate = alias_db.safeToChangeAliasingRelationship(
        node->inputs(), node->outputs());

      const auto g_in = g->inputs();
      const auto g_out = g->outputs();
      bool is_input = std::find(
        g_in.begin(), g_in.end(), input_value) != g_in.end();
      bool is_output = std::find(
        g_out.begin(), g_out.end(), output_value) != g_out.end();
      // Since this pass takes an inlined graph, we assume that aliasing is
      // safe to update on inputs and outputs if they do not have writers.
      bool input_safe_to_mutate = (is_input &&
        !alias_db.hasWriters(input_value) &&
        !alias_db.hasWriters(output_value));
      bool output_safe_to_mutate = (is_output &&
        !alias_db.hasWriters(input_value) &&
        !alias_db.hasWriters(output_value));

      if (always_safe_to_mutate || input_safe_to_mutate || output_safe_to_mutate) {
        output_value->replaceAllUsesWith(input_value);
        node->destroy();
      }
    }
  }

  return module;
}

} // namespace jit
} // namespace torch
