#include <torch/csrc/jit/passes/dbr_quantization/remove_redundant_aliases.h>

#include <torch/csrc/jit/ir/alias_analysis.h>
#include <torch/csrc/jit/jit_log.h>
#include <torch/csrc/jit/passes/quantization/helper.h>
#include <torch/csrc/jit/runtime/graph_iterator.h>

namespace torch::jit {

namespace {

void DBRQuantRemoveRedundantAliasesImpl(const Method& method) {
  auto g = method.graph();
  const bool is_frozen = false;
  const bool descend_function_calls = true;
  AliasDb alias_db(g, is_frozen, descend_function_calls);
  // find the alias nodes
  std::vector<Node*> alias_nodes;
  DepthFirstGraphNodeIterator it(g);
  Node* node = nullptr;
  while ((node = it.next()) != nullptr) {
    if (node->kind() == Symbol::aten("alias")) {
      alias_nodes.push_back(node);
    }
  }

  // remove the alias nodes, if it is safe to do so
  for (auto* node : alias_nodes) {
    GRAPH_DEBUG(*node);

    Value* input_value = node->input();
    Value* output_value = node->output();

    bool always_safe_to_mutate = alias_db.safeToChangeAliasingRelationship(
        node->inputs(), node->outputs());

    const auto g_in = g->inputs();
    const auto g_out = g->outputs();
    bool is_input =
        std::find(g_in.begin(), g_in.end(), input_value) != g_in.end();
    bool is_output =
        std::find(g_out.begin(), g_out.end(), output_value) != g_out.end();
    // We assume that aliasing is safe to update on inputs and outputs if they
    // do not have writers.
    bool input_safe_to_mutate =
        (is_input && !alias_db.hasWriters(input_value) &&
         !alias_db.hasWriters(output_value));
    bool output_safe_to_mutate =
        (is_output && !alias_db.hasWriters(input_value) &&
         !alias_db.hasWriters(output_value));

    if (always_safe_to_mutate || input_safe_to_mutate ||
        output_safe_to_mutate) {
      output_value->replaceAllUsesWith(input_value);
      node->destroy();
    }
  }
}

} // namespace

Module DBRQuantRemoveRedundantAliases(Module& module) {
  for (const auto& child : module.modules()) {
    for (const auto& method : child.get_methods()) {
      DBRQuantRemoveRedundantAliasesImpl(method);
    }
  }

  return module;
}

} // namespace torch::jit
