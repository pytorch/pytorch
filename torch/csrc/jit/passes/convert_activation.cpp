#include <torch/csrc/jit/passes/convert_activation.h>
#include <torch/csrc/jit/passes/remove_mutation.h>
#include "ATen/core/interned_strings.h"

namespace torch {
namespace jit {

namespace {
static const std::unordered_set<Symbol> inplace_activation_ops = {
    aten::hardsigmoid_,
    aten::hardtanh_,
    aten::hardswish_,
    aten::relu_,
    aten::relu6_,
    aten::sigmoid_,
    aten::tanh_};
} // namespace

FunctionalActivationRewriter::FunctionalActivationRewriter(
    std::shared_ptr<Graph> graph)
    : aliasDb_(nullptr), graph_(std::move(graph)) {
  for (auto op : inplace_activation_ops) {
    std::string name = std::string(op.toQualString());
    name.pop_back();
    functional_activation_ops_.insert(Symbol::fromQualString(name));
  }
}

bool FunctionalActivationRewriter::FunctionalToInplaceActivation(Block* block) {
  bool changed = false;
  for (auto it = block->nodes().begin(); it != block->nodes().end();) {
    auto* node = *it;
    it++;

    for (Block* sub_block : node->blocks()) {
      changed |= FunctionalToInplaceActivation(sub_block);
    }

    if (functional_activation_ops_.count(node->kind()) == 0) {
      continue;
    }

    Value* input = node->inputs().at(0);
    Value* output = node->output();

    // Use y = relu(x) as an example:
    // If y and x have different type, skip the conversion
    if (input->type() != output->type()) {
      continue;
    }

    // If x is an input parameter or alias with an input parameter, skip the
    // conversion
    if (input->node()->kind() == prim::Param ||
        getOrCreateAliasDb()->mayContainAlias(input->node()->inputs(), input)) {
      continue;
    }

    // If x has more than one use, skip the converson.
    // TODO: Use more accurate analysis to relax this restriction.
    if (input->uses().size() != 1) {
      continue;
    }

    changed = true;
    Node* inplace_node = node->replaceWithNewSymbol(
        Symbol::fromQualString(node->schema().name() + "_"));
    inplace_node->output()->replaceAllUsesWith(input);
    getOrCreateAliasDb()->replaceWithNewValue(output, input);

    // it is an invariant that all mutable types have an element in the memory
    // dag so we must regive x an alias db element. We have already verified
    // that the mutated value is a fresh alias with a single use.
    // getOrCreateAliasDb()->createValue(mutated_value);

    // We must erase the destroyed node from the AliasDb lists of writes
    // getOrCreateAliasDb()->writeIndex_->erase(node);

    // now that we have removed a mutating op, the write cache is stale
    // TODO: don't strictly need to reset write cache, evaluate on models
    // getOrCreateAliasDb()->buildWrittenToLocationsIndex();

    node->destroy();
  }
  return changed;
}

bool InplaceToFunctionalActivation(const std::shared_ptr<Graph>& graph) {
  return RemoveTensorMutation(graph, [](Node* node) {
    return inplace_activation_ops.count(node->kind()) != 0;
  });
}

bool FunctionalToInplaceActivation(const std::shared_ptr<Graph>& graph) {
  FunctionalActivationRewriter f(graph);
  return f.FunctionalToInplaceActivation(graph->block());
}

} // namespace jit
} // namespace torch
