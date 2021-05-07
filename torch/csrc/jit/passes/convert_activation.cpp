#include <torch/csrc/jit/passes/convert_activation.h>
#include <torch/csrc/jit/passes/remove_mutation.h>
#include "ATen/core/interned_strings.h"
#include "ATen/core/jit_type.h"

namespace torch {
namespace jit {

namespace {
static const std::unordered_set<Symbol>
    inplace_activation_possible_type_promotion =
        {aten::hardsigmoid_, aten::hardswish_, aten::sigmoid_, aten::tanh_};

static const std::unordered_set<Symbol> inplace_activation_no_type_promotion = {
    aten::hardtanh_,
    aten::relu_,
    aten::relu6_};
} // namespace

FunctionalActivationRewriter::FunctionalActivationRewriter(
    std::shared_ptr<Graph> graph)
    : aliasDb_(nullptr), graph_(std::move(graph)) {
  for (auto op : inplace_activation_possible_type_promotion) {
    std::string name = std::string(op.toQualString());
    name.pop_back();
    functional_activation_possible_type_promotion.insert(
        Symbol::fromQualString(name));
  }
  for (auto op : inplace_activation_no_type_promotion) {
    std::string name = std::string(op.toQualString());
    name.pop_back();
    functional_activation_no_type_promotion.insert(
        Symbol::fromQualString(name));
  }
}

bool FunctionalActivationRewriter::CanBeInplace(Node* node) {
  if (functional_activation_possible_type_promotion.count(node->kind()) == 0 &&
      functional_activation_no_type_promotion.count(node->kind()) == 0) {
    return false;
  }

  Value* input = node->inputs().at(0);
  Value* output = node->output();
  if (input->type() != TensorType::get() ||
      output->type() != TensorType::get()) {
    return false;
  }
  auto inputDtype = input->type()->cast<TensorType>()->scalarType();
  auto outputDtype = output->type()->cast<TensorType>()->scalarType();

  // In general, we don't need to check shape as activation ops are
  // element-wise. But for those where type promotion could happen, we need to
  // make sure the dtype of input and output are the same. For now the dtype
  // checking will always fail until the type inference is ready.
  if (functional_activation_possible_type_promotion.count(node->kind()) != 0 &&
      (!inputDtype || !outputDtype ||
       inputDtype.value() != outputDtype.value())) {
    return false;
  }

  // Skip if input's def node has side effect or input has alias
  if (!MutationRemover::hasNoSideEffectOrAlias(input, getOrCreateAliasDb())) {
    return false;
  }

  // If x has more than one use, skip the converson.
  // TODO: Use liveness analysis to catch more general scenario
  return (input->uses().size() == 1);
}

bool FunctionalActivationRewriter::FunctionalToInplace(Block* block) {
  bool changed = false;
  for (auto it = block->nodes().begin(); it != block->nodes().end();) {
    auto* node = *it;
    it++;

    for (Block* sub_block : node->blocks()) {
      changed |= FunctionalToInplace(sub_block);
    }

    if (!CanBeInplace(node)) {
      continue;
    }

    changed = true;
    Node* inplace_node = node->replaceWithNewSymbol(
        Symbol::fromQualString(node->schema().name() + "_"));
    inplace_node->output()->replaceAllUsesWith(node->inputs().at(0));
    getOrCreateAliasDb()->replaceWithNewValue(
        node->output(), inplace_node->output());

    node->destroy();
  }
  return changed;
}

bool InplaceToFunctionalActivation(const std::shared_ptr<Graph>& graph) {
  return RemoveTensorMutation(graph, [](Node* node) {
    return inplace_activation_possible_type_promotion.count(node->kind()) !=
        0 ||
        inplace_activation_no_type_promotion.count(node->kind()) != 0;
  });
}

bool FunctionalToInplaceActivation(const std::shared_ptr<Graph>& graph) {
  FunctionalActivationRewriter f(graph);
  return f.FunctionalToInplace(graph->block());
}

} // namespace jit
} // namespace torch
