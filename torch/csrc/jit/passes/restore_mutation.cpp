#include <ATen/core/interned_strings.h>
#include <ATen/core/jit_type.h>
#include <torch/csrc/jit/passes/remove_mutation.h>
#include <torch/csrc/jit/passes/restore_mutation.h>

namespace torch {
namespace jit {
namespace {
static const std::unordered_set<Symbol>
    functional_activation_possible_type_promotion =
        {aten::hardsigmoid, aten::hardswish, aten::sigmoid, aten::tanh};

static const std::unordered_set<Symbol>
    functional_activation_no_type_promotion = {
        aten::hardtanh,
        aten::relu,
        aten::relu6};

static const std::unordered_set<Symbol> inplace_activation = []() {
  std::unordered_set<Symbol> target_ops;
  for (auto op : functional_activation_possible_type_promotion) {
    std::string name = std::string(op.toQualString()) + "_";
    target_ops.insert(Symbol::fromQualString(name));
  }
  for (auto op : functional_activation_no_type_promotion) {
    std::string name = std::string(op.toQualString()) + "_";
    target_ops.insert(Symbol::fromQualString(name));
  }
  return target_ops;
}();

} // namespace

FunctionalToInplaceRewriter::FunctionalToInplaceRewriter(
    std::shared_ptr<Graph> graph,
    c10::optional<std::function<FunctionalToInplaceRewriterFlags(Node*)>>
        filter)
    : aliasDb_(nullptr),
      graph_(std::move(graph)),
      node_filter_(std::move(filter)) {}

bool FunctionalToInplaceRewriter::CanBeInplace(Node* node) {
  bool check_shape = true;
  bool check_dtype = true;

  if (node_filter_) {
    auto flags = (*node_filter_)(node);
    if (!flags.transform) {
      return false;
    }
    check_shape = flags.check_shape;
    check_dtype = flags.check_dtype;
  }

  Value* input = node->inputs().at(0);
  Value* output = node->output();
  if (input->type() != TensorType::get() ||
      output->type() != TensorType::get()) {
    return false;
  }
  auto inputDtype = input->type()->cast<TensorType>()->scalarType();
  auto outputDtype = output->type()->cast<TensorType>()->scalarType();

  // if (check_shape) {
  //  return false;
  //}

  // In general, we don't need to check shape as activation ops are
  // element-wise. But for those where type promotion could happen, we need to
  // make sure the dtype of input and output are the same. For now the dtype
  // checking will always fail until the type inference is ready.
  if (check_dtype &&
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

bool FunctionalToInplaceRewriter::FunctionalToInplace(Block* block) {
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
    return inplace_activation.count(node->kind()) != 0;
  });
}

bool FunctionalToInplaceActivation(const std::shared_ptr<Graph>& graph) {
  FunctionalToInplaceRewriter f(graph, [](Node* node) {
    if (functional_activation_no_type_promotion.count(node->kind()) != 0) {
      // FunctionalToInplaceRewriterFlags{transform, check_shape, check_dtype}
      return FunctionalToInplaceRewriterFlags{true, false, false};
    } else if (
        functional_activation_possible_type_promotion.count(node->kind()) !=
        0) {
      return FunctionalToInplaceRewriterFlags{true, false, true};
    }
    return FunctionalToInplaceRewriterFlags{false, false, false};
  });
  return f.FunctionalToInplace(graph->block());
}

} // namespace jit
} // namespace torch
