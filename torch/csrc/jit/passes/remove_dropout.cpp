#include <torch/csrc/jit/passes/remove_dropout.h>

namespace torch::jit {

namespace {
bool isDropoutRemovable(const Node* node) {
  const auto inputs = node->inputs();
  TORCH_INTERNAL_ASSERT(inputs.size() == 3);
  const Value* training_input = inputs[2];
  auto optional_ivalue = toIValue(training_input);
  if (!optional_ivalue) {
    return false;
  }
  const IValue& val = optional_ivalue.value();
  TORCH_INTERNAL_ASSERT(val.isBool());
  const bool is_training = val.toBool();
  return !is_training;
}

void removeDropoutImpl(Block* block) {
  std::vector<Node*> deleted_nodes;

  for (auto it = block->nodes().rbegin(); it != block->nodes().rend(); it++) {
    Node* node = *it;
    for (auto block : node->blocks()) {
      removeDropoutImpl(block);
    }
    if ((node->kind() == c10::Symbol::fromQualString("aten::dropout") ||
         node->kind() == c10::Symbol::fromQualString("aten::dropout_") ||
         node->kind() == c10::Symbol::fromQualString("aten::feature_dropout") ||
         node->kind() ==
             c10::Symbol::fromQualString("aten::feature_dropout_")) &&
        isDropoutRemovable(*it)) {
      // Input tensor of dropout.
      Value* input_value = node->inputs()[0];
      // Output tensor.
      Value* output_value = node->outputs()[0];
      output_value->replaceAllUsesWith(input_value);
      deleted_nodes.push_back(node);
    }
  }
  for (auto del_node : deleted_nodes) {
    del_node->destroy();
  }
}
} // namespace

void removeDropout(std::shared_ptr<Graph>& graph) {
  removeDropoutImpl(graph->block());
}

void removeDropout(script::Module& module) {
  TORCH_CHECK(
      !module.hasattr("training") || !module.is_training(),
      "Dropout removal module in training mode is not yet supported");
  auto graph = module.get_method("forward").graph();
  removeDropout(graph);
}

} // namespace torch::jit
