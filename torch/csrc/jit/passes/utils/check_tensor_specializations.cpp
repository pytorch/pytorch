#include <torch/csrc/jit/passes/utils/check_tensor_specializations.h>
#include <torch/csrc/jit/jit_log.h>

namespace torch {
namespace jit {
// Code to detect specializations for testing CustomPasses.
static std::unique_ptr<RegisterPass> tTSpecializationDetectionPass = nullptr;
static bool hasSpecializations = false;

bool hasTensorTypeSpecialization(Value* v) {
  if (!v->type()->cast<TensorType>()) {
    return false;
  }
  // Constants & TensorExprGroup will always produce specialized tensor type,
  // TypeCheck are inserted by this pass and only used by fusion groups that
  // insert proper guards
  if (v->node()->kind() == prim::Constant ||
      v->node()->kind() == prim::TypeCheck ||
      v->node()->kind() == prim::TensorExprGroup) {
    return false;
  }
  if (v->type() == TensorType::get()) {
    return false;
  }
  return true;
}

bool hasTensorTypeSpecializations(torch::jit::Block* block) {
  for (Value* v : block->inputs()) {
    if (hasTensorTypeSpecialization(v))
      return true;
  }
  for (Node* n : block->nodes()) {
    for (torch::jit::Block* b : n->blocks()) {
      if (hasTensorTypeSpecializations(b))
        return true;
    }
    for (Value* v : n->outputs()) {
      if (hasTensorTypeSpecialization(v))
        return true;
    }
  }
  return false;
}

void detectTTSpecializationPass(std::shared_ptr<Graph>& graph) {
  GRAPH_DUMP("In detectTTSpecialization Custom Post Pass: ", graph);
  hasSpecializations = hasTensorTypeSpecializations(graph->block());
}

TORCH_API void addTensorTypeSpecializationDetectionPass() {
  tTSpecializationDetectionPass =
      std::make_unique<RegisterPass>(detectTTSpecializationPass);
}

TORCH_API void removeTensorTypeSpecializationDetectionPass() {
  tTSpecializationDetectionPass = nullptr;
}

TORCH_API bool passDetectedSpecializedTensors() {
  TORCH_INTERNAL_ASSERT(
      tTSpecializationDetectionPass,
      "TensorType specialization detection pass not registered");
  return hasSpecializations;
}
} // namespace jit
} // namespace torch
