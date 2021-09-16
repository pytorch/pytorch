#include <torch/csrc/jit/codegen/onednn/guard_shape.h>

#include <torch/csrc/jit/jit_log.h>
#include <torch/csrc/jit/passes/tensorexpr_fuser.h>
#include <torch/csrc/jit/passes/utils/subgraph_utils.h>
#include <torch/csrc/jit/runtime/graph_executor.h>

namespace torch {
namespace jit {
namespace fuser {
namespace onednn {

using tensor_type_converter_t =
    c10::function_ref<TensorTypePtr(const TensorTypePtr& t)>;

void insertTypeGuardForFusionGroup(
    Node* guarded_node,
    tensor_type_converter_t type_converter,
    Symbol kind) {
  GRAPH_DEBUG("Inserting a typecheck guard for a node", *guarded_node);
  auto subgraph = SubgraphUtils::getSubgraph(guarded_node);

  // Fixup types of the subgraph inputs
  std::vector<Value*> inputs_to_check;
  std::vector<TypePtr> guard_types;
  for (Value* input : guarded_node->inputs()) {
    // We only check inputs of the guarded nodes and expect user to infer
    // intermediates and outputs shapes
    if (!input->type()->cast<TensorType>()) {
      continue;
    }

    // fusion outputs are already guarded
    if (input->node()->kind() == prim::Constant ||
        input->node()->kind() == prim::LlgaFusionGroup) {
      continue;
    }
    inputs_to_check.push_back(input);
    guard_types.push_back(type_converter(input->type()->expect<TensorType>()));
  }
  if (!inputs_to_check.size()) {
    return;
  }

  // Add ipex::LlgaFusionGuard node
  //
  // ipex::LlgaFusionGuard nodes  look like the following:
  //   %out1 : Float(2, 3), %out2 : Int(10, 30), %types_match : bool =
  //   ipex::LlgaFusionGuard(%inp1 : Tensor, %inp2 : Tensor)
  //
  // They have N inputs whose types we are going to check and N+1 outputs. The
  // first N outputs specify expected types and N+1-th output holds the result
  // of the check (bool).
  Node* typecheck_node =
      guarded_node->owningGraph()
          ->create(kind, inputs_to_check, inputs_to_check.size() + 1)
          ->insertBefore(guarded_node);
  typecheck_node->tys_(attr::types, guard_types);
  Value* typecheck_result = typecheck_node->output(inputs_to_check.size());

  std::unordered_map<Value*, Value*> typechecked_inputs;
  for (size_t i = 0; i < typecheck_node->inputs().size(); ++i) {
    typechecked_inputs[typecheck_node->input(i)] = typecheck_node->output(i);
  }

  // Fixup types of the typecheck node outputs, which are used by the op in
  // execution
  typecheck_node->output(inputs_to_check.size())->setType(BoolType::get());
  for (size_t i = 0; i < typecheck_node->inputs().size(); ++i) {
    typecheck_node->output(i)->setType(typecheck_node->input(i)->type());
  }

  // Insert if
  auto versioning_if =
      guarded_node->owningGraph()
          ->create(prim::If, {typecheck_result}, guarded_node->outputs().size())
          ->insertAfter(typecheck_node);
  for (size_t idx = 0; idx < guarded_node->outputs().size(); ++idx) {
    versioning_if->output(idx)->setType(guarded_node->output(idx)->type());
    guarded_node->output(idx)->replaceAllUsesWith(versioning_if->output(idx));
  }
  auto true_block = versioning_if->addBlock();
  auto false_block = versioning_if->addBlock();

  // Fill in the false block. It should contain the unoptimized
  // copy of the fused subgraph.
  WithInsertPoint guard(false_block->return_node());
  const auto subgraph_outputs = insertGraph(
      *guarded_node->owningGraph(), *subgraph, guarded_node->inputs());
  for (Value* output : subgraph_outputs) {
    false_block->registerOutput(output);
  }

  // types get copied to the fallback graph, so remove specializations before
  // replacing
  removeTensorTypeSpecializations(false_block);
  replaceBlockWithFallbackGraph(false_block, guarded_node->inputs());

  // Fill in the true block. It has all inputs type-checked and its
  // body should be the fusion group node.
  guarded_node->moveBefore(true_block->return_node());
  for (size_t idx = 0; idx < guarded_node->inputs().size(); ++idx) {
    if (typechecked_inputs.count(guarded_node->input(idx))) {
      guarded_node->replaceInput(
          idx, typechecked_inputs.at(guarded_node->input(idx)));
    }
  }
  for (Value* output : guarded_node->outputs()) {
    true_block->registerOutput(output);
  }
}

//! [ Note -- prepareFusionGroupAndGuardOutputs implementation ]
//! shamelessly copying code from NNC (tensorexpr_fuser)  with very little
//! modification, original code at:
//! `torch/csrc/jit/passes/tensorexpr_fuser.cpp:prepareFusionGroupAndGuardOutputs`
//!
//! We have the assumption that LLGA does not have operators
//! depending on the content of the tensor.
void prepareFusionGroupAndGuardOutputs(Block* block) {
  std::vector<Node*> fusion_groups;
  for (Node* n : block->nodes()) {
    for (Block* b : n->blocks()) {
      prepareFusionGroupAndGuardOutputs(b);
    }
    if (n->kind() == prim::LlgaFusionGroup) {
      fusion_groups.push_back(n);
    }
  }
  for (Node* fusion_group : fusion_groups) {
    // TODO: add further optimization pass to removeOutputsUsedOnlyInSize,
    // refer to
    // `torch/csrc/jit/passes/tensorexpr_fuser.cpp:removeOutputsUsedOnlyInSize`
    // removeOutputsUsedOnlyInSize(fusion_group);
    insertTypeGuardForFusionGroup(
        fusion_group,
        [](const TensorTypePtr& t) { return t; },
        prim::LlgaFusionGuard);
  }
}

} // namespace onednn
} // namespace fuser
} // namespace jit
} // namespace torch