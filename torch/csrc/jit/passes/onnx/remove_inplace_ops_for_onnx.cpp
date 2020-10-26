#include <torch/csrc/jit/passes/onnx/remove_inplace_ops_for_onnx.h>
#include <torch/csrc/jit/passes/remove_inplace_ops.h>
#include <torch/csrc/jit/passes/remove_mutation.h>
#include <limits>

namespace torch {
namespace jit {

namespace {

void ReplaceCopyWithIndexPut(Block* block) {
  for (auto it = block->nodes().begin(), end = block->nodes().end(); it != end;
       ++it) {
    auto node = *it;
    for (auto block : node->blocks()) {
      ReplaceCopyWithIndexPut(block);
    }

    if (node->kind() == aten::copy_) {
      // aten::copy_ can be viewed as a special case of index_put, where the
      // tensor indices input is empty.
      // Remove aten::copy_, and replace it with index_put.
      // 1. create an empty listConstruct node as indices input for index_put.
      // 2. create index_put node.

      // Tracing aten::copy_ broadcasts the rhs values.
      // 3. Apply broadcasting for scripting.
      auto graph = node->owningGraph();
      auto dummy_list = graph->createList(OptionalType::ofTensor(), {})
                            ->insertBefore(node)
                            ->output();

      auto expanded_value =
          graph->create(aten::expand_as, {node->input(1), node->input(0)})
              ->insertBefore(node)
              ->output();
      expanded_value->node()->setSourceRange(node->sourceRange());
      expanded_value->copyMetadata(node->input(1));

      auto index_put =
          graph
              ->create(
                  aten::index_put_,
                  {node->input(0), dummy_list, expanded_value, node->input(2)})
              ->insertBefore(node)
              ->output();

      index_put->node()->setSourceRange(node->sourceRange());
      index_put->copyMetadata(node->output());
      node->output()->replaceAllUsesWith(index_put);

      it->removeAllInputs();
      it.destroyCurrent();
    }
  }
}

// aten::pop is inplace. The tensor list input is updated.
// This pass creates an aten::__getitem__ op to return the original output from
// aten::pop. Then it makes the original aten::pop operator return the updated
// tensor list, and replaces all later uses of that tensor list with this new
// output.
static void PrepareListPopForONNX(Block* b) {
  for (auto it = b->nodes().begin(), end = b->nodes().end(); it != end; ++it) {
    for (auto* child_block : it->blocks()) {
      PrepareListPopForONNX(child_block);
    }

    if (it->kind() == aten::pop) {
      //   %ten : Tensor = aten::pop(%seq, %pos)
      // Convert to
      //   %ten : Tensor = aten::__getitem__(%seq, %pos)
      //   %new_seq : Tensor[] = aten::pop(%seq, %pos)
      // And replace all uses of %seq afterwards with %new_seq
      Node* getitem_node =
          b->owningGraph()->create(aten::__getitem__, {it->inputs()});
      getitem_node->output()->copyMetadata(it->output());
      getitem_node->insertBefore(*it);
      it->output()->replaceAllUsesWith(getitem_node->output());

      it->output()->copyMetadata(it->inputs()[0]);
      it->inputs()[0]->replaceAllUsesAfterNodeWith(*it, it->output());
    }
  }
}

static void PrepareListAppendAndInsertForONNX(Block* b) {
  for (auto it = b->nodes().begin(), end = b->nodes().end(); it != end; ++it) {
    for (auto* child_block : it->blocks()) {
      PrepareListPopForONNX(child_block);
    }

    if (it->kind() == aten::insert || it->kind() == aten::append) {
      if (it->outputs().size() == 0) {
        it->addOutput();
        it->output()->copyMetadata(it->inputs()[0]);
      }
      it->inputs()[0]->replaceAllUsesAfterNodeWith(*it, it->output());
    }
  }
}

// Remove Mutation pass does not handle mutation on block inputs.
// To fix this, insert a clone node following the graph input:
// Example for graph input node %0:
// Before:
// graph(%0 : Tensor):
//   %5 : Tensor = aten::zero_(%0)
//   ...
// After:
// graph(%0 : Tensor):
//   %2 : None = prim::Constant()
//   %3 : Tensor = aten::clone(%0, %2)
//   %5 : Tensor = aten::zero_(%3)
//   ...

static void PrepareForRemoveMutations(MutationRemover& mr, Block* b) {
  for (auto it = b->nodes().begin(), end = b->nodes().end(); it != end; ++it) {
    for (auto* child_block : it->blocks()) {
      PrepareForRemoveMutations(mr, child_block);
    }
  }

  for (auto input : b->inputs()) {
    for (auto use : input->uses()) {
      Node* node = use.user;
      if (!mr.inplaceOpVariant(node)) {
        continue;
      }

      auto it = std::find(node->inputs().begin(), node->inputs().end(), input);

      if (it != node->inputs().end()) {
        int index = std::distance(node->inputs().begin(), it);

        std::cerr
            << "Warning: ONNX Preprocess - Removing mutation on block inputs. "
            << "This changes graph semantics." << std::endl;

        if (input->type()->kind() == TypeKind::ListType) {
          // Create an aten::list to clone the list in graph inputs
          auto newNode = node->owningGraph()->create(aten::list, 1);
          newNode->output()->copyMetadata(input);
          newNode->addInput(input);
          newNode->insertBefore(node);
          node->replaceInput(index, newNode->output());
          input->replaceAllUsesAfterNodeWith(node, newNode->output());
        } else {
          // Create an aten::clone to clone the tensor in graph inputs
          auto newNode = node->owningGraph()->create(aten::clone, 1);
          newNode->output()->copyMetadata(input);
          newNode->addInput(input);

          auto* noneNode = node->owningGraph()->create(prim::Constant);
          noneNode->output()->setType(NoneType::get());
          newNode->addInput(noneNode->output());

          newNode->insertBefore(node);
          noneNode->insertBefore(newNode);
          node->replaceInput(index, newNode->output());
          input->replaceAllUsesAfterNodeWith(node, newNode->output());
        }
      }
    }
  }
}

} // namespace

void PrepareInplaceOpsForONNX(const std::shared_ptr<Graph>& graph) {
  ReplaceCopyWithIndexPut(graph->block());
  PrepareListPopForONNX(graph->block());
  PrepareListAppendAndInsertForONNX(graph->block());
}

void RemoveInplaceOpsForONNX(const std::shared_ptr<Graph>& graph) {
  MutationRemover mr(graph);
  PrepareForRemoveMutations(mr, graph->block());
  RemoveTensorMutation(graph);
  RemoveListMutation(graph);
}

} // namespace jit
} // namespace torch
