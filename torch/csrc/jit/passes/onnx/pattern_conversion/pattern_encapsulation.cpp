#include <torch/csrc/jit/passes/dead_code_elimination.h>
#include <torch/csrc/jit/passes/onnx.h>
#include <torch/csrc/jit/passes/onnx/pattern_conversion/common.h>
#include <torch/csrc/jit/passes/onnx/pattern_conversion/pattern_encapsulation.h>
#include <torch/csrc/jit/passes/onnx/remove_inplace_ops_for_onnx.h>

// EDITING THIS FILE? READ THIS FIRST!
// see Note [Edit Pattern Encapsulation] in pattern_encapsulation.h

namespace torch {
namespace jit {

namespace {

// Trace back all the slice & select nodes associated with the index_put node,
// and copy them under the placeholder subblock.
// E.g. The IR for x[1:3, 0] = update
//    ...
//    %8 : Float(2, 4) = aten::slice(%0, %4, %5, %6, %7)
//    ...
//    %11 : Float(2) = aten::select(%8, %9, %10)
//    ...
//    %13 : Tensor?[] = prim::ListConstruct()
//    ...
//    %16 : Float(2) = aten::index_put(%11, %13, %14, %15)
// The aten::index_put node alone does not contain any indices (%13 : Tensor?[]
// = prim::ListConstruct()).
Node* EncapsulateInplaceIndexPutForONNX(Node* index_put_node) {
  auto graph = index_put_node->owningGraph();

  // Find slice and select operators that are associated with this index
  // operator. E.g. x[1:3, 0] = y will generate one slice operator(1:3) and one
  // select operator(0).
  std::vector<Node*> slice_and_select_nodes =
      IndexingPatternFinder::FetchSliceAndSelect(index_put_node);
  Node* last_node = !slice_and_select_nodes.empty()
      ? slice_and_select_nodes.back()
      : index_put_node;
  Value* orig_data = last_node->input(0);

  // Copy related nodes into subblock of a new special placeholder node.
  Node* placeholder_node =
      graph->create(Symbol::fromQualString("onnx::Placeholder"));
  placeholder_node->s_(attr::name, index_put_node->kind().toUnqualString());
  placeholder_node->addInput(orig_data);

  // Construct subblock
  auto subblock = placeholder_node->addBlock();
  std::unordered_map<Value*, Value*> env;

  // slice_and_select_nodes are in reversed order.
  for (auto it = slice_and_select_nodes.rbegin();
       it != slice_and_select_nodes.rend();
       ++it) {
    auto n = *it;
    auto cloned_n = subblock->appendNode(graph->createClone(
        n, [&](Value* v) { return env.find(v) != env.end() ? env[v] : v; }));
    for (size_t i = 0; i < cloned_n->outputs().size(); ++i) {
      env[n->outputs().at(i)] = cloned_n->outputs().at(i);
    }
  }

  Node* new_index_put_node =
      subblock->appendNode(graph->createClone(index_put_node, [&](Value* v) {
        return env.find(v) != env.end() ? env[v] : v;
      }));
  for (auto o : new_index_put_node->outputs()) {
    subblock->registerOutput(o);
  }

  placeholder_node->insertBefore(index_put_node);
  placeholder_node->copyMetadata(index_put_node);
  index_put_node->replaceAllUsesWith(placeholder_node);

  return placeholder_node;
}

} // namespace

std::optional<Node*> EncapsulatePatternIntoSubblock(Node* n) {
  switch (n->kind()) {
    case aten::index_put_:
    case aten::index_put: {
      return EncapsulateInplaceIndexPutForONNX(n);
    }
  }
  return c10::nullopt;
}

} // namespace jit
} // namespace torch
