#include <torch/csrc/jit/passes/onnx/pattern_conversion/pattern_encapsulation.h>
#include <torch/csrc/jit/passes/dead_code_elimination.h>
#include <torch/csrc/jit/passes/erase_number_types.h>
#include <torch/csrc/jit/passes/onnx.h>
#include <torch/csrc/jit/passes/onnx/pattern_conversion/common.h>

namespace torch {
namespace jit {

namespace {

// TODO: Add comment here for index_put pattern
void EncapsulateInplaceIndexPutForONNX(Node* index_put_node) {
  auto graph = index_put_node->owningGraph();

  // Find slice and select operators that are associated with this index
  // operator. E.g. x[1:3, 0] = y will generate one slice operator(1:3) and one
  // select operator(0).
  std::vector<Node*> slice_and_select_nodes =
      IndexPutPatternFinder::FetchSliceAndSelect(index_put_node);
  Node* last_node = slice_and_select_nodes.size() > 0
      ? slice_and_select_nodes.back()
      : index_put_node;
  Value* orig_data = last_node->input(0);

  orig_data->replaceAllUsesAfterNodeWith(
      index_put_node, index_put_node->output());

  // Copy related nodes into subblock of a new special placeholder node.
  Node* placeholder_node =
      graph->create(Symbol::fromQualString("onnx::Placeholder"));
  placeholder_node->s_(attr::name, "index_put");

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
  index_put_node->replaceAllUsesWith(placeholder_node);
}

void EncapsulateInplaceIndexPutForONNX(Block* block) {
  auto it = block->nodes().begin();
  while (it != block->nodes().end()) {
    auto node = *it;
    ++it;
    for (auto block : node->blocks()) {
      EncapsulateInplaceIndexPutForONNX(block);
    }

    if (node->kind() == aten::index_put_) {
      EncapsulateInplaceIndexPutForONNX(node);
    }
  }
}

} // namespace

void EncapsulatePatternIntoSubblock(const std::shared_ptr<Graph>& graph) {
  // Other patterns to be added here.
  EncapsulateInplaceIndexPutForONNX(graph->block());
}

} // namespace jit
} // namespace torch
