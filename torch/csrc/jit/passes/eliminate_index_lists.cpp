#include "torch/csrc/jit/passes/eliminate_index_lists.h"

namespace torch { namespace jit {

void eraseIndexWithLists(Block* block) {
  auto g = block->owningGraph();
  for (auto it = block->nodes().begin(), end = block->nodes().end();
        it != end;) {
    Node * n = *it;
    ++it;

    for (auto b : n->blocks()) {
      eraseIndexWithLists(b);
    }

    // Replace a sequence of ListConstruct -> Index with a series of index_select
    // ops, one for each dimension in the list specification.
    if (n->kind() == prim::ListConstruct && n->output()->uses().size() == 1
        && n->output()->uses()[0].user->kind() == aten::index) {
      Node *index_node = n->output()->uses()[0].user;
      WithInsertPoint guard(n);
      Value *self = index_node->inputs()[0]; // Note this is carried across iterations
      for (size_t dim = 0; dim < n->inputs().size(); ++dim) {
        std::vector<NamedValue> input_args = {
          /*self=*/NamedValue(self),
          /*dim=*/NamedValue(g->insertConstant((int64_t)dim)),
          /*index=*/NamedValue(n->inputs()[dim])
        };
        self = g->insert(aten::index_select, input_args);
      } //  for (size_t dim = 0; ...
      index_node->output()->replaceAllUsesWith(self);
      // Let DCE clean up the original nodes.
    } //  if (n->kind() == prim::ListConstruct ...
  }
}

void eraseIndexWithLists(Graph* graph) {
  eraseIndexWithLists(graph->block());
}

}} // namespace torch::jit
