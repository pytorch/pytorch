
#include <ATen/core/interned_strings.h>
#include <torch/csrc/jit/ir/alias_analysis.h>
#include <torch/csrc/jit/ir/constants.h>
#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/ir/ir_views.h>
#include <torch/csrc/jit/jit_log.h>
// IDEA for dtype_analysis:
// Anything that is not a Scalar or Tensor is by default assumed to be
// not part of dtype analysis

// Some ops force upgrade to at least certain dtype

void PropagateDtypesOnBlock(Block* b, const AliasDb& db) {
  for (Node* n : b->nodes()) {
    // TODO: handle loop
    if (n->kind() == prim::If) {
      IfView if_v(n);
      PropagateDtypesOnBlock(if_v.thenBlock(), db);
      PropagateDtypesOnBlock(if_v.elseBlock(), db);
      mergeTypes(if_v.thenOutputs(), if_v.elseOutputs(), if_v.outputs());
    } else if (n->maybeSchema()) {
      if (auto maybe_graph = shapeComputeGraphForSchema(n->schema())) {
        PropagateDtypesWithShapeFunction(n, *maybe_graph, db);
      }
    } else if (n->kind() == prim::TupleConstruct) {
      auto orig_type = n->output()->type()->expect<TupleType>();
      auto new_types = fmap(n->inputs(), [](Value* v) { return v->type(); });
      n->output()->setType(
          orig_type->createWithContained(std::move(new_types)));
    }
  }
}

void PropagateDtypesOnGraph(std::shared_ptr<Graph>& graph) {
  AliasDb db(graph);
  PropagateDtypesOnBlock(graph->block(), db);
}

} // namespace jit
} // namespace torch
