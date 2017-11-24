#include "torch/csrc/autograd/functions/basic_ops.h"

namespace torch { namespace autograd {

jit::value_list Add::symbolic(
    SymbolicContext* ctx,
    jit::value_list inputs,
    std::shared_ptr<jit::SourceLocation> sl
) {
  auto & g = ctx->graph;
  auto op_node = g->create(jit::kAdd, inputs);
  op_node->setSourceLocation(sl);
  auto node = g->appendNode(op_node)->output();
  return {node};
}

}}
