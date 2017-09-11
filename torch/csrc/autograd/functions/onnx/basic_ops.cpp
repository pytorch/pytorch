#include "torch/csrc/autograd/functions/basic_ops.h"

namespace torch { namespace autograd {

jit::node_list Add::symbolic(SymbolicContext* ctx, jit::node_list inputs) {
  auto & g = ctx->graph;
  auto node = g->create(jit::kAdd, inputs);
  g->appendNode(node);
  return {node};
}

}}
