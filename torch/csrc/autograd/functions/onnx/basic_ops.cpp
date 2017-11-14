#include "torch/csrc/autograd/functions/basic_ops.h"

namespace torch { namespace autograd {

jit::value_list Add::symbolic(SymbolicContext* ctx, jit::value_list inputs) {
  auto & g = ctx->graph;
  auto node = g->appendNode(g->create(jit::kAdd, inputs))->output();
  return {node};
}

}}
