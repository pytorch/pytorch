#include "torch/csrc/jit/passes/onnx/peephole.h"

namespace torch { namespace jit {

// Broadcasting operators have the following property:
// They support a 'broadcast' flag, which enables broadcasting
// on the last argument.  ATM this is not full-Numpy broadcasting,
// only left-size extension (no size 1 to size n broadcast)
std::unordered_set<NodeKind> broadcasting = {
  kAdd,
  kDiv,
  kMul,
  kPow,
  kSub,
  kGemm,
};

bool isBroadcasting(Node *node) {
  return broadcasting.count(node->kind());
}

// When iterating over the dimension sizes, starting at the trailing dimension,
// the dimension sizes must either be equal, or one of them does not exist.
//
//  equivalently:
//
// Test that 'from' is a suffix of 'to'.
bool fusibleExpandTo(at::IntList from, at::IntList to) {
  auto f = from.rbegin();
  auto t = to.rbegin();
  for (; f != from.rend() && t != to.rend(); f++, t++) {
    // TODO: if 1->n expansion is supported, adjust this conditional.
    if (*f != *t) return false;
  }
  return f == from.rend();
}

void fuseBroadcast(std::shared_ptr<Graph>& graph) {
  for (auto it = graph->begin(); it != graph->end(); ++it) {
    auto* n = *it;

    // Can't fuse into nodes that don't support broadcasting
    if (!isBroadcasting(n)) continue;

    // If the node already broadcasts, can't "rebroadcast"
    // TODO: Actually, maybe you can, if there is a broadcast for some
    // dims, and then another broadcast for the rest.  But this will
    // never happen in practice so I didn't implement it.
    if (n->hasAttribute(kbroadcast) && n->i(kbroadcast)) continue;
    JIT_ASSERT(!n->hasAttribute(kaxis));

    auto input_index = n->inputs().size() - 1;
    auto* expanded_rhs = n->inputs().at(input_index);

    // The expanded_rhs input isn't actually an expand, so no fusion available
    if (expanded_rhs->kind() != kExpand) continue;

    auto* unexpanded_rhs = expanded_rhs->input();

    // We need to know what the type pre-expand is.  We should basically
    // always have this information (because expands are only ever traced,
    // not generated from symbolic), but if for some reason we don't
    // have it, we need to skip.
    if (!unexpanded_rhs->hasType()) continue;

    // Not all broadcasts are supported by ONNX broadcast.
    if (!fusibleExpandTo(unexpanded_rhs->type()->expect<TensorType>()->sizes(), // from
                         expanded_rhs->type()->expect<TensorType>()->sizes())   // to
       ) continue;

    n->replaceInput(input_index, unexpanded_rhs);
    n->i_(kbroadcast, 1);
    if (expanded_rhs->uses().size() == 0) {
      expanded_rhs->destroy();
    }
  }
}

// This optimization does ONNX-specific peephole optimizations.
//
// At the moment, here are the optimizations it does:
//  - This optimization fuses expand calls into ONNX operators, because it is
//    easier for non-strided backends to more efficiently do broadcasts if this is
//    local information.  This optimization is not useful for PyTorch as 'expand'
//    is free.
//
// Before you write an optimization here, ask yourself, "Could I do this
// optimization on ATen operators"?  If so, you should seriously consider
// writing your optimization in jit/passes/peephole.cpp rather than
// here, as it will be generally applicable to the JIT as well.  The
// optimizations here are ONLY applied on ONNX update
void PeepholeOptimizeONNX(std::shared_ptr<Graph>& graph) {
  // TODO: decide on fixpoint strategy
  // TODO: make it easier not to do O(k) iterations over the graph, where
  // k is the number of distinct peephole optimizations
  fuseBroadcast(graph);
}

}}
