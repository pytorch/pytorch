#include "torch/csrc/jit/passes/onnx/peephole.h"

#if defined(_MSC_VER)
#include <BaseTsd.h>
typedef SSIZE_T ssize_t;
#endif

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

bool isNopTranspose(const std::vector<int64_t> & perm) {
  for (int64_t i = 0; i < perm.size(); i++)
    if (perm[i] != i)
      return false;
  return true;
}

// returns a vector `ret` such that transposing by `ret` is equivalent
// to transposing by `t1` and then by `t2`
std::vector<int64_t> composeTransposes(const std::vector<int64_t> & t1,
                                       const std::vector<int64_t> & t2) {
  JIT_ASSERT(t1.size() == t2.size());
  std::vector<int64_t> ret;
  for (size_t i = 0; i < t1.size(); i++) {
    JIT_ASSERT(   t1[i]  < int64_t(t2.size()));
    JIT_ASSERT(t2[t1[i]] < int64_t(t2.size()));
    ret.push_back(t2[t1[i]]);
  }
  return ret;
}

bool isBroadcasting(Node *node) {
  return broadcasting.count(node->kind());
}

// First iterate over the 'from' tensor sizes. Ignore all leading and trailing
// dimensions that are simply one, since they can be trivially broadcasted.
// When iterating over the dimension sizes (with reduced 'from' tensor),
// starting at the trailing dimension, the dimension sizes must either be equal,
// or one of them does not exist.
//
// Note that this is NOT equivalent to numpy broadcasting semantics, and do
// not represent that generalized broadcasting that Pytorch implements in
// general. Rather, this is Caffe2-style broadcasting.
bool fusibleExpandTo(at::IntList from, at::IntList to) {
  if (from.size() > to.size()) {
    return false;
  }
  ssize_t from_dim_start = 0, from_dim_end = from.size() - 1;
  while (from_dim_start < (ssize_t) from.size() && from[from_dim_start] == 1) {
    from_dim_start++;
  }
  while (from_dim_end > from_dim_start && from[from_dim_end] == 1) {
    from_dim_end--;
  }

  ssize_t f = from_dim_end;
  ssize_t t = to.size() - 1;
  for (; f >= from_dim_start && t >= 0; --f, --t) {
    if (from[f] != to[t]) return false;
  }

  // In the case that the 'to' tensor has leading ones in the same place that
  // the 'from' tensor does, f will be less than from_dim_start rather than
  // strictly equal. E.x.: to := [5, 1, 768] and from := [1, 1, 768]
  return f <= from_dim_start;
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
    auto* expanded_rhs = n->input(input_index)->node();

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
                         expanded_rhs->output()->type()->expect<TensorType>()->sizes())   // to
       ) continue;

    n->replaceInput(input_index, unexpanded_rhs);
    n->i_(kbroadcast, 1);
    if (!expanded_rhs->hasUses()) {
      expanded_rhs->destroy();
    }
  }
}

void fuseConsecutiveTransposes(std::shared_ptr<Graph>& graph) {
  for (auto it = graph->begin(); it != graph->end(); ++it) {
    auto* n = *it;

    if (n->kind() == kTranspose && n->input()->node()->kind() == kTranspose) {
      auto origInput = n->input();
      n->is_(kperm, composeTransposes(origInput->node()->is(kperm), n->is(kperm)));
      n->replaceInput(0, origInput->node()->input());
      if (origInput->uses().size() == 0) {
        origInput->node()->destroy();
      }
      continue;
    }
  }
}

void eliminateNopTranspose(std::shared_ptr<Graph>& graph) {
  for (auto it = graph->begin(); it != graph->end(); ++it) {
    auto* n = *it;

    if (n->kind() == kTranspose) {
      if (isNopTranspose(n->is(kperm))) {
        n->replaceAllUsesWith(n->input()->node());
        it.destroyCurrent();
        continue;
      }
    }
  }
}

void fuseTransposeIntoGemm(std::shared_ptr<Graph>& graph) {
  static const std::vector<int64_t> simpleTransPerm({1,0});

  for (auto it = graph->begin(); it != graph->end(); ++it) {
    auto* n = *it;

    if (n->kind() == kGemm) {
      for (size_t i : {0,1}) {
        auto inp = n->inputs()[i];
        auto trans = i == 0 ? ktransA : ktransB;
        if (inp->node()->kind() == kTranspose && inp->node()->is(kperm) == simpleTransPerm) {
          n->replaceInput(i, inp->node()->input());
          n->i_(trans, n->hasAttribute(trans) ? !n->i(trans) : 1);
          if (inp->uses().size() == 0) {
            inp->node()->destroy();
          }
        }
      }
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
//  - Fusing of consecutive transposes
//  - Elimiation of NOP transposes
//  - Fusing of transposes into Gemm
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
  fuseConsecutiveTransposes(graph);
  eliminateNopTranspose(graph);
  fuseTransposeIntoGemm(graph);
}

}}
