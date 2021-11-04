#pragma once

#include <memory>

#include <torch/csrc/jit/ir/ir.h>

namespace torch {
namespace jit {
namespace interpreter {
/*
This is an optimization that reduces the number of store/load/move nodes needed
by recognizing that parts of the graph are simple trees like a*x + b*y. When
this happens it is possible to work directly off of the stack by emitting the
tree in a depth-first left-to-right manner:
  load a
  load x
  mul # stack now is a*x
  load b
  load y
  mul # stack now is a*x, b*y
  add

can_emit_inline_[node] == true means that this node participates as a non-root
member of one of these trees. The code emitter will not emit this node when
it is encountered in the node. Instead the node is emitted in a depth first
traversal from where it is used in a tree.

To participate in a tree a node must have a single use (otherwise it is not
tree-like) and output a single value (for simplicity.) If our IR was functional,
these would be the only constraints. However, many nodes have side effects, so
we must ensure that emitting the nodes in depth first order from the tree's root
_does not reorder the emission of the nodes_. To ensure this, we work backward
from the root of a potential tree, visiting its inputs in reverse depth first
order, while scanning the node list backward (with the block_point node). When
these traversal line up we know it is safe to emit the tree in this way. We
ignore constant nodes, which do not have side effects.
*/
struct CanEmitInline {
  explicit CanEmitInline(Graph& graph) {
    scanBlock(graph.block());
  }
  bool canInline(Value* v) {
    return v->node()->kind() != prim::Param &&
        // without this a BailOut may float downstream past some later
        // BailOut
        // and receive a higher jf_index. Then a GUARD instruction
        // we generated for the floated BailOut will get popped up from the
        // instruction stack
        // by the later BailOut in createBailoutBlock and its jf_index
        // will become invalid.
        v->node()->kind() != prim::TensorExprGroup &&
        v->node()->kind() != prim::StaticSubgraph &&
        v->node()->kind() != prim::CudaFusionGroup &&
        v->node()->kind() != prim::FusionGroup &&
        v->node()->kind() != prim::BailOut && v->uses().size() == 1 &&
        v->node()->outputs().size() == 1;
  }

  Node* previousNonConstant(Node* n) {
    do {
      n = n->prev();
    } while (n->kind() == prim::Constant);
    return n;
  }

  Node* scanValue(Node* block_point, Value* v) {
    // this node is a candidate for inline, if our reverse scan of the
    // node list lines up with the use of v, we know it will be emitted in
    // tree order, and we can inlining. Scan continutes for further nodes.
    if (v->node() == block_point && canInline(v)) {
      // since we inlined this node, we may be able to recursively inline
      // its inputs, so we continue scanning it
      block_point = scanNode(v->node());
      can_emit_inline_[v->node()] = true;
    }
    // if it does not line up, we can't inline 'v', and will just generate
    // a load/move for it. However, other inputs may still appear in tree
    // order so we continue the scan of the inputs.
    return block_point;
  }

  Node* scanNode(Node* n) {
    // don't bother to scan nodes we have already determined to be inline
    if (can_emit_inline_.count(n)) {
      return nullptr;
    }
    for (auto b : n->blocks()) {
      scanBlock(b);
    }
    Node* block_point = previousNonConstant(n);
    for (auto it = n->inputs().rbegin(), end = n->inputs().rend(); it != end;
         ++it) {
      block_point = scanValue(block_point, *it);
    }
    return block_point;
  }

  void scanBlock(Block* b) {
    scanNode(b->return_node());
    for (auto node : b->nodes().reverse()) {
      scanNode(node);
    }
  }
  std::unordered_map<Node*, bool> can_emit_inline_;
};

} // namespace interpreter
} // namespace jit
} // namespace torch
