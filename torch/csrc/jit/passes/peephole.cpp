#include "torch/csrc/jit/passes/peephole.h"
#include "torch/csrc/jit/symbolic_variable.h"
#include "torch/csrc/jit/passes/dead_code_elimination.h"

namespace torch { namespace jit {

// The intent for this optimization pass is to catch all of the small, easy to
// catch peephole optimizations you might be interested in doing.
//
// Right now, it does:
//    - Eliminate no-op 'expand' nodes
//    - Simply x.t().t() to x
//
// TODO: Decide what kind of fixed point strategy we will have
void PeepholeOptimize(Block * block) {
  for (auto it = block->nodes().begin(); it != block->nodes().end(); ++it) {
    auto* n = *it;

    for (Block * sub_block : n->blocks()) {
        PeepholeOptimize(sub_block);
    }

    // XXX: remember that if you want to simplify an expression by combining multiple nodes
    // into a different one, then you need to check that they all belong to the given block
    switch (n->kind()) {
      case aten::expand: {
        // Eliminate redundant expand
        if (!n->input()->isTensor()) break;
        // the sizes are dynamic
        if(n->inputs().size() != 1) break;
        if (n->is(attr::size) == n->input()->type()->expect<TensorType>()->sizes()) {
          n->output()->replaceAllUsesWith(n->input());
          // Let DCE clean up any unused nodes at this point
        }
      } break;
      case aten::t: {
        // x.t().t() == x
        auto input_node = n->input()->node();
        if (input_node->kind() == aten::t)  {
          n->output()->replaceAllUsesWith(input_node->input());
          // Let DCE clean up any unused nodes at this point
        }
      } break;
      // Fuse mm + add into addmm
      case aten::add: {
        // Must have two inputs
        if (n->inputs().size() != 2) {
          continue;
        }
        // Alpha parameter must be 1.0
        auto alpha = at::Scalar(it->t(attr::alpha));
        if (alpha.to<double>() != 1.0) {
          continue;
        }

        auto input_node = n->input(1)->node();
        // Input must be an mm node
        if (input_node->kind() != aten::mm) {
          continue;
        }

        WithInsertPoint guard(n);

        SymbolicVariable mat(n->input(0));
        SymbolicVariable mat1(input_node->input(0));
        SymbolicVariable mat2(input_node->input(1));
        SymbolicVariable addmm_value = mat.addmm(mat1, mat2);

        // Copy shape information from output node
        ((Value*)addmm_value)->copyMetadata(n->output());
        n->output()->replaceAllUsesWith(addmm_value);
        // Let DCE clean up any unused nodes at this point
      } break;
    }
  }
}

void PeepholeOptimize(std::shared_ptr<Graph>& graph) {
  PeepholeOptimize(graph->block());
  // Eliminate dead code created by any peephole passes we've just done
  EliminateDeadCode(graph->block());
}

}}
