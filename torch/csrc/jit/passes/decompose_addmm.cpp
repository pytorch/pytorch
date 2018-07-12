#include "torch/csrc/jit/passes/dead_code_elimination.h"
#include "torch/csrc/jit/symbolic_variable.h"

namespace torch { namespace jit {

static void DecomposeAddmm(Block* block) {
  for (auto it = block->nodes().begin(), end = block->nodes().end(); it != end;
       ++it) {
    for (auto sub : it->blocks())
      DecomposeAddmm(sub);
    // For the case where we have an addmm where alpha and beta are Attributes
    // and both of those scalars are equal to 1.0, decompose this into an mm
    // followed by an add so that it can go through the existing optimization,
    // shape analysis and differentiation passes for those two individual ops.
    // Later, we will fuse together those two ops into a single addmm.
    if (it->kind() == aten::addmm && it->inputs().size() == 3) {
      auto alpha = at::Scalar(it->t(attr::alpha));
      auto beta = at::Scalar(it->t(attr::beta));

      if (alpha.to<double>() != 1.0 || beta.to<double>() != 1.0) {
        continue;
      }

      WithInsertPoint guard(*it);

      SymbolicVariable mat(it->inputs()[0]);
      SymbolicVariable mat1(it->inputs()[1]);
      SymbolicVariable mat2(it->inputs()[2]);

      auto mm_result = mat1.mm(mat2);
      auto result = mat + mm_result;

      it->output()->replaceAllUsesWith(result);
      it.destroyCurrent();
    }
  }
}

void DecomposeAddmm(const std::shared_ptr<Graph>& graph) {
  DecomposeAddmm(graph->block());
}


}}
