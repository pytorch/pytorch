#include <torch/csrc/jit/operator.h>
#include <torch/csrc/jit/script/compiler.h>
#include <torch/csrc/jit/passes/decompose_ops.h>
#include <torch/csrc/jit/passes/dead_code_elimination.h>
#include <torch/csrc/jit/passes/constant_propagation.h>
#include <torch/csrc/jit/passes/shape_analysis.h>
#include <torch/csrc/jit/passes/utils/subgraph_utils.h>

namespace torch {
namespace jit {

bool DecomposeOps(Block* block, script::CompilationUnit& decompose_funcs) {
  bool decomposed = false;
  for (auto it = block->nodes().begin(), end = block->nodes().end(); it != end;
      ++it) {
    for (auto sub : it->blocks()) {
      DecomposeOps(sub, decompose_funcs);
    }

    if (it->matches(
          "aten::addmm(Tensor self, Tensor mat1, Tensor mat2, *, Scalar beta, Scalar alpha) -> Tensor",
          /*const_inputs=*/{attr::beta, attr::alpha})) {
      // For the case where we have an addmm where alpha and beta are Attributes
      // and both of those scalars are equal to 1.0, decompose this into an mm
      // followed by an add so that it can go through the existing optimization (batchmm)
      if (it->get<at::Scalar>(attr::alpha)->toDouble() != 1.0 ||
          it->get<at::Scalar>(attr::beta)->toDouble() != 1.0) {
        continue;
      }

      decomposed = true;
      WithInsertPoint guard(*it);

      std::shared_ptr<Graph> d_graph = decompose_funcs.get_function("addmm").graph();
      Value* new_output = inlineCallTo(*it->owningGraph(), *d_graph, it->inputs()).at(0);
      // Set the output of the decomposed graph to have the same output type as the
      // original op otherwise the canonicalized graph will have
      // TensorType as the output of this node which is incorrect
      new_output->setType(it->output()->type());
      it->output()->replaceAllUsesWith(new_output);
      it.destroyCurrent();
    }
  }
  return decomposed;
}

void DecomposeOps(std::shared_ptr<Graph>& graph) {
  static script::CompilationUnit decompose_funcs(R"SCRIPT(
      def addmm(self: Tensor, mat1: Tensor, mat2: Tensor, beta: number = 1.0, alpha: number = 1.0):
          return self + mat1.mm(mat2)
      )SCRIPT");
  bool is_decomposed = DecomposeOps(graph->block(), decompose_funcs);
  if (is_decomposed) {
    // we only re-run those passes when the graph get decomposed
    PropagateInputShapes(graph);
    ConstantPropagation(graph);
    EliminateDeadCode(graph);
  }
}

} // namespace jit
} // namespace torch
