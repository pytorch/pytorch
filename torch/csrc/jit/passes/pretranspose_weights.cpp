#include <torch/csrc/jit/jit_log.h>
#include <torch/csrc/jit/passes/pretranspose_weights.h>

namespace torch {
namespace jit {

namespace {
class PretransposeWeightsHelper {
  public:
    PretransposeWeightsHelper() {}
    void run(std::shared_ptr<Graph>& graph) {
      /* We just call contiguous because the tensor passed into addmm is a
       * transposed, possibly strided tensor
       */
      Block* block = graph->block();
      for (auto it = block->nodes().begin(), end = block->nodes().end();
           it != end; ++it) {
        if (it->matches("aten::addmm(Tensor self, Tensor mat1, Tensor mat2, *, Scalar beta, Scalar alpha) -> Tensor")) {
          auto mat2_tensor = it->get<at::Tensor>(attr::mat2);
          auto mat2_node = it->namedInput(attr::mat2)->node();
          if (mat2_node->kind() == prim::Constant &&
              !mat2_tensor->is_contiguous()) {
            mat2_node->t_(attr::value, mat2_tensor->contiguous());
            GRAPH_UPDATE("Pretranspose weight for ", *it);
          }
        }
      }
    }

};
} // namespace

void PretransposeWeights(std::shared_ptr<Graph>& g) {
  PretransposeWeightsHelper pretransposer;
  pretransposer.run(g);
  GRAPH_DUMP("After pretranspose weights: ", g);
}

} // namespace jit
} // namespace torch
