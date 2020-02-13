#include <torch/csrc/jit/passes/fuse_linear.h>
#include <torch/csrc/jit/passes/subgraph_rewrite.h>

namespace torch {
namespace jit {

void FuseLinear(std::shared_ptr<Graph>& graph) {
  std::string addmm_pattern = R"IR(
    graph(%input, %weight, %bias, %4):
        %weight_t = aten::t(%weight)
        %res = aten::addmm(%bias, %input, %weight_t, %4, %4)
        return (%res))IR";
  std::string matmul_add_pattern = R"IR(
    graph(%input, %weight, %bias, %4):
        %weight_t = aten::t(%weight)
        %output = aten::matmul(%input, %weight_t)
        %res = aten::add_(%output, %bias, %4)
        return (%res))IR";
  std::string fused_linear = R"IR(
    graph(%input, %weight, %bias, %4):
        %res = aten::linear(%input, %weight, %bias)
        return (%res))IR";

  std::string matmul_pattern = R"IR(
    graph(%input, %weight):
        %weight_t = aten::t(%weight)
        %output = aten::matmul(%input, %weight_t)
        return (%output))IR";
  std::string fused_linear_bias_none = R"IR(
    graph(%input, %weight):
        %bias: Tensor? = prim::Constant()
        %res = aten::linear(%input, %weight, %bias)
        return (%res))IR";

  // replace addmm pattern to linear
  SubgraphRewriter addmm_to_linear;
  addmm_to_linear.RegisterRewritePattern(addmm_pattern, fused_linear);
  addmm_to_linear.runOnGraph(graph);

  // replace matmul + add pattern to linear
  SubgraphRewriter matmuladd_to_linear;
  matmuladd_to_linear.RegisterRewritePattern(matmul_add_pattern, fused_linear);
  matmuladd_to_linear.runOnGraph(graph);

  // replace matmul with bias=None pattern to linear
  SubgraphRewriter matmul_to_linear;
  matmul_to_linear.RegisterRewritePattern(
      matmul_pattern, fused_linear_bias_none);
  matmul_to_linear.runOnGraph(graph);
}
} // namespace jit
} // namespace torch
