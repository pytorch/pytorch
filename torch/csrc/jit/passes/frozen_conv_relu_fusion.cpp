#include <torch/csrc/jit/passes/fuse_relu.h>

#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/ir/subgraph_matcher.h>
#include <torch/csrc/jit/passes/subgraph_rewrite.h>

namespace torch {
namespace jit {

namespace {
void fuseFrozenConvReluImpl(std::shared_ptr<Graph>& graph) {
  SubgraphRewriter rewriter;

  rewriter.RegisterDefaultPatterns();

  std::string add_relu_0 = R"(
    graph(%a, %b, %alpha):
        %add_res = aten::add(%a, %b, %alpha)
        %res = aten::relu(%add_res)
        return (%res))";
  std::string add_relu_fused = R"(
    graph(%a, %b, %alpha):
        %res = aten::_add_relu(%a, %b, %alpha)
        return (%res))";
  rewriter.RegisterRewritePattern(add_relu_0, add_relu_fused);

  rewriter.runOnGraph(graph);
}
}

void FuseFrozenConvRelu(std::shared_ptr<Graph>& graph) {
  fuseFrozenConvReluImpl(graph);
}

} // namespace jit
} // namespace torch
