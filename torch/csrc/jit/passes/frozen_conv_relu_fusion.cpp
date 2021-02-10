#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/ir/subgraph_matcher.h>
#include <torch/csrc/jit/passes/frozen_conv_relu_fusion.h>
#include <torch/csrc/jit/passes/subgraph_rewrite.h>

namespace torch {
namespace jit {

namespace {
void fuseFrozenConvReluImpl(std::shared_ptr<Graph>& graph) {
  SubgraphRewriter rewriter;

  std::string conv_relu = R"(
    graph(%input, %weight, %bias, %stride:int[], %padding:int[], %dilation:int[], %groups:int):
        %c = aten::conv2d(%input, %weight, %bias, %stride, %padding, %dilation, %groups)
        %res = aten::relu(%c)
        return (%res))";
  // TODO: add an operator for conv2d_relu
  std::string conv_relu_fused = R"(
    graph(%input, %weight, %bias, %stride:int[], %padding:int[], %dilation:int[], %groups:int):
        %res = aten::conv2d(%input, %weight, %bias, %stride, %padding, %dilation, %groups)
        return (%res))";
  rewriter.RegisterRewritePattern(conv_relu, conv_relu_fused);
  rewriter.runOnGraph(graph);
}
}

void FuseFrozenConvRelu(std::shared_ptr<Graph>& graph) {
  fuseFrozenConvReluImpl(graph);
}

} // namespace jit
} // namespace torch
