#include <torch/csrc/jit/ir/constants.h>
#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/ir/subgraph_matcher.h>
#include <torch/csrc/jit/passes/frozen_conv_relu_fusion.h>
#include <torch/csrc/jit/passes/subgraph_rewrite.h>

namespace torch {
namespace jit {

namespace {
void fuseFrozenConvReluImpl(std::shared_ptr<Graph>& graph) {
#ifdef USE_CUDNN
  std::string conv1d_relu = R"(
    graph(%input, %weight, %bias, %stride:int[], %padding:int[], %dilation:int[], %groups:int):
        %c = aten::conv1d(%input, %weight, %bias, %stride, %padding, %dilation, %groups)
        %res = aten::relu(%c)
        return (%res))";

  std::string conv2d_relu = R"(
    graph(%input, %weight, %bias, %stride:int[], %padding:int[], %dilation:int[], %groups:int):
        %c = aten::conv2d(%input, %weight, %bias, %stride, %padding, %dilation, %groups)
        %res = aten::relu(%c)
        return (%res))";

  std::string conv3d_relu = R"(
    graph(%input, %weight, %bias, %stride:int[], %padding:int[], %dilation:int[], %groups:int):
        %c = aten::conv3d(%input, %weight, %bias, %stride, %padding, %dilation, %groups)
        %res = aten::relu(%c)
        return (%res))";

  std::string conv_relu_fused = R"(
    graph(%input, %weight, %bias, %stride:int[], %padding:int[], %dilation:int[], %groups:int):
        %res = aten::cudnn_convolution_bias_relu(%input, %weight, %bias, %stride, %padding, %dilation, %groups)
        return (%res))";

  auto is_cuda = [](const Match& match,
                    const std::unordered_map<std::string, Value*>& vmap) {
    auto weight = toIValue(match.values_map.at(vmap.at("weight"))).value();
    return weight.toTensor().storage().data_ptr().device().is_cuda();
  };

  SubgraphRewriter rewriter;
  rewriter.RegisterRewritePattern(conv1d_relu, conv_relu_fused);
  rewriter.RegisterRewritePattern(conv2d_relu, conv_relu_fused);
  rewriter.RegisterRewritePattern(conv3d_relu, conv_relu_fused);
  rewriter.runOnGraph(graph, is_cuda);
#endif
}
} // namespace

void FuseFrozenConvRelu(std::shared_ptr<Graph>& graph) {
  fuseFrozenConvReluImpl(graph);
}

} // namespace jit
} // namespace torch
