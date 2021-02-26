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
  SubgraphRewriter rewriter;

  std::string operators[][2] = {
    {"conv1d", "relu"},
    {"conv2d", "relu"},
    {"conv3d", "relu"},
    {"conv1d", "relu_"},
    {"conv2d", "relu_"},
    {"conv3d", "relu_"}
  };

  for (auto op : operators) {
    std::string conv_relu = R"(
      graph(%input, %weight, %bias, %stride:int[], %padding:int[], %dilation:int[], %groups:int):
          %c = aten::)" + op[0] + R"((%input, %weight, %bias, %stride, %padding, %dilation, %groups)
          %res = aten::)" + op[1] + R"((%c)
          return (%res))";
    std::string conv_relu_fused = R"(
      graph(%input, %weight, %bias, %stride:int[], %padding:int[], %dilation:int[], %groups:int):
          %res = aten::cudnn_convolution_bias_relu(%input, %weight, %bias, %stride, %padding, %dilation, %groups)
          return (%res))";
      rewriter.RegisterRewritePattern(conv_relu, conv_relu_fused);
  }

  auto is_cuda = [](const Match& match,
                    const std::unordered_map<std::string, Value*>& vmap) {
    auto weight = toIValue(match.values_map.at(vmap.at("weight"))).value();
    return weight.toTensor().storage().data_ptr().device().is_cuda();
  };
  rewriter.runOnGraph(graph, is_cuda);
#endif
}
} // namespace

void FuseFrozenConvRelu(std::shared_ptr<Graph>& graph) {
  fuseFrozenConvReluImpl(graph);
}

} // namespace jit
} // namespace torch
