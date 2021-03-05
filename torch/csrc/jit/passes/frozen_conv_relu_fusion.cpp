#include <torch/csrc/jit/ir/constants.h>
#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/ir/subgraph_matcher.h>
#include <torch/csrc/jit/passes/frozen_conv_relu_fusion.h>
#include <torch/csrc/jit/passes/graph_rewrite_helper.h>
#include <torch/csrc/jit/passes/remove_mutation.h>
#include <torch/csrc/jit/passes/subgraph_rewrite.h>

#include <torch/csrc/jit/frontend/code_template.h>

namespace torch {
namespace jit {

namespace {
void fuseFrozenConvReluImpl(std::shared_ptr<Graph>& graph) {
#ifdef USE_CUDNN
  SubgraphRewriter rewriter;

  std::string conv_operators[] = {"conv1d", "conv2d", "conv3d"};

  auto conv_relu_rstring = CodeTemplate(R"(
    graph(%input, %weight, %bias, %stride:int[], %padding:int[], %dilation:int[], %groups:int):
      %x = aten::${conv}(%input, %weight, %bias, %stride, %padding, %dilation, %groups)
      %res = aten::relu(%x)
      return (%res))");

  std::string conv_relu_fused = R"(
    graph(%input, %weight, %bias, %stride:int[], %padding:int[], %dilation:int[], %groups:int):
        %res = prim::cudnn_convolution_add_relu(%input, %weight, %bias, %stride, %padding, %dilation, %groups)
        return (%res))";

  auto conv_add_relu_rstring = CodeTemplate(R"(
    graph(%input, %weight, %bias, %z, %alpha, %stride:int[], %padding:int[], %dilation:int[], %groups:int):
      %x = aten::${conv}(%input, %weight, %bias, %stride, %padding, %dilation, %groups)
      %y = aten::add(%x, %z, %alpha)
      %res = aten::relu(%y)
      return (%res))");

  std::string conv_add_relu_fused = R"(
    graph(%input, %weight, %bias, %z, %alpha, %stride:int[], %padding:int[], %dilation:int[], %groups:int):
        %res = prim::cudnn_convolution_add_relu(%input, %weight, %bias, %z, %alpha, %stride, %padding, %dilation, %groups)
        return (%res))";

  for (auto conv : conv_operators) {
    TemplateEnv env;
    env.s("conv", conv);
    rewriter.RegisterRewritePattern(
        conv_relu_rstring.format(env), conv_relu_fused);
    rewriter.RegisterRewritePattern(
        conv_add_relu_rstring.format(env), conv_add_relu_fused);
  }

  auto is_cuda = [](const Match& match,
                    const std::unordered_map<std::string, Value*>& vmap) {
    // auto input = toIValue(match.values_map.at(vmap.at("input"))).value();
    // if (input.toTensor().suggest_memory_format() !=
    // at::MemoryFormat::ChannelsLast)
    //  return false;
    auto weight = toIValue(match.values_map.at(vmap.at("weight"))).value();
    return weight.toTensor().storage().data_ptr().device().is_cuda();
  };

  // Convert _convolution and in-place operators for simpler replacement pattern
  // matching
  graph_rewrite_helper::replaceConvolutionWithAtenConv(graph);
  RemoveTensorMutation(graph, [](Node* node) {
    return node->kind() == aten::add_ || node->kind() == aten::relu_;
  });

  rewriter.runOnGraph(graph, is_cuda);
#endif
}
} // namespace

void FuseFrozenConvRelu(std::shared_ptr<Graph>& graph) {
  fuseFrozenConvReluImpl(graph);
}

} // namespace jit
} // namespace torch
