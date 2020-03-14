#include <torch/csrc/jit/ir/subgraph_matcher.h>
#include <torch/csrc/jit/passes/constant_pooling.h>
#include <torch/csrc/jit/passes/constant_propagation.h>
#include <torch/csrc/jit/passes/fuse_linear.h>
#include <torch/csrc/jit/passes/graph_rewrite_helper.h>
#include <torch/csrc/jit/passes/subgraph_rewrite.h>
#include <torch/csrc/jit/passes/xnnpack_rewrite.h>

namespace torch {
namespace jit {

#ifdef USE_XNNPACK

namespace {

void insertXNNPACKLinearOp(std::shared_ptr<Graph>& graph) {
  std::string linear_before_inline = R"(
    graph(%linear, %input, %weight, %bias):
        %r = prim::CallFunction(%linear, %input, %weight, %bias)
        return (%r))";
  std::string xnnpack_pattern_before_inline = R"(
    graph(%linear, %input, %weight, %bias):
        %packed_weight_bias = _xnnpack::linear_prepack(%weight, %bias)
        %res = _xnnpack::linear_packed(%input, %packed_weight_bias)
        return (%res))";
  std::string linear_pattern = R"(
    graph(%input, %weight, %bias):
        %r = aten::linear(%input, %weight, %bias)
        return (%r))";
  std::string xnnpack_pattern = R"(
    graph(%input, %weight, %bias):
        %packed_weight_bias = _xnnpack::linear_prepack(%weight, %bias)
        %res = _xnnpack::linear_packed(%input, %packed_weight_bias)
        return (%res))";

  auto filter = [](const Match& match,
                   const std::unordered_map<std::string, Value*>& vmap) {
    const auto& match_vmap = match.values_map;
    auto linear_value = match_vmap.at(vmap.at("linear"));
    auto func_name = graph_rewrite_helper::getFuncName(linear_value);
    if (func_name == "linear") {
      return true;
    }
    return false;
  };

  SubgraphRewriter linear_call_fn_rewriter;
  linear_call_fn_rewriter.RegisterRewritePattern(
      linear_before_inline, xnnpack_pattern_before_inline);
  linear_call_fn_rewriter.runOnGraph(graph, filter);

  SubgraphRewriter linear_rewriter;
  linear_rewriter.RegisterRewritePattern(linear_pattern, xnnpack_pattern);
  linear_rewriter.runOnGraph(graph);
}

void insertXNNPACKConv2dOp(std::shared_ptr<Graph>& graph) {
  // Replace _convolution with conv2d
  graph_rewrite_helper::replaceConvolutionWithConv2d(graph);

  std::string conv_2d_pattern = R"(
    graph(%input, %weight, %bias, %stride:int[], %padding:int[], %dilation:int[], %groups:int):
        %r = aten::conv2d(%input, %weight, %bias, %stride, %padding, %dilation, %groups)
        return (%r) )";

  std::string xnnpack_conv2d_pattern = R"(
    graph(%input, %weight, %bias, %stride:int[], %padding:int[], %dilation:int[], %groups:int):
        %packed_weight_bias = _xnnpack::conv2d_prepack(%weight, %bias, %stride, %padding, %dilation, %groups)
        %r = _xnnpack::conv2d_packed(%input, %packed_weight_bias)
        return (%r) )";

  SubgraphRewriter rewriter;
  rewriter.RegisterRewritePattern(conv_2d_pattern, xnnpack_conv2d_pattern);
  rewriter.runOnGraph(graph);
}

} // namespace

void insertXNNPACKOps(std::shared_ptr<Graph>& graph) {
  ConstantPooling(graph);
  ConstantPropagation(graph);
  insertXNNPACKLinearOp(graph);
  insertXNNPACKConv2dOp(graph);
}

void insertXNNPACKOps(script::Module& module) {
  for (auto& method : module.get_methods()) {
    auto graph = method.graph();
    insertXNNPACKOps(graph);
  }
  for (script::Module m : module.children()) {
    insertXNNPACKOps(m);
  }
}

#else

void insertXNNPACKOps(std::shared_ptr<Graph>& graph) {
  TORCH_INTERNAL_ASSERT(
      "XNNPACK is not enabled. Please build with USE_XNNPACK=1");
}

void insertXNNPACKOps(script::Module& module) {
  TORCH_INTERNAL_ASSERT(
      "XNNPACK is not enabled. Please build with USE_XNNPACK=1");
}

#endif
} // namespace jit
} // namespace torch
