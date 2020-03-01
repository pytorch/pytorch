#include <torch/csrc/jit/passes/constant_pooling.h>
#include <torch/csrc/jit/passes/constant_propagation.h>
#include <torch/csrc/jit/passes/xnnpack_rewrite.h>
#include <torch/csrc/jit/passes/fuse_linear.h>
#include <torch/csrc/jit/passes/graph_rewrite_helper.h>
#include <torch/csrc/jit/passes/subgraph_rewrite.h>
#include <torch/csrc/jit/ir/subgraph_matcher.h>

namespace torch {
namespace jit {

namespace {
std::string getFuncName(Value* func_value) {
  auto func_node = func_value->node();
  auto func = func_node->output()->type()->expect<FunctionType>()->function();
  const auto& qname = func->qualname();
  const auto& name = qname.qualifiedName();
  auto rdot_idx = name.rfind('.');
  if (rdot_idx != std::string::npos) {
    return name.substr(rdot_idx + 1, name.length());
  } else {
    return name;
  }
}

void insertXNNPACKLinearOp(std::shared_ptr<Graph>& graph) {
  std::string linear_before_inline = R"(
    graph(%linear, %input, %weight, %bias):
        %r = prim::CallFunction(%linear, %input, %weight, %bias)
        return (%r))";
  std::string xnnpack_pattern_before_inline = R"(
    graph(%linear, %input, %weight, %bias):
        %packed_weight_bias = xnnpack::linear_prepack(%weight, %bias)
        %res = xnnpack::linear_packed(%input, %packed_weight_bias)
        return (%res))";
  std::string linear_pattern = R"(
    graph(%input, %weight, %bias):
        %r = aten::linear(%input, %weight, %bias)
        return (%r))";
  std::string xnnpack_pattern = R"(
    graph(%input, %weight, %bias):
        %packed_weight_bias = xnnpack::linear_prepack(%weight, %bias)
        %res = xnnpack::linear_packed(%input, %packed_weight_bias)
        return (%res))";
  std::string linear_pattern_wo_bias = R"(
    graph(%input, %weight):
        %bias: Tensor? = prim::Constant()
        %res = aten::linear(%input, %weight, %bias)
        return (%res))";
  std::string xnnpack_pattern_wo_bias = R"(
    graph(%input, %weight):
        %bias: Tensor? = prim::Constant()
        %packed_weight_bias = xnnpack::linear_prepack(%weight, %bias)
        %res = xnnpack::linear_packed(%input, %packed_weight_bias)
        return (%res))";

  auto filter = [](const Match& match,
                   const std::unordered_map<std::string, Value*>& vmap) {
    const auto& match_vmap = match.values_map;
    auto linear_value = match_vmap.at(vmap.at("linear"));
    auto func_name = getFuncName(linear_value);
    if (func_name == "linear") {
      return true;
    }
    return false;
  };

  SubgraphRewriter linear_call_fn_rewriter;
  linear_call_fn_rewriter.RegisterRewritePattern(linear_before_inline, xnnpack_pattern_before_inline);
  linear_call_fn_rewriter.runOnGraph(graph, filter);

  SubgraphRewriter linear_rewriter;
  linear_rewriter.RegisterRewritePattern(linear_pattern, xnnpack_pattern);
  linear_rewriter.runOnGraph(graph);

  SubgraphRewriter linear_rewriter_wo_bias;
  linear_rewriter_wo_bias.RegisterRewritePattern(linear_pattern_wo_bias, xnnpack_pattern_wo_bias);
  linear_rewriter_wo_bias.runOnGraph(graph);
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
        %packed_weight_bias = xnnpack::conv2d_prepack(%weight, %bias, %stride, %padding, %dilation, %groups)
        %r = xnnpack::conv2d_packed(%input, %packed_weight_bias)
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
} // namespace jit
} // namespace torch
