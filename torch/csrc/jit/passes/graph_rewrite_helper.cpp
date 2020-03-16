#include <torch/csrc/jit/ir/subgraph_matcher.h>
#include <torch/csrc/jit/passes/graph_rewrite_helper.h>
#include <torch/csrc/jit/passes/subgraph_rewrite.h>

namespace torch {
namespace jit {
namespace graph_rewrite_helper {

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

Value* getValue(
    const std::string& name,
    const std::unordered_map<const Value*, Value*>& match_vmap,
    const std::unordered_map<std::string, Value*>& vmap) {
  return match_vmap.at(vmap.at(name));
}

c10::optional<IValue> getIValue(
    const std::string& name,
    const std::unordered_map<const Value*, Value*>& match_vmap,
    const std::unordered_map<std::string, Value*>& vmap) {
  return toIValue(getValue(name, match_vmap, vmap));
}

void replaceConvolutionWithConv2d(std::shared_ptr<Graph>& graph) {
  std::string convolution = R"(
      graph(%a, %w, %b, %stride:int[], %padding:int[], %dilation:int[],
          %transposed:bool, %output_padding:int[], %groups:int, %benchmark:bool,
          %deterministic:bool, %cudnn_enabled:bool):
        %r = aten::_convolution(%a, %w, %b, %stride, %padding, %dilation,
            %transposed, %output_padding, %groups, %benchmark, %deterministic, %cudnn_enabled)
        return (%r) )";

  std::string conv2d = R"(
      graph(%a, %w, %b, %stride:int[], %padding:int[], %dilation:int[],
          %transposed:bool, %output_padding:int[], %groups:int, %benchmark:bool,
          %deterministic:bool, %cudnn_enabled:bool):
        %r = aten::conv2d(%a, %w, %b, %stride, %padding, %dilation, %groups)
        return (%r) )";

  // Filter the unsupported case
  auto filter = [](const Match& match,
                   const std::unordered_map<std::string, Value*>& vmap) {
    const auto& match_vmap = match.values_map;
    auto transposed_value =
        getIValue("transposed", match_vmap, vmap).value().toBool();
    auto benchmark_value =
        getIValue("benchmark", match_vmap, vmap).value().toBool();
    auto deterministic_value =
        getIValue("deterministic", match_vmap, vmap).value().toBool();
    auto cudnn_enabled_value =
        getIValue("cudnn_enabled", match_vmap, vmap).value().toBool();
    auto output_padding_value =
        getIValue("output_padding", match_vmap, vmap).value().toIntList();

    if (!transposed_value && !benchmark_value && !deterministic_value &&
        cudnn_enabled_value && (output_padding_value[0] == 0) &&
        (output_padding_value[1] == 0)) {
      return true;
    }
    return false;
  };

  SubgraphRewriter rewriter;
  rewriter.RegisterRewritePattern(convolution, conv2d);
  rewriter.runOnGraph(graph, filter);
}

} // namespace graph_rewrite_helper
} // namespace jit
} // namespace torch
