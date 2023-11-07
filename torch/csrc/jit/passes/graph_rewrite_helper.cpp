#include <torch/csrc/jit/passes/graph_rewrite_helper.h>

#include <torch/csrc/jit/ir/subgraph_matcher.h>
#include <torch/csrc/jit/passes/constant_propagation.h>
#include <torch/csrc/jit/passes/subgraph_rewrite.h>

namespace torch {
namespace jit {
namespace graph_rewrite_helper {

std::string getFuncName(Value* func_value) {
  auto func = func_value->type()->expectRef<FunctionType>().function();
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

static std::unordered_map<std::string, c10::IValue> getConvParams(
    const Match& match,
    const std::unordered_map<std::string, Value*>& vmap) {
  std::unordered_map<std::string, c10::IValue> calc_values;
  const auto& match_vmap = match.values_map;
  auto transposed_value = getIValue("transposed", match_vmap, vmap).value();
  calc_values["transposed"] = transposed_value;
  auto output_padding_value =
      getIValue("output_padding", match_vmap, vmap).value();
  calc_values["output_padding"] = output_padding_value;
  auto stride_value = getIValue("stride", match_vmap, vmap).value();
  calc_values["stride"] = stride_value;
  auto padding_value = getIValue("padding", match_vmap, vmap).value();
  calc_values["padding"] = padding_value;
  auto dilation_value = getIValue("dilation", match_vmap, vmap).value();
  calc_values["dilation"] = dilation_value;
  return calc_values;
}

void replaceConvolutionWithAtenConv(std::shared_ptr<Graph>& graph) {
  // TODO: remove constant prop in the pass
  ConstantPropagation(graph);
  std::string convolution_deprecated = R"(
      graph(%a, %w, %b, %stride:int[], %padding:int[], %dilation:int[],
          %transposed:bool, %output_padding:int[], %groups:int, %benchmark:bool,
          %deterministic:bool, %cudnn_enabled:bool):
        %r = aten::_convolution(%a, %w, %b, %stride, %padding, %dilation,
            %transposed, %output_padding, %groups, %benchmark, %deterministic, %cudnn_enabled)
        return (%r) )";

  std::string convolution = R"(
      graph(%a, %w, %b, %stride:int[], %padding:int[], %dilation:int[],
          %transposed:bool, %output_padding:int[], %groups:int, %benchmark:bool,
          %deterministic:bool, %cudnn_enabled:bool, %allow_tf32:bool):
        %r = aten::_convolution(%a, %w, %b, %stride, %padding, %dilation,
            %transposed, %output_padding, %groups, %benchmark, %deterministic, %cudnn_enabled, %allow_tf32)
        return (%r) )";

  std::string conv2d_for_deprecated_conv = R"(
      graph(%a, %w, %b, %stride:int[], %padding:int[], %dilation:int[],
          %transposed:bool, %output_padding:int[], %groups:int, %benchmark:bool,
          %deterministic:bool, %cudnn_enabled:bool):
        %r = aten::conv2d(%a, %w, %b, %stride, %padding, %dilation, %groups)
        return (%r) )";
  std::string conv2d = R"(
      graph(%a, %w, %b, %stride:int[], %padding:int[], %dilation:int[],
          %transposed:bool, %output_padding:int[], %groups:int, %benchmark:bool,
          %deterministic:bool, %cudnn_enabled:bool, %allow_tf32:bool):
        %r = aten::conv2d(%a, %w, %b, %stride, %padding, %dilation, %groups)
        return (%r) )";

  std::string conv1d_for_deprecated_conv = R"(
      graph(%a, %w, %b, %stride:int[], %padding:int[], %dilation:int[],
          %transposed:bool, %output_padding:int[], %groups:int, %benchmark:bool,
          %deterministic:bool, %cudnn_enabled:bool):
        %r = aten::conv1d(%a, %w, %b, %stride, %padding, %dilation, %groups)
        return (%r) )";
  std::string conv1d = R"(
      graph(%a, %w, %b, %stride:int[], %padding:int[], %dilation:int[],
          %transposed:bool, %output_padding:int[], %groups:int, %benchmark:bool,
          %deterministic:bool, %cudnn_enabled:bool, %allow_tf32:bool):
        %r = aten::conv1d(%a, %w, %b, %stride, %padding, %dilation, %groups)
        return (%r) )";

  std::string conv3d_for_deprecated_conv = R"(
      graph(%a, %w, %b, %stride:int[], %padding:int[], %dilation:int[],
          %transposed:bool, %output_padding:int[], %groups:int, %benchmark:bool,
          %deterministic:bool, %cudnn_enabled:bool):
        %r = aten::conv3d(%a, %w, %b, %stride, %padding, %dilation, %groups)
        return (%r) )";
  std::string conv3d = R"(
      graph(%a, %w, %b, %stride:int[], %padding:int[], %dilation:int[],
          %transposed:bool, %output_padding:int[], %groups:int, %benchmark:bool,
          %deterministic:bool, %cudnn_enabled:bool, %allow_tf32:bool):
        %r = aten::conv3d(%a, %w, %b, %stride, %padding, %dilation, %groups)
        return (%r) )";

  std::string conv_transpose1d_for_deprecated_conv = R"(
      graph(%a, %w, %b, %stride:int[], %padding:int[], %dilation:int[],
          %transposed:bool, %output_padding:int[], %groups:int, %benchmark:bool,
          %deterministic:bool, %cudnn_enabled:bool):
        %r = aten::conv_transpose1d(%a, %w, %b, %stride, %padding, %output_padding, %groups, %dilation)
        return (%r) )";

  std::string conv_transpose1d = R"(
      graph(%a, %w, %b, %stride:int[], %padding:int[], %dilation:int[],
          %transposed:bool, %output_padding:int[], %groups:int, %benchmark:bool,
          %deterministic:bool, %cudnn_enabled:bool, %allow_tf32:bool):
        %r = aten::conv_transpose1d(%a, %w, %b, %stride, %padding, %output_padding, %groups, %dilation)
        return (%r) )";

  std::string conv_transpose2d_for_deprecated_conv = R"(
      graph(%a, %w, %b, %stride:int[], %padding:int[], %dilation:int[],
          %transposed:bool, %output_padding:int[], %groups:int, %benchmark:bool,
          %deterministic:bool, %cudnn_enabled:bool):
        %r = aten::conv_transpose2d(%a, %w, %b, %stride, %padding, %output_padding, %groups, %dilation)
        return (%r) )";

  std::string conv_transpose2d = R"(
      graph(%a, %w, %b, %stride:int[], %padding:int[], %dilation:int[],
          %transposed:bool, %output_padding:int[], %groups:int, %benchmark:bool,
          %deterministic:bool, %cudnn_enabled:bool, %allow_tf32:bool):
        %r = aten::conv_transpose2d(%a, %w, %b, %stride, %padding, %output_padding, %groups, %dilation)
        return (%r) )";

  std::string conv_transpose3d_for_deprecated_conv = R"(
      graph(%a, %w, %b, %stride:int[], %padding:int[], %dilation:int[],
          %transposed:bool, %output_padding:int[], %groups:int, %benchmark:bool,
          %deterministic:bool, %cudnn_enabled:bool):
        %r = aten::conv_transpose3d(%a, %w, %b, %stride, %padding, %output_padding, %groups, %dilation)
        return (%r) )";

  std::string conv_transpose3d = R"(
      graph(%a, %w, %b, %stride:int[], %padding:int[], %dilation:int[],
          %transposed:bool, %output_padding:int[], %groups:int, %benchmark:bool,
          %deterministic:bool, %cudnn_enabled:bool, %allow_tf32:bool):
        %r = aten::conv_transpose3d(%a, %w, %b, %stride, %padding, %output_padding, %groups, %dilation)
        return (%r) )";

  // Filter the unsupported case
  auto filter_conv1d = [](const Match& match,
                          const std::unordered_map<std::string, Value*>& vmap) {
    auto calc_value_map = getConvParams(match, vmap);
    if (calc_value_map["output_padding"].toIntList().size() != 1 ||
        calc_value_map["stride"].toIntList().size() != 1 ||
        calc_value_map["padding"].toIntList().size() != 1 ||
        calc_value_map["dilation"].toIntList().size() != 1) {
      return false;
    }
    return !calc_value_map["transposed"].toBool();
  };
  auto filter_conv2d = [](const Match& match,
                          const std::unordered_map<std::string, Value*>& vmap) {
    auto calc_value_map = getConvParams(match, vmap);
    if (calc_value_map["output_padding"].toIntList().size() != 2 ||
        calc_value_map["stride"].toIntList().size() != 2 ||
        calc_value_map["padding"].toIntList().size() != 2 ||
        calc_value_map["dilation"].toIntList().size() != 2) {
      return false;
    }
    return !calc_value_map["transposed"].toBool();
  };
  auto filter_conv3d = [](const Match& match,
                          const std::unordered_map<std::string, Value*>& vmap) {
    auto calc_value_map = getConvParams(match, vmap);
    if (calc_value_map["output_padding"].toIntList().size() != 3 ||
        calc_value_map["stride"].toIntList().size() != 3 ||
        calc_value_map["padding"].toIntList().size() != 3 ||
        calc_value_map["dilation"].toIntList().size() != 3) {
      return false;
    }
    return !calc_value_map["transposed"].toBool();
  };
  auto filter_conv_transpose1d =
      [](const Match& match,
         const std::unordered_map<std::string, Value*>& vmap) {
        auto calc_value_map = getConvParams(match, vmap);
        if (calc_value_map["output_padding"].toIntList().size() != 1 ||
            calc_value_map["stride"].toIntList().size() != 1 ||
            calc_value_map["padding"].toIntList().size() != 1 ||
            calc_value_map["dilation"].toIntList().size() != 1) {
          return false;
        }
        return calc_value_map["transposed"].toBool();
      };
  auto filter_conv_transpose2d =
      [](const Match& match,
         const std::unordered_map<std::string, Value*>& vmap) {
        auto calc_value_map = getConvParams(match, vmap);
        if (calc_value_map["output_padding"].toIntList().size() != 2 ||
            calc_value_map["stride"].toIntList().size() != 2 ||
            calc_value_map["padding"].toIntList().size() != 2 ||
            calc_value_map["dilation"].toIntList().size() != 2) {
          return false;
        }
        return calc_value_map["transposed"].toBool();
      };
  auto filter_conv_transpose3d =
      [](const Match& match,
         const std::unordered_map<std::string, Value*>& vmap) {
        auto calc_value_map = getConvParams(match, vmap);
        if (calc_value_map["output_padding"].toIntList().size() != 3 ||
            calc_value_map["stride"].toIntList().size() != 3 ||
            calc_value_map["padding"].toIntList().size() != 3 ||
            calc_value_map["dilation"].toIntList().size() != 3) {
          return false;
        }
        return calc_value_map["transposed"].toBool();
      };

  SubgraphRewriter rewriter_conv1d;
  rewriter_conv1d.RegisterRewritePattern(convolution, conv1d);
  rewriter_conv1d.RegisterRewritePattern(
      convolution_deprecated, conv1d_for_deprecated_conv);
  rewriter_conv1d.runOnGraph(graph, filter_conv1d);

  SubgraphRewriter rewriter_conv2d;
  rewriter_conv2d.RegisterRewritePattern(convolution, conv2d);
  rewriter_conv2d.RegisterRewritePattern(
      convolution_deprecated, conv2d_for_deprecated_conv);
  rewriter_conv2d.runOnGraph(graph, filter_conv2d);

  SubgraphRewriter rewriter_conv3d;
  rewriter_conv3d.RegisterRewritePattern(convolution, conv3d);
  rewriter_conv3d.RegisterRewritePattern(
      convolution_deprecated, conv3d_for_deprecated_conv);
  rewriter_conv3d.runOnGraph(graph, filter_conv3d);

  SubgraphRewriter rewriter_conv_transpose1d;
  rewriter_conv_transpose1d.RegisterRewritePattern(
      convolution, conv_transpose1d);
  rewriter_conv_transpose1d.RegisterRewritePattern(
      convolution_deprecated, conv_transpose1d_for_deprecated_conv);
  rewriter_conv_transpose1d.runOnGraph(graph, filter_conv_transpose1d);

  SubgraphRewriter rewriter_conv_transpose2d;
  rewriter_conv_transpose2d.RegisterRewritePattern(
      convolution, conv_transpose2d);
  rewriter_conv_transpose2d.RegisterRewritePattern(
      convolution_deprecated, conv_transpose2d_for_deprecated_conv);
  rewriter_conv_transpose2d.runOnGraph(graph, filter_conv_transpose2d);

  SubgraphRewriter rewriter_conv_transpose3d;
  rewriter_conv_transpose3d.RegisterRewritePattern(
      convolution, conv_transpose3d);
  rewriter_conv_transpose3d.RegisterRewritePattern(
      convolution_deprecated, conv_transpose3d_for_deprecated_conv);
  rewriter_conv_transpose3d.runOnGraph(graph, filter_conv_transpose3d);
}

bool isClampFusable(
    const Match& match,
    const std::unordered_map<std::string, Value*>& vmap) {
  const auto& match_vmap = match.values_map;
  TORCH_CHECK(
      vmap.find("dummy_min_max") != vmap.end(),
      "Expected to find dummy_min_max Value in the subgraph to be replaced.");
  auto dummy_min_max =
      graph_rewrite_helper::getIValue("dummy_min_max", match_vmap, vmap);

  auto is_fusable = !dummy_min_max || dummy_min_max.value().isNone();

  // Also check if the output_min and output_max values are actually constant.
  // If hardtanh's min/max Value's are not actually constants, we will end up
  // rerouting those values to prepack op. And if they are not constants
  // we will not be able to remove prepacking ops.
  if (vmap.find("output_min") != vmap.end()) {
    // aten::relu pattern does not have output_min/output_max.
    // aten::hardtanh/_ does.
    TORCH_CHECK(
        vmap.find("output_max") != vmap.end(),
        "Expected to find output_max as well given "
        "output_min exist in pattern graph.");
    // If output_min/max are not constant, we get c10::nullopt.
    auto output_min =
        graph_rewrite_helper::getIValue("output_min", match_vmap, vmap);
    auto output_max =
        graph_rewrite_helper::getIValue("output_max", match_vmap, vmap);
    is_fusable =
        is_fusable && (output_min.has_value() && output_max.has_value());
  }

  return is_fusable;
}

} // namespace graph_rewrite_helper
} // namespace jit
} // namespace torch
