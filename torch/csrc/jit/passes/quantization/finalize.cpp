#include <torch/csrc/jit/passes/quantization/finalize.h>
#include <torch/csrc/jit/passes/freeze_module.h>
#include <torch/csrc/jit/passes/prepack_folding.h>
#include <torch/csrc/jit/passes/quantization/quantization_patterns.h>

namespace torch {
namespace jit {

namespace {
void insertPrepackUnpackForLinear(std::shared_ptr<Graph>& graph) {
  std::string linear_with_quant = R"(
graph(%a_dequant, %w_quant, %b):
        %w_dequant = aten::dequantize(%w_quant)
        %r = aten::linear(%a_dequant, %w_dequant, %b)
        return (%r) )";

  std::string linear_with_quant_prepack = R"(
graph(%a_dequant, %w_quant, %b):
        %packed_params = quantized::linear_prepack(%w_quant, %b)
        %w_quant_unpacked : Tensor, %b_unpacked : Tensor? = quantized::linear_unpack(%packed_params)
        %w_dequant = aten::dequantize(%w_quant_unpacked)
        %r = aten::linear(%a_dequant, %w_dequant, %b_unpacked)
        return (%r) )";

  SubgraphRewriter rewriter;
  rewriter.RegisterRewritePattern(linear_with_quant, linear_with_quant_prepack);
  rewriter.runOnGraph(graph);
}

void insertPrepackUnpackForConv(std::shared_ptr<Graph>& graph) {
  std::string conv2d_with_quant = R"(
graph(%a_dequant, %w_quant, %b, %stride, %padding, %dilation, %groups):
        %w_dequant = aten::dequantize(%w_quant)
        %r = aten::conv2d(%a_dequant, %w_dequant, %b, %stride, %padding, %dilation, %groups)
        return (%r) )";

  std::string conv2d_with_quant_prepack = R"(
graph(%a_dequant, %w_quant, %b, %stride, %padding, %dilation, %groups):
        %packed_params : __torch__.torch.classes.quantized.Conv2dPackedParamsBase = quantized::conv2d_prepack(%w_quant, %b, %stride, %padding, %dilation, %groups)
        %w_quant_unpacked : Tensor, %b_unpacked : Tensor? = quantized::conv2d_unpack(%packed_params)
        %w_dequant = aten::dequantize(%w_quant_unpacked)
        %r = aten::conv2d(%a_dequant, %w_dequant, %b_unpacked, %stride, %padding, %dilation, %groups)
        return (%r) )";

  std::string conv3d_with_quant = R"(
graph(%a_dequant, %w_quant, %b, %stride, %padding, %dilation, %groups):
        %w_dequant = aten::dequantize(%w_quant)
        %r = aten::conv3d(%a_dequant, %w_dequant, %b, %stride, %padding, %dilation, %groups)
        return (%r) )";

  std::string conv3d_with_quant_prepack = R"(
graph(%a_dequant, %w_quant, %b, %stride, %padding, %dilation, %groups):
        %packed_params : __torch__.torch.classes.quantized.Conv3dPackedParamsBase = quantized::conv3d_prepack(%w_quant, %b, %stride, %padding, %dilation, %groups)
        %w_quant_unpacked : Tensor, %b_unpacked : Tensor? = quantized::conv3d_unpack(%packed_params)
        %w_dequant = aten::dequantize(%w_quant_unpacked)
        %r = aten::conv3d(%a_dequant, %w_dequant, %b_unpacked, %stride, %padding, %dilation, %groups)
        return (%r) )";

  std::vector<std::vector<std::string>> patterns_and_replacements = {
      {conv2d_with_quant, conv2d_with_quant_prepack},
      {conv3d_with_quant, conv3d_with_quant_prepack}};
  for (const auto& item : patterns_and_replacements) {
    SubgraphRewriter rewriter;
    const auto& pattern = item[0];
    const auto& replacement = item[1];
    rewriter.RegisterRewritePattern(pattern, replacement);
    rewriter.runOnGraph(graph);
  }
}

} // namespace

void QuantFusion(std::shared_ptr<Graph>& graph, bool is_dynamic) {
  std::vector<QuantFusionInfo> patterns;
  if (is_dynamic) {
    patterns = dynamic_quant_fusion_pattern_and_replacements();
  } else {
    patterns = quant_fusion_pattern_and_replacements();
  }
  for (const auto& info : patterns) {
    SubgraphRewriter rewriter;
    rewriter.RegisterRewritePattern(info.pattern, info.replacement);
    rewriter.runOnGraph(graph, info.filter);
  }
}

void InsertPrepackUnpack(std::shared_ptr<Graph>& graph) {
  insertPrepackUnpackForLinear(graph);
  insertPrepackUnpackForConv(graph);
}

void InsertPrepackUnpack(Module& module) {
  for (auto& method : module.get_methods()) {
    auto graph = method.graph();
    InsertPrepackUnpack(graph);
  }
  for (Module m : module.children()) {
    InsertPrepackUnpack(m);
  }
}

void FoldQuantizedPrepackingOps(Module& module) {
  auto filter_fn = [](const Node* n) -> bool {
    return (
        (n->kind() == Symbol::fromQualString("quantized::linear_prepack")) ||
        n->kind() == Symbol::fromQualString("quantized::conv2d_prepack") ||
        n->kind() == Symbol::fromQualString("quantized::conv3d_prepack"));
  };
  PrePackingOpsFolder(module, filter_fn, "quantized");
}

Module Finalize(Module& module, bool is_dynamic) {
  auto graph = module.get_method("forward").graph();
  InsertPrepackUnpack(graph);
  QuantFusion(graph, is_dynamic);
  auto frozen = freeze_module(module);
  FoldQuantizedPrepackingOps(frozen);
  return frozen;
}

} // namespace jit
} // namespace torch
