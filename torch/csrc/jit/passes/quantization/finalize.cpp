#include <torch/csrc/jit/passes/quantization/finalize.h>
#include <torch/csrc/jit/jit_log.h>
#include <torch/csrc/jit/passes/freeze_module.h>
#include <torch/csrc/jit/passes/graph_rewrite_helper.h>
#include <torch/csrc/jit/passes/prepack_folding.h>
#include <torch/csrc/jit/passes/quantization/quantization_patterns.h>

namespace torch {
namespace jit {

struct PatternReplaceInfo {
  std::string pattern;
  std::string replacement;
  std::vector<MatchFilter> filters = {};
};

namespace {

using graph_rewrite_helper::PatternInfo;

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
  std::string linear_fp16_with_cast = R"(
graph(%w, %a_dq, %b, %dtype_fp16, %dtype_fp32, %false):
        %fp16_tensor = aten::to(%w, %dtype_fp16, %false, %false)
        %fp32_tensor = aten::to(%fp16_tensor, %dtype_fp32, %false, %false)
        %r = aten::linear(%a_dq, %fp32_tensor, %b)
        return (%r) )";
  std::string linear_fp16_with_prepack = R"(
graph(%w, %a_dq, %b, %dtype_fp16, %dtype_fp32, %false):
        %packed_params = quantized::linear_prepack_fp16(%w, %b)
        %w_unpacked : Tensor, %b_unpacked : Tensor? = quantized::linear_unpack_fp16(%packed_params)
        %r = aten::linear(%a_dq, %w_unpacked, %b_unpacked)
        return (%r) )";

  std::vector<PatternReplaceInfo> patterns_and_replacements = {
      {linear_with_quant, linear_with_quant_prepack},
      {linear_fp16_with_cast,
       linear_fp16_with_prepack,
       {is_half_dtype, is_float_dtype, is_false_value}}};

  for (const auto& entry : patterns_and_replacements) {
    SubgraphRewriter rewriter;
    rewriter.RegisterRewritePattern(entry.pattern, entry.replacement);
    rewriter.runOnGraph(graph, entry.filters);
  }
}

void insertPrepackUnpackForConv(std::shared_ptr<Graph>& graph) {
  std::string conv1d_with_quant = R"(
graph(%a_dequant, %w_quant, %b, %stride, %padding, %dilation, %groups):
        %w_dequant = aten::dequantize(%w_quant)
        %r = aten::conv1d(%a_dequant, %w_dequant, %b, %stride, %padding, %dilation, %groups)
        return (%r) )";

  std::string conv1d_with_quant_prepack = R"(
graph(%a_dequant, %w_quant, %b, %stride, %padding, %dilation, %groups):
        %packed_params : __torch__.torch.classes.quantized.Conv2dPackedParamsBase = quantized::conv1d_prepack(%w_quant, %b, %stride, %padding, %dilation, %groups)
        %w_quant_unpacked : Tensor, %b_unpacked : Tensor? = quantized::conv1d_unpack(%packed_params)
        %w_dequant = aten::dequantize(%w_quant_unpacked)
        %r = aten::conv1d(%a_dequant, %w_dequant, %b_unpacked, %stride, %padding, %dilation, %groups)
        return (%r) )";

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

  std::vector<PatternReplaceInfo> patterns_and_replacements = {
      {conv1d_with_quant, conv1d_with_quant_prepack},
      {conv2d_with_quant, conv2d_with_quant_prepack},
      {conv3d_with_quant, conv3d_with_quant_prepack}};

  for (const auto& entry : patterns_and_replacements) {
    SubgraphRewriter rewriter;
    rewriter.RegisterRewritePattern(entry.pattern, entry.replacement);
    rewriter.runOnGraph(graph, entry.filters);
  }
}

void rewriteListAddToAppend(std::shared_ptr<Graph>& graph) {
  GRAPH_DUMP("Before restore append", graph);
  std::string list_add = R"IR(
graph(%list, %x):
    %x_list : Tensor[]  = prim::ListConstruct(%x)
    %result : Tensor[] = aten::add(%list, %x_list)
    return (%result) )IR";

  /* Rewrite the above pattern to
  std::string append = R"IR(
graph(%list, %x):
    %ignore : Tensor[] = aten::append(%list, %x)
    return (%list) )IR";
   this is not supported by subgraph rewriter, so we'll do
   this manually.
  */

  const PatternInfo& list_add_pattern_info =
      PatternInfo::parse_from_str(list_add);
  const Graph& list_add_graph = *list_add_pattern_info.pattern_graph;
  const auto& list_add_vmap = list_add_pattern_info.vmap;
  const auto& matches = findPatternMatches(list_add_graph, *graph);
  for (const auto& match : matches) {
    Value* result = match.values_map.at(list_add_vmap.at("result"));
    Node* list_add_node = result->node();
    Value* list = list_add_node->input(0);
    Value* x_list = list_add_node->input(1);

    Node* x_list_node = x_list->node();
    Value* x = x_list_node->input(0);

    result->replaceAllUsesWith(list);
    WithInsertPoint ins(list_add_node);
    Node* append_node = graph->create(Symbol::aten("append"), {list, x});
    append_node->output()->setType(ListType::ofTensors());
    graph->insertNode(append_node);
    for (Node* n : {list_add_node, x_list_node}) {
      n->removeAllInputs();
      n->destroy();
    }
  }
  GRAPH_DUMP("After restore append", graph);
}

} // namespace

void QuantFusion(std::shared_ptr<Graph>& graph, QuantType quant_type) {
  std::vector<QuantFusionInfo> patterns;
  if (quant_type == QuantType::DYNAMIC) {
    patterns = dynamic_quant_fusion_pattern_and_replacements();
  } else {
    patterns = quant_fusion_pattern_and_replacements();
  }
  for (const auto& info : patterns) {
    SubgraphRewriter rewriter;
    rewriter.RegisterRewritePattern(info.pattern, info.replacement);
    rewriter.runOnGraph(graph, info.filters);
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
        n->kind() == Symbol::fromQualString("quantized::conv1d_prepack") ||
        n->kind() == Symbol::fromQualString("quantized::conv2d_prepack") ||
        n->kind() == Symbol::fromQualString("quantized::conv3d_prepack"));
  };
  PrePackingOpsFolder(module, filter_fn, "quantized");
}

Module Finalize(Module& module, QuantType quant_type) {
  auto graph = module.get_method("forward").graph();
  GRAPH_DUMP("Before rewrite list add to append:", graph);
  rewriteListAddToAppend(graph);
  InsertPrepackUnpack(graph);
  GRAPH_DUMP("Before QuantFusion:", graph);
  QuantFusion(graph, quant_type);
  auto frozen = freeze_module(module);
  FoldQuantizedPrepackingOps(frozen);
  return frozen;
}

} // namespace jit
} // namespace torch
