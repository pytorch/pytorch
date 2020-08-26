#include <torch/csrc/jit/passes/quantization/finalize.h>
#include <torch/csrc/jit/jit_log.h>
#include <torch/csrc/jit/passes/freeze_module.h>
#include <torch/csrc/jit/passes/prepack_folding.h>
#include <torch/csrc/jit/passes/quantization/freeze_quant_ops.h>
#include <torch/csrc/jit/passes/quantization/quantization_patterns.h>

namespace torch {
namespace jit {

namespace {

void insertPrepackUnpackForLinear(std::shared_ptr<Graph>& graph) {
  std::vector<QuantFusionInfo> patterns_and_replacements =
      linear_prepack_unpack_patterns();

  for (const auto& entry : patterns_and_replacements) {
    SubgraphRewriter rewriter;
    rewriter.RegisterRewritePattern(entry.pattern, entry.replacement);
    rewriter.runOnGraph(graph, entry.filters);
  }
}

void insertPrepackUnpackForConv(std::shared_ptr<Graph>& graph) {
  std::vector<QuantFusionInfo> patterns_and_replacements =
      conv_prepack_unpack_patterns();

  for (const auto& entry : patterns_and_replacements) {
    SubgraphRewriter rewriter;
    rewriter.RegisterRewritePattern(entry.pattern, entry.replacement);
    rewriter.runOnGraph(graph, entry.filters);
  }
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

Module FreezeQuantizedAttrs(Module& module) {
  auto filter_fn = [](const Node* n) -> bool {
    return (
        (n->kind() == Symbol::fromQualString("quantized::linear_prepack")) ||
        n->kind() == Symbol::fromQualString("quantized::conv1d_prepack") ||
        n->kind() == Symbol::fromQualString("quantized::conv2d_prepack") ||
        n->kind() == Symbol::fromQualString("quantized::conv3d_prepack") ||
        n->kind() == Symbol::fromQualString("quantized::embedding_bag_4bit_prepack") ||
        n->kind() == Symbol::fromQualString("quantized::embedding_bag_byte_prepack") ||
        n->kind() == Symbol::fromQualString("quantized::linear"));
  };
  return FreezeAndFoldQuantOps(module, filter_fn);
}

Module Finalize(Module& module, QuantType quant_type, bool freeze_only_quant_ops) {
  auto graph = module.get_method("forward").graph();
  InsertPrepackUnpack(graph);
  GRAPH_DUMP("Before QuantFusion:", graph);
  QuantFusion(graph, quant_type);
  Module frozen;
  if (freeze_only_quant_ops) {
    frozen = FreezeQuantizedAttrs(module);
  } else {
    frozen = freeze_module(module);
  }
  FoldQuantizedPrepackingOps(frozen);
  return frozen;
}

} // namespace jit
} // namespace torch
