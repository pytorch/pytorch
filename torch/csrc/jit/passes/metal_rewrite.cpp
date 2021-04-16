#include <ATen/core/jit_type.h>

#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/ir/subgraph_matcher.h>
#include <torch/csrc/jit/passes/constant_pooling.h>
#include <torch/csrc/jit/passes/fold_conv_bn.h>
#include <torch/csrc/jit/passes/freeze_module.h>
#include <torch/csrc/jit/passes/fuse_linear.h>
#include <torch/csrc/jit/passes/graph_rewrite_helper.h>
#include <torch/csrc/jit/passes/metal_rewrite.h>
#include <torch/csrc/jit/passes/prepack_folding.h>
#include <torch/csrc/jit/passes/remove_dropout.h>
#include <torch/csrc/jit/passes/subgraph_rewrite.h>
#include <torch/csrc/jit/runtime/graph_executor_impl.h>

namespace torch {
namespace jit {

namespace {

void insertPrePackedConv2dOp(std::shared_ptr<Graph>& graph) {
  graph_rewrite_helper::replaceConvolutionWithAtenConv(graph);

  std::string conv_2d_pattern = R"(
    graph(%input, %weight, %bias, %stride:int[], %padding:int[], %dilation:int[], %groups:int):
        %r = aten::conv2d(%input, %weight, %bias, %stride, %padding, %dilation, %groups)
        return (%r) )";

  std::string prepacked_ops_conv2d_pattern = R"(
    graph(%input, %weight, %bias, %stride:int[], %padding:int[],
          %dilation:int[], %groups:int):
        %output_min_max : None = prim::Constant()
        %packed_weight_bias = metal_prepack::conv2d_prepack(
            %weight, %bias, %stride, %padding, %dilation, %groups,
            %output_min_max, %output_min_max)
        %r = metal_prepack::conv2d_run(%input, %packed_weight_bias)
        return (%r) )";

  SubgraphRewriter rewriter;
  rewriter.RegisterRewritePattern(
      conv_2d_pattern, prepacked_ops_conv2d_pattern);
  rewriter.runOnGraph(graph);
}

void fuseReluWithPackedOps(std::shared_ptr<Graph>& graph) {
  SubgraphRewriter rewriter;

  std::string conv2d_prepack_run_relu = R"(
    graph(%input, %weight, %bias, %stride:int[], %padding:int[],
          %dilation:int[], %groups:int, %dummy_min_max):
        %packed_weight_bias = metal_prepack::conv2d_prepack(
            %weight, %bias, %stride, %padding, %dilation, %groups,
            %dummy_min_max, %dummy_min_max)
        %r = metal_prepack::conv2d_run(%input, %packed_weight_bias)
        %r = aten::relu(%r)
        return (%r) )";

  std::string conv2d_prepack_run_relu_fused = R"(
  graph(%input, %weight, %bias, %stride:int[], %padding:int[],
        %dilation:int[], %groups:int, %dummy_min_max):
      %output_min: float = prim::Constant[value=0.0]()
      %output_max: None = prim::Constant()
      %packed_weight_bias: __torch__.torch.classes.metal.Conv2dOpContext = metal_prepack::conv2d_prepack(
          %weight, %bias, %stride, %padding, %dilation, %groups,
          %output_min, %output_max)
      %r = metal_prepack::conv2d_run(%input, %packed_weight_bias)
      return (%r) )";

  rewriter.RegisterRewritePattern(
      conv2d_prepack_run_relu, conv2d_prepack_run_relu_fused);

  std::string conv2d_prepack_run_relu_inplace = R"(
  graph(%input, %weight, %bias, %stride:int[], %padding:int[],
        %dilation:int[], %groups:int, %dummy_min_max):
      %packed_weight_bias = metal_prepack::conv2d_prepack(
          %weight, %bias, %stride, %padding, %dilation, %groups,
          %dummy_min_max, %dummy_min_max)
      %r = metal_prepack::conv2d_run(%input, %packed_weight_bias)
      %r = aten::relu_(%r)
      return (%r) )";

  rewriter.RegisterRewritePattern(
      conv2d_prepack_run_relu_inplace, conv2d_prepack_run_relu_fused);

  rewriter.runOnGraph(graph, torch::jit::graph_rewrite_helper::isClampFusable);
}

void fuseHardtanhWithPackedOps(std::shared_ptr<Graph>& graph) {
  SubgraphRewriter rewriter;

  std::string conv2d_prepack_run_hardtanh_fused = R"(
    graph(%input, %weight, %bias, %stride:int[], %padding:int[],
          %dilation:int[], %groups:int, %output_min, %output_max, %dummy_min_max):
        %packed_weight_bias: __torch__.torch.classes.metal.Conv2dOpContext = metal_prepack::conv2d_prepack(
            %weight, %bias, %stride, %padding, %dilation, %groups,
            %output_min, %output_max)
        %r = metal_prepack::conv2d_run(%input, %packed_weight_bias)
        return (%r) )";

  std::string conv2d_prepack_run_hardtanh = R"(
    graph(%input, %weight, %bias, %stride:int[], %padding:int[],
          %dilation:int[], %groups:int, %output_min, %output_max, %dummy_min_max):
        %packed_weight_bias = metal_prepack::conv2d_prepack(
            %weight, %bias, %stride, %padding, %dilation, %groups,
            %dummy_min_max, %dummy_min_max)
        %r = metal_prepack::conv2d_run(%input, %packed_weight_bias)
        %r = aten::hardtanh(%r, %output_min, %output_max)
        return (%r) )";

  rewriter.RegisterRewritePattern(
      conv2d_prepack_run_hardtanh, conv2d_prepack_run_hardtanh_fused);

  std::string conv2d_prepack_run_hardtanh_inplace = R"(
    graph(%input, %weight, %bias, %stride:int[], %padding:int[],
          %dilation:int[], %groups:int, %output_min, %output_max, %dummy_min_max):
        %packed_weight_bias = metal_prepack::conv2d_prepack(
            %weight, %bias, %stride, %padding, %dilation, %groups,
            %dummy_min_max, %dummy_min_max)
        %r = metal_prepack::conv2d_run(%input, %packed_weight_bias)
        %r = aten::hardtanh_(%r, %output_min, %output_max)
        return (%r) )";

  rewriter.RegisterRewritePattern(
      conv2d_prepack_run_hardtanh_inplace, conv2d_prepack_run_hardtanh_fused);

  rewriter.runOnGraph(graph, torch::jit::graph_rewrite_helper::isClampFusable);
}

} // namespace

void metalInsertPrePackedOps(std::shared_ptr<Graph>& graph) {
  insertPrePackedConv2dOp(graph);
}

void metalInsertPrePackedOps(script::Module& module) {
  for (auto& method : module.get_methods()) {
    auto graph = method.graph();
    metalInsertPrePackedOps(graph);
  }
  for (script::Module m : module.children()) {
    metalInsertPrePackedOps(m);
  }
}

void metalFoldPrePackingOps(script::Module& m) {
  PrePackingOpsFilterFn filter_fn = [](const Node* n) -> bool {
    return (
        n->kind() == Symbol::fromQualString("metal_prepack::conv2d_prepack"));
  };
  PrePackingOpsFolder(m, filter_fn, "prepack_folding");
}

void metalFusePrePackedConvWithClamp(script::Module& module) {
  auto graph = module.get_method("forward").graph();
  fuseReluWithPackedOps(graph);
  fuseHardtanhWithPackedOps(graph);
}

void metalInsertCopyOps(script::Module& module) {
  auto graph = module.get_method("forward").graph();
  auto&& outputs = graph->outputs();
  for (size_t i = 0; i < outputs.size(); ++i) {
    Value* output = outputs[i];
    auto namedValue = NamedValue("", output);
    if (namedValue.type()->kind() == TypeKind::TensorType) {
      // find the insertion point
      WithInsertPoint ip(output->node()->next());
      Value* replaced_output = graph->insert(
          Symbol::fromQualString("metal::copy_to_host"), {namedValue});
      // replaced the output
      graph->block()->replaceOutput(i, replaced_output);
    }
  }
  SubgraphRewriter rewriter;
  rewriter.runOnGraph(graph);
}

void runCanonicalOptimizations(script::Module& module) {
  auto graph = module.get_method("forward").graph();
  runOptimization(graph, false /* no loop unrolling */);
}

script::Module metalOptimizeForMobile(
    const script::Module& m,
    const std::vector<std::string>& preserved_methods) {
  auto cloned_module = m.clone();
  cloned_module.eval();
  cloned_module = FoldConvBatchNorm(cloned_module);
  metalInsertPrePackedOps(cloned_module);
  cloned_module = freeze_module(cloned_module, preserved_methods);
  metalFusePrePackedConvWithClamp(cloned_module);
  metalFoldPrePackingOps(cloned_module);
  metalInsertCopyOps(cloned_module);
  removeDropout(cloned_module);
  // remove duplicated constants
  runCanonicalOptimizations(cloned_module);
  cloned_module.register_attribute(
      "optimized_for_metal", BoolType::get(), true);
  return cloned_module;
}

} // namespace jit
} // namespace torch
