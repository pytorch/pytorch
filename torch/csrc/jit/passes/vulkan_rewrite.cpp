#include <ATen/core/jit_type.h>
#ifdef USE_VULKAN
#include <ATen/native/vulkan/VulkanOpContext.h>
#endif

#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/ir/subgraph_matcher.h>
#include <torch/csrc/jit/passes/constant_pooling.h>
#include <torch/csrc/jit/passes/fold_conv_bn.h>
#include <torch/csrc/jit/passes/freeze_module.h>
#include <torch/csrc/jit/passes/fuse_linear.h>
#include <torch/csrc/jit/passes/graph_rewrite_helper.h>
#include <torch/csrc/jit/passes/prepack_folding.h>
#include <torch/csrc/jit/passes/remove_dropout.h>
#include <torch/csrc/jit/passes/subgraph_rewrite.h>
#include <torch/csrc/jit/passes/vulkan_rewrite.h>

namespace torch {
namespace jit {

#ifdef USE_VULKAN

namespace {

void insertPrePackedLinearOp(std::shared_ptr<Graph>& graph) {
  // fuse decomposed linear into aten::linear
  FuseLinear(graph);

  std::string linear_before_inline = R"(
    graph(%linear, %input, %weight, %bias):
        %r = prim::CallFunction(%linear, %input, %weight, %bias)
        return (%r))";
  std::string prepacked_ops_pattern_before_inline = R"(
    graph(%linear, %input, %weight, %bias):
        %weight_t = aten::t(%weight)
        %packed_weight_bias = vulkan_prepack::linear_prepack(
            %weight_t, %bias)
        %res = vulkan_prepack::linear_run(%input, %packed_weight_bias)
        return (%res))";
  std::string linear_pattern = R"(
    graph(%input, %weight, %bias):
        %r = aten::linear(%input, %weight, %bias)
        return (%r))";
  std::string prepacked_ops_pattern = R"(
    graph(%input, %weight, %bias):
        %weight_t = aten::t(%weight)
        %packed_weight_bias = vulkan_prepack::linear_prepack(
            %weight_t, %bias)
        %res = vulkan_prepack::linear_run(%input, %packed_weight_bias)
        return (%res))";

  const auto filter = [](const Match& match,
                         const std::unordered_map<std::string, Value*>& vmap) {
    const auto& match_vmap = match.values_map;
    const auto linear_value = match_vmap.at(vmap.at("linear"));
    const auto func_name = graph_rewrite_helper::getFuncName(linear_value);
    return (func_name == "linear");
  };

  SubgraphRewriter linear_call_fn_rewriter;
  linear_call_fn_rewriter.RegisterRewritePattern(
      linear_before_inline, prepacked_ops_pattern_before_inline);
  linear_call_fn_rewriter.runOnGraph(graph, filter);

  SubgraphRewriter linear_rewriter;
  linear_rewriter.RegisterRewritePattern(linear_pattern, prepacked_ops_pattern);
  linear_rewriter.runOnGraph(graph);
}

void insertPrePackedConv2dOp(std::shared_ptr<Graph>& graph) {
  graph_rewrite_helper::replaceConvolutionWithAtenConv(graph);

  std::string conv_2d_pattern = R"(
    graph(%input, %weight, %bias, %stride:int[], %padding:int[], %dilation:int[], %groups:int):
        %r = aten::conv2d(%input, %weight, %bias, %stride, %padding, %dilation, %groups)
        return (%r) )";

  std::string prepacked_ops_conv2d_pattern = R"(
    graph(%input, %weight, %bias, %stride:int[], %padding:int[], %dilation:int[], %groups:int):
        %output_min_max : None = prim::Constant()
        %packed_weight_bias = vulkan_prepack::conv2d_clamp_prepack(
            %weight, %bias, %stride, %padding, %dilation, %groups,
            %output_min_max, %output_min_max)
        %r = vulkan_prepack::conv2d_clamp_run(%input, %packed_weight_bias)
        return (%r) )";

  SubgraphRewriter rewriter;
  rewriter.RegisterRewritePattern(
      conv_2d_pattern, prepacked_ops_conv2d_pattern);
  rewriter.runOnGraph(graph);
}

void fuseHardtanhWithPackedOps(std::shared_ptr<Graph>& graph) {
  SubgraphRewriter rewriter;

  std::string conv2d_prepack_run_hardtanh_fused = R"(
    graph(%input, %weight, %bias, %stride:int[], %padding:int[],
          %dilation:int[], %groups:int, %output_min, %output_max, %dummy_min_max):
        %packed_weight_bias : __torch__.torch.classes.vulkan.Conv2dOpContext = vulkan_prepack::conv2d_clamp_prepack(
            %weight, %bias, %stride, %padding, %dilation, %groups,
            %output_min, %output_max)
        %r = vulkan_prepack::conv2d_clamp_run(%input, %packed_weight_bias)
        return (%r) )";

  std::string conv2d_prepack_run_hardtanh = R"(
    graph(%input, %weight, %bias, %stride:int[], %padding:int[],
          %dilation:int[], %groups:int, %output_min, %output_max, %dummy_min_max):
        %packed_weight_bias = vulkan_prepack::conv2d_clamp_prepack(
            %weight, %bias, %stride, %padding, %dilation, %groups,
            %dummy_min_max, %dummy_min_max)
        %conv2d_res = vulkan_prepack::conv2d_clamp_run(%input, %packed_weight_bias)
        %r = aten::hardtanh(%conv2d_res, %output_min, %output_max)
        return (%r) )";

  rewriter.RegisterRewritePattern(
      conv2d_prepack_run_hardtanh, conv2d_prepack_run_hardtanh_fused);

  std::string conv2d_prepack_run_hardtanh_inplace = R"(
    graph(%input, %weight, %bias, %stride:int[], %padding:int[],
          %dilation:int[], %groups:int, %output_min, %output_max, %dummy_min_max):
        %packed_weight_bias = vulkan_prepack::conv2d_clamp_prepack(
            %weight, %bias, %stride, %padding, %dilation, %groups,
            %dummy_min_max, %dummy_min_max)
        %conv2d_res = vulkan_prepack::conv2d_clamp_run(%input, %packed_weight_bias)
        %r = aten::hardtanh_(%conv2d_res, %output_min, %output_max)
        return (%r) )";

  rewriter.RegisterRewritePattern(
      conv2d_prepack_run_hardtanh_inplace, conv2d_prepack_run_hardtanh_fused);

  rewriter.runOnGraph(graph, torch::jit::graph_rewrite_helper::isClampFusable);
}

void fuseReluWithPackedOps(std::shared_ptr<Graph>& graph) {
  SubgraphRewriter rewriter;

  std::string conv2d_prepack_run_relu_fused = R"(
    graph(%input, %weight, %bias, %stride:int[], %padding:int[],
          %dilation:int[], %groups:int, %dummy_min_max):
        %output_min: float = prim::Constant[value=0.0]()
        %output_max: None = prim::Constant()
        %packed_weight_bias : __torch__.torch.classes.vulkan.Conv2dOpContext = vulkan_prepack::conv2d_clamp_prepack(
            %weight, %bias, %stride, %padding, %dilation, %groups,
            %output_min, %output_max)
        %r = vulkan_prepack::conv2d_clamp_run(%input, %packed_weight_bias)
        return (%r) )";

  std::string conv2d_prepack_run_relu = R"(
    graph(%input, %weight, %bias, %stride:int[], %padding:int[],
          %dilation:int[], %groups:int, %dummy_min_max):
        %packed_weight_bias = vulkan_prepack::conv2d_clamp_prepack(
            %weight, %bias, %stride, %padding, %dilation, %groups,
            %dummy_min_max, %dummy_min_max)
        %conv2d_res = vulkan_prepack::conv2d_clamp_run(%input, %packed_weight_bias)
        %r = aten::relu(%conv2d_res)
        return (%r) )";

  rewriter.RegisterRewritePattern(
      conv2d_prepack_run_relu, conv2d_prepack_run_relu_fused);

  std::string conv2d_prepack_run_relu_inplace = R"(
    graph(%input, %weight, %bias, %stride:int[], %padding:int[],
          %dilation:int[], %groups:int, %dummy_min_max):
        %packed_weight_bias = vulkan_prepack::conv2d_clamp_prepack(
            %weight, %bias, %stride, %padding, %dilation, %groups,
            %dummy_min_max, %dummy_min_max)
        %conv2d_res = vulkan_prepack::conv2d_clamp_run(%input, %packed_weight_bias)
        %r = aten::relu_(%conv2d_res)
        return (%r) )";

  rewriter.RegisterRewritePattern(
      conv2d_prepack_run_relu_inplace, conv2d_prepack_run_relu_fused);
  rewriter.runOnGraph(graph, torch::jit::graph_rewrite_helper::isClampFusable);
}

} // namespace

void vulkanInsertPrePackedOps(std::shared_ptr<Graph>& graph) {
  insertPrePackedLinearOp(graph);
  insertPrePackedConv2dOp(graph);
}

void vulkanInsertPrePackedOps(script::Module& module) {
  for (auto& method : module.get_methods()) {
    auto graph = method.graph();
    vulkanInsertPrePackedOps(graph);
  }
  for (script::Module m : module.children()) {
    vulkanInsertPrePackedOps(m);
  }
}

void vulkanFusePrePackedConvWithClamp(script::Module& module) {
  auto graph = module.get_method("forward").graph();
  fuseReluWithPackedOps(graph);
  fuseHardtanhWithPackedOps(graph);
}

void vulkanFoldPrePackingOps(script::Module& m) {
  PrePackingOpsFilterFn filter_fn = [](const Node* n) -> bool {
    return (
        (n->kind() ==
         Symbol::fromQualString("vulkan_prepack::conv2d_clamp_prepack")) ||
        (n->kind() ==
         Symbol::fromQualString("vulkan_prepack::linear_prepack")));
  };
  PrePackingOpsFolder(m, filter_fn, "prepack_folding");
}

script::Module vulkanOptimizeForMobile(
    const script::Module& m,
    const std::vector<std::string>& preserved_methods) {
  auto cloned_module = m.clone();
  cloned_module.eval();
  cloned_module = FoldConvBatchNorm(cloned_module);
  vulkanInsertPrePackedOps(cloned_module);
  cloned_module = freeze_module(cloned_module, preserved_methods);
  vulkanFusePrePackedConvWithClamp(cloned_module);
  vulkanFoldPrePackingOps(cloned_module);
  removeDropout(cloned_module);
  return cloned_module;
}

#else

void vulkanInsertPrePackedOps(std::shared_ptr<Graph>& graph) {
  TORCH_INTERNAL_ASSERT(
      "Vulkan is not enabled. Please build with USE_VULKAN=1");
}

void vulkanInsertPrePackedOps(script::Module& module) {
  TORCH_INTERNAL_ASSERT(
      "Vulkan is not enabled. Please build with USE_VULKAN=1");
}

void vulkanFusePrePackedConvWithClamp(script::Module& module) {
  TORCH_INTERNAL_ASSERT(
      "Vulkan is not enabled. Please build with USE_VULKAN=1");
}

void vulkanFoldPrePackingOps(script::Module& m) {
  TORCH_INTERNAL_ASSERT(
      "Vulkan is not enabled. Please build with USE_VULKAN=1");
}

script::Module vulkanOptimizeForMobile(
    const script::Module& module,
    const std::vector<std::string>& preserved_methods) {
  TORCH_INTERNAL_ASSERT(
      "Mobile optimizaiton only available with Vulkan at the moment. "
      "Vulkan is not enabled. Please build with USE_VULKAN=1");
  return module;
}

#endif
} // namespace jit
} // namespace torch
