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
#include <torch/csrc/jit/passes/remove_mutation.h>
#include <torch/csrc/jit/passes/subgraph_rewrite.h>
#include <torch/csrc/jit/passes/vulkan_rewrite.h>
#include <torch/csrc/jit/runtime/graph_executor_impl.h>

namespace torch {
namespace jit {

#ifdef USE_VULKAN

namespace {

void insertPrePackedLinearOp(std::shared_ptr<Graph>& graph) {
  // fuse decomposed linear into aten::linear
  FuseLinear(graph);

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

  SubgraphRewriter linear_rewriter;
  linear_rewriter.RegisterRewritePattern(linear_pattern, prepacked_ops_pattern);
  linear_rewriter.runOnGraph(graph);
}

void insertPrePackedMcLarenEncoderBlockOp(std::shared_ptr<Graph>& graph) {
  graph_rewrite_helper::replaceConvolutionWithAtenConv(graph);

  std::string conv_2d_pattern = R"(
    graph(%dim:int,
          %input1, %weight1, %bias1, %stride1:int[], %padding1:int[], %dilation1:int[], %groups1:int,
          %input2, %weight2, %bias2, %stride2:int[], %padding2:int[], %dilation2:int[], %groups2:int):
        %list : Tensor[] = prim::ListConstruct(%input1, %input2)
        %input = aten::cat(%list, %dim)
        %r1 = aten::conv2d(%input, %weight1, %bias1, %stride1, %padding1, %dilation1, %groups1)
        %r2 = aten::conv2d(%input, %weight2, %bias2, %stride2, %padding2, %dilation2, %groups2)
        %r3 = aten::sigmoid(%r2)
        %r = aten::mul(%r1, %r3)
        return (%r) )";

  std::string prepacked_ops_conv2d_pattern = R"(
    graph(%dim:int, %input1, %weight1, %bias1, %stride1:int[], %padding1:int[], %dilation1:int[], %groups1:int,
          %input2, %weight2, %bias2, %stride2:int[], %padding2:int[], %dilation2:int[], %groups2:int):
        %transposed : bool = prim::Constant[value=0]()
        %output_padding : int[] = prim::Constant[value=[0, 0]]()
        %packed = mclaren_prepack::mclaren_encoder_block_prepack(
          %weight1, %bias1, %stride1, %padding1, %output_padding, %dilation1, %groups1,
          %weight2, %bias2, %stride2, %padding2, %output_padding, %dilation2, %groups2,
          %transposed)
        %r = mclaren_prepack::mclaren_encoder_block_run(%input1, %input2, %packed)
        return (%r) )";

  SubgraphRewriter rewriter;
  rewriter.RegisterRewritePattern(
      conv_2d_pattern, prepacked_ops_conv2d_pattern);
  rewriter.runOnGraph(graph);

  std::string conv_2d_transpose_pattern = R"(
    graph(%dim:int,
          %input1, %weight1, %bias1, %stride1:int[], %padding1:int[], %dilation1:int[], %output_padding1:int[], %groups1:int,
          %input2, %weight2, %bias2, %stride2:int[], %padding2:int[], %dilation2:int[], %output_padding2:int[], %groups2:int):
        %list : Tensor[] = prim::ListConstruct(%input1, %input2)
        %input = aten::cat(%list, %dim)
        %r1 = aten::conv_transpose2d(%input, %weight1, %bias1, %stride1, %padding1, %output_padding1, %groups1, %dilation1)
        %r2 = aten::conv_transpose2d(%input, %weight2, %bias2, %stride2, %padding2, %output_padding2, %groups2, %dilation2)
        %r3 = aten::sigmoid(%r2)
        %r = aten::mul(%r1, %r3)
        return (%r) )";

  std::string prepacked_ops_conv2d_transpose_pattern = R"(
    graph(%dim:int,
          %input1, %weight1, %bias1, %stride1:int[], %padding1:int[], %dilation1:int[], %output_padding1:int[], %groups1:int,
          %input2, %weight2, %bias2, %stride2:int[], %padding2:int[], %dilation2:int[], %output_padding2:int[], %groups2:int):
        %transposed : bool = prim::Constant[value=1]()
        %packed = mclaren_prepack::mclaren_encoder_block_prepack(
          %weight1, %bias1, %stride1, %padding1, %output_padding1, %dilation1, %groups1,
          %weight2, %bias2, %stride2, %padding2, %output_padding2, %dilation2, %groups2,
          %transposed)
        %r = mclaren_prepack::mclaren_encoder_block_run(%input1, %input2, %packed)
        return (%r) )";

  SubgraphRewriter transpose_rewriter;
  transpose_rewriter.RegisterRewritePattern(
      conv_2d_transpose_pattern, prepacked_ops_conv2d_transpose_pattern);
  transpose_rewriter.runOnGraph(graph);
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

  std::string conv_2d_transpose_pattern = R"(
      graph(%input, %weight, %bias, %stride:int[], %padding:int[], %dilation:int[],
          %output_padding:int[], %groups:int):
        %res = aten::conv_transpose2d(%input, %weight, %bias, %stride, %padding, %output_padding, %groups, %dilation)
        return (%res) )";

  std::string prepacked_ops_conv2d_transpose_pattern = R"(
    graph(%input, %weight, %bias, %stride:int[], %padding:int[], %dilation:int[], %output_padding:int[], %groups:int):
        %output_min_max : None = prim::Constant()
        %packed_weight_bias = vulkan_prepack::conv2d_transpose_clamp_prepack(
            %weight, %bias, %stride, %padding, %output_padding, %dilation, %groups,
            %output_min_max, %output_min_max)
        %res = vulkan_prepack::conv2d_transpose_clamp_run(%input, %packed_weight_bias)
        return (%res) )";

  SubgraphRewriter transpose_rewriter;
  transpose_rewriter.RegisterRewritePattern(
      conv_2d_transpose_pattern, prepacked_ops_conv2d_transpose_pattern);
  transpose_rewriter.runOnGraph(graph);
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
  insertPrePackedMcLarenEncoderBlockOp(graph);
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
         Symbol::fromQualString("vulkan_prepack::linear_prepack")) ||
        (n->kind() ==
         Symbol::fromQualString("mclaren_prepack::mclaren_encoder_block_prepack")) ||
        (n->kind() ==
         Symbol::fromQualString(
             "vulkan_prepack::conv2d_transpose_clamp_prepack")));
  };
  PrePackingOpsFolder(m, filter_fn, "prepack_folding");
}

void vulkanRemoveMutation(script::Module& module) {
  auto graph = module.get_method("forward").graph();
  RemoveTensorMutation(graph);
}

void vulkanRunCanonicalOptimizations(script::Module& module) {
  auto graph = module.get_method("forward").graph();
  for (const auto& method : module.get_methods()) {
    auto graph = method.graph();
    runOptimization(graph, false /* no loop unrolling */);
  }
}

script::Module vulkanOptimizeForMobile(
    const script::Module& m,
    const std::vector<std::string>& preserved_methods) {
  auto cloned_module = m.clone();
  cloned_module.eval();
  cloned_module = FoldConvBatchNorm(cloned_module);
  cloned_module = freeze_module(cloned_module, preserved_methods);
  cloned_module.dump(true, false, false);
  vulkanInsertPrePackedOps(cloned_module);
  //cloned_module.dump(true, false, false);
  vulkanFusePrePackedConvWithClamp(cloned_module);
  vulkanFoldPrePackingOps(cloned_module);
  removeDropout(cloned_module);
  vulkanRemoveMutation(cloned_module);
  // remove duplicated constants
  vulkanRunCanonicalOptimizations(cloned_module);

  cloned_module.register_attribute(
      "optimized_for_vulkan", BoolType::get(), true);
  return cloned_module;
}

#else

void vulkanInsertPrePackedOps(std::shared_ptr<Graph>& graph) {
  TORCH_INTERNAL_ASSERT(
      false, "Vulkan is not enabled. Please build with USE_VULKAN=1");
}

void vulkanInsertPrePackedOps(script::Module& module) {
  TORCH_INTERNAL_ASSERT(
      false, "Vulkan is not enabled. Please build with USE_VULKAN=1");
}

void vulkanFusePrePackedConvWithClamp(script::Module& module) {
  TORCH_INTERNAL_ASSERT(
      false, "Vulkan is not enabled. Please build with USE_VULKAN=1");
}

void vulkanFoldPrePackingOps(script::Module& m) {
  TORCH_INTERNAL_ASSERT(
      false, "Vulkan is not enabled. Please build with USE_VULKAN=1");
}

script::Module vulkanOptimizeForMobile(
    const script::Module& module,
    const std::vector<std::string>& preserved_methods) {
  TORCH_INTERNAL_ASSERT(
      false,
      "Mobile optimizaiton only available with Vulkan at the moment. "
      "Vulkan is not enabled. Please build with USE_VULKAN=1");
  return module;
}

#endif
} // namespace jit
} // namespace torch
