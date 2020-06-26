#include <ATen/core/jit_type.h>
#include <ATen/native/xnnpack/OpContext.h>

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
#include <torch/csrc/jit/passes/xnnpack_rewrite.h>
#include <torch/csrc/jit/runtime/graph_executor_impl.h>

namespace torch {
namespace jit {

#ifdef USE_XNNPACK

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
        %output_min_max : None = prim::Constant()
        %packed_weight_bias = prepacked::linear_clamp_prepack(
            %weight, %bias, %output_min_max, %output_min_max)
        %res = prepacked::linear_clamp_run(%input, %packed_weight_bias)
        return (%res))";
  std::string linear_pattern = R"(
    graph(%input, %weight, %bias):
        %r = aten::linear(%input, %weight, %bias)
        return (%r))";
  std::string prepacked_ops_pattern = R"(
    graph(%input, %weight, %bias):
        %output_min_max : None = prim::Constant()
        %packed_weight_bias = prepacked::linear_clamp_prepack(
            %weight, %bias, %output_min_max, %output_min_max)
        %res = prepacked::linear_clamp_run(%input, %packed_weight_bias)
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
      linear_before_inline, prepacked_ops_pattern_before_inline);
  linear_call_fn_rewriter.runOnGraph(graph, filter);

  SubgraphRewriter linear_rewriter;
  linear_rewriter.RegisterRewritePattern(linear_pattern, prepacked_ops_pattern);
  linear_rewriter.runOnGraph(graph);
}

void insertPrePackedConv2dOp(std::shared_ptr<Graph>& graph) {
  // Replace _convolution with conv2d
  graph_rewrite_helper::replaceConvolutionWithAtenConv(graph);

  std::string conv_2d_pattern = R"(
    graph(%input, %weight, %bias, %stride:int[], %padding:int[], %dilation:int[], %groups:int):
        %r = aten::conv2d(%input, %weight, %bias, %stride, %padding, %dilation, %groups)
        return (%r) )";

  std::string prepacked_ops_conv2d_pattern = R"(
    graph(%input, %weight, %bias, %stride:int[], %padding:int[], %dilation:int[], %groups:int):
        %output_min_max : None = prim::Constant()
        %packed_weight_bias = prepacked::conv2d_clamp_prepack(
            %weight, %bias, %stride, %padding, %dilation, %groups,
            %output_min_max, %output_min_max)
        %r = prepacked::conv2d_clamp_run(%input, %packed_weight_bias)
        return (%r) )";

  SubgraphRewriter rewriter;
  rewriter.RegisterRewritePattern(
      conv_2d_pattern, prepacked_ops_conv2d_pattern);
  rewriter.runOnGraph(graph);
}

void fuseHardtanhWithPackedOps(std::shared_ptr<Graph>& graph) {
  SubgraphRewriter rewriter;

  std::string linear_prepack_run_hardtanh_fused = R"(
    graph(%input, %weight, %bias, %output_min, %output_max, %dummy_min_max):
        %packed_weight_bias : __torch__.torch.classes.xnnpack.LinearOpContext = prepacked::linear_clamp_prepack(
            %weight, %bias, %output_min, %output_max)
        %res = prepacked::linear_clamp_run(%input, %packed_weight_bias)
        return (%res))";

  std::string conv2d_prepack_run_hardtanh_fused = R"(
    graph(%input, %weight, %bias, %stride:int[], %padding:int[],
          %dilation:int[], %groups:int, %output_min, %output_max, %dummy_min_max):
        %packed_weight_bias : __torch__.torch.classes.xnnpack.Conv2dOpContext = prepacked::conv2d_clamp_prepack(
            %weight, %bias, %stride, %padding, %dilation, %groups,
            %output_min, %output_max)
        %r = prepacked::conv2d_clamp_run(%input, %packed_weight_bias)
        return (%r) )";

  std::string linear_prepack_run_hardtanh = R"(
    graph(%input, %weight, %bias, %output_min, %output_max, %dummy_min_max):
        %packed_weight_bias = prepacked::linear_clamp_prepack(
            %weight, %bias, %dummy_min_max, %dummy_min_max)
        %linear_res = prepacked::linear_clamp_run(%input, %packed_weight_bias)
        %res = aten::hardtanh(%linear_res, %output_min, %output_max)
        return (%res))";

  rewriter.RegisterRewritePattern(
      linear_prepack_run_hardtanh, linear_prepack_run_hardtanh_fused);

  std::string conv2d_prepack_run_hardtanh = R"(
    graph(%input, %weight, %bias, %stride:int[], %padding:int[],
          %dilation:int[], %groups:int, %output_min, %output_max, %dummy_min_max):
        %packed_weight_bias = prepacked::conv2d_clamp_prepack(
            %weight, %bias, %stride, %padding, %dilation, %groups,
            %dummy_min_max, %dummy_min_max)
        %conv2d_res = prepacked::conv2d_clamp_run(%input, %packed_weight_bias)
        %r = aten::hardtanh(%conv2d_res, %output_min, %output_max)
        return (%r) )";

  rewriter.RegisterRewritePattern(
      conv2d_prepack_run_hardtanh, conv2d_prepack_run_hardtanh_fused);

  std::string linear_prepack_run_hardtanh_inplace = R"(
    graph(%input, %weight, %bias, %output_min, %output_max, %dummy_min_max):
        %packed_weight_bias = prepacked::linear_clamp_prepack(
            %weight, %bias, %dummy_min_max, %dummy_min_max)
        %linear_res = prepacked::linear_clamp_run(%input, %packed_weight_bias)
        %res = aten::hardtanh_(%linear_res, %output_min, %output_max)
        return (%res))";

  std::string conv2d_prepack_run_hardtanh_inplace = R"(
    graph(%input, %weight, %bias, %stride:int[], %padding:int[],
          %dilation:int[], %groups:int, %output_min, %output_max, %dummy_min_max):
        %packed_weight_bias = prepacked::conv2d_clamp_prepack(
            %weight, %bias, %stride, %padding, %dilation, %groups,
            %dummy_min_max, %dummy_min_max)
        %conv2d_res = prepacked::conv2d_clamp_run(%input, %packed_weight_bias)
        %r = aten::hardtanh_(%conv2d_res, %output_min, %output_max)
        return (%r) )";

  rewriter.RegisterRewritePattern(
      linear_prepack_run_hardtanh_inplace, linear_prepack_run_hardtanh_fused);
  rewriter.RegisterRewritePattern(
      conv2d_prepack_run_hardtanh_inplace, conv2d_prepack_run_hardtanh_fused);

  rewriter.runOnGraph(graph, torch::jit::graph_rewrite_helper::isClampFusable);
}

void fuseReluWithPackedOps(std::shared_ptr<Graph>& graph) {
  SubgraphRewriter rewriter;

  std::string linear_prepack_run_relu_fused = R"(
    graph(%input, %weight, %bias, %dummy_min_max):
        %output_min: float = prim::Constant[value=0.0]()
        %output_max: None = prim::Constant()
        %packed_weight_bias : __torch__.torch.classes.xnnpack.LinearOpContext = prepacked::linear_clamp_prepack(
            %weight, %bias, %output_min, %output_max)
        %res = prepacked::linear_clamp_run(%input, %packed_weight_bias)
        return (%res))";

  std::string conv2d_prepack_run_relu_fused = R"(
    graph(%input, %weight, %bias, %stride:int[], %padding:int[],
          %dilation:int[], %groups:int, %dummy_min_max):
        %output_min: float = prim::Constant[value=0.0]()
        %output_max: None = prim::Constant()
        %packed_weight_bias : __torch__.torch.classes.xnnpack.Conv2dOpContext = prepacked::conv2d_clamp_prepack(
            %weight, %bias, %stride, %padding, %dilation, %groups,
            %output_min, %output_max)
        %r = prepacked::conv2d_clamp_run(%input, %packed_weight_bias)
        return (%r) )";

  std::string linear_prepack_run_relu = R"(
    graph(%input, %weight, %bias, %dummy_min_max):
        %packed_weight_bias = prepacked::linear_clamp_prepack(
            %weight, %bias, %dummy_min_max, %dummy_min_max)
        %linear_res = prepacked::linear_clamp_run(%input, %packed_weight_bias)
        %res = aten::relu(%linear_res)
        return (%res))";

  rewriter.RegisterRewritePattern(
      linear_prepack_run_relu, linear_prepack_run_relu_fused);

  std::string conv2d_prepack_run_relu = R"(
    graph(%input, %weight, %bias, %stride:int[], %padding:int[],
          %dilation:int[], %groups:int, %dummy_min_max):
        %packed_weight_bias = prepacked::conv2d_clamp_prepack(
            %weight, %bias, %stride, %padding, %dilation, %groups,
            %dummy_min_max, %dummy_min_max)
        %conv2d_res = prepacked::conv2d_clamp_run(%input, %packed_weight_bias)
        %r = aten::relu(%conv2d_res)
        return (%r) )";

  rewriter.RegisterRewritePattern(
      conv2d_prepack_run_relu, conv2d_prepack_run_relu_fused);

  std::string linear_prepack_run_relu_inplace = R"(
    graph(%input, %weight, %bias, %dummy_min_max):
        %packed_weight_bias = prepacked::linear_clamp_prepack(
            %weight, %bias, %dummy_min_max, %dummy_min_max)
        %linear_res = prepacked::linear_clamp_run(%input, %packed_weight_bias)
        %res = aten::relu_(%linear_res)
        return (%res))";

  std::string conv2d_prepack_run_relu_inplace = R"(
    graph(%input, %weight, %bias, %stride:int[], %padding:int[],
          %dilation:int[], %groups:int, %dummy_min_max):
        %packed_weight_bias = prepacked::conv2d_clamp_prepack(
            %weight, %bias, %stride, %padding, %dilation, %groups,
            %dummy_min_max, %dummy_min_max)
        %conv2d_res = prepacked::conv2d_clamp_run(%input, %packed_weight_bias)
        %r = aten::relu_(%conv2d_res)
        return (%r) )";

  rewriter.RegisterRewritePattern(
      linear_prepack_run_relu_inplace, linear_prepack_run_relu_fused);
  rewriter.RegisterRewritePattern(
      conv2d_prepack_run_relu_inplace, conv2d_prepack_run_relu_fused);
  rewriter.runOnGraph(graph, torch::jit::graph_rewrite_helper::isClampFusable);
}

void runCanonicalOptimizations(script::Module& module) {
  auto graph = module.get_method("forward").graph();
  // Not sure if we have models running on mobile that require loop unrolling.
  // Perhaps language/speech models? Conservatively setting that to false.
  runOptimization(graph, false /* no loop unrolling */);
}

} // namespace

void insertPrePackedOps(std::shared_ptr<Graph>& graph) {
  insertPrePackedLinearOp(graph);
  insertPrePackedConv2dOp(graph);
}

void insertPrePackedOps(script::Module& module) {
  for (auto& method : module.get_methods()) {
    auto graph = method.graph();
    insertPrePackedOps(graph);
  }
  for (script::Module m : module.children()) {
    insertPrePackedOps(m);
  }
}

void fusePrePackedLinearConvWithClamp(script::Module& module) {
  auto graph = module.get_method("forward").graph();
  fuseReluWithPackedOps(graph);
  fuseHardtanhWithPackedOps(graph);
}

void FoldPrePackingOps(script::Module& m) {
  PrePackingOpsFilterFn filter_fn = [](const Node* n) -> bool {
    return (
        (n->kind() ==
         Symbol::fromQualString("prepacked::linear_clamp_prepack")) ||
        n->kind() == Symbol::fromQualString("prepacked::conv2d_clamp_prepack"));
  };
  PrePackingOpsFolder(m, filter_fn, "prepack_folding");
}

script::Module optimizeForMobile(
    const script::Module& m,
    const std::set<MobileOptimizerType>& optimization_blacklist) {
  auto cloned_module = m.clone();
  cloned_module.eval();

  if (!optimization_blacklist.count(MobileOptimizerType::CONV_BN_FUSION)) {
    cloned_module = FoldConvBatchNorm(cloned_module);
  }

  if (!optimization_blacklist.count(
          MobileOptimizerType::INSERT_FOLD_PREPACK_OPS)) {
    insertPrePackedOps(cloned_module);
    cloned_module = freeze_module(cloned_module);
    fusePrePackedLinearConvWithClamp(cloned_module);
    FoldPrePackingOps(cloned_module);
  }

  // Run canonical optimizations post freezing
  // since freezing inlines the graph. Otherwise we
  // will have to explicitly call Inlining pass.
  runCanonicalOptimizations(cloned_module);

  if (!optimization_blacklist.count(MobileOptimizerType::REMOVE_DROPOUT)) {
    removeDropout(cloned_module);
  }

  return cloned_module;
}

#else

void insertPrePackedOps(std::shared_ptr<Graph>& graph) {
  TORCH_INTERNAL_ASSERT(
      "XNNPACK is not enabled. Please build with USE_XNNPACK=1");
}

void insertPrePackedOps(script::Module& module) {
  TORCH_INTERNAL_ASSERT(
      "XNNPACK is not enabled. Please build with USE_XNNPACK=1");
}

void fusePrePackedLinearConvWithClamp(script::Module& module) {
  TORCH_INTERNAL_ASSERT(
      "XNNPACK is not enabled. Please build with USE_XNNPACK=1");
}

void FoldPrePackingOps(script::Module& m) {
  TORCH_INTERNAL_ASSERT(
      "XNNPACK is not enabled. Please build with USE_XNNPACK=1");
}

script::Module optimizeForMobile(
    const script::Module& module,
    const std::set<MobileOptimizerType>& blacklist) {
  TORCH_INTERNAL_ASSERT(
      "Mobile optimizaiton only available with XNNPACK at the moment. "
      "XNNPACK is not enabled. Please build with USE_XNNPACK=1");
  return module;
}

#endif
} // namespace jit
} // namespace torch
