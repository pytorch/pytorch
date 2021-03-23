#include <ATen/core/jit_type.h>
#include <ATen/native/xnnpack/OpContext.h>

#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/ir/subgraph_matcher.h>
#include <torch/csrc/jit/passes/constant_pooling.h>
#include <torch/csrc/jit/passes/constant_propagation.h>
#include <torch/csrc/jit/passes/fold_conv_bn.h>
#include <torch/csrc/jit/passes/freeze_module.h>
#include <torch/csrc/jit/passes/fuse_linear.h>
#include <torch/csrc/jit/passes/fuse_relu.h>
#include <torch/csrc/jit/passes/graph_rewrite_helper.h>
#include <torch/csrc/jit/passes/hoist_conv_packed_params.h>
#include <torch/csrc/jit/passes/inliner.h>
#include <torch/csrc/jit/passes/prepack_folding.h>
#include <torch/csrc/jit/passes/remove_dropout.h>
#include <torch/csrc/jit/passes/subgraph_rewrite.h>
#include <torch/csrc/jit/passes/xnnpack_rewrite.h>
#include <torch/csrc/jit/runtime/graph_executor_impl.h>

namespace torch {
namespace jit {

namespace {

void replaceConv1dWithConv2d(std::shared_ptr<Graph>& graph) {
  std::string conv_1d_pattern = R"(
    graph(%input, %weight, %bias, %stride:int[], %padding:int[], %dilation:int[], %groups:int):
        %r = aten::conv1d(%input, %weight, %bias, %stride, %padding, %dilation, %groups)
        return (%r) )";

  std::string conv_2d_pattern = R"(
    graph(%input, %weight, %bias, %stride:int[], %padding:int[], %dilation:int[], %groups:int):
        %zero : int = prim::Constant[value=0]()
        %one : int = prim::Constant[value=1]()
        %stride_w : int = prim::ListUnpack(%stride)
        %stride_2d : int[] = prim::ListConstruct(%one, %stride_w)
        %padding_w : int = prim::ListUnpack(%padding)
        %padding_2d : int[] = prim::ListConstruct(%zero, %padding_w)
        %dilation_w : int = prim::ListUnpack(%dilation)
        %dilation_2d : int[] = prim::ListConstruct(%one, %dilation_w)
        %two : int = prim::Constant[value=2]()
        %input_2d : Tensor = aten::unsqueeze(%input, %two)
        %weight_2d : Tensor = aten::unsqueeze(%weight, %two)
        %output_2d = aten::conv2d(
            %input_2d, %weight_2d, %bias, %stride_2d, %padding_2d, %dilation_2d, %groups)
        %output : Tensor = aten::squeeze(%output_2d, %two)
        return (%output) )";

  SubgraphRewriter rewriter;
  rewriter.RegisterRewritePattern(conv_1d_pattern, conv_2d_pattern);
  rewriter.runOnGraph(graph);
}

} // namespace

void transformConv1dToConv2d(std::shared_ptr<Graph>& graph) {
  // Replace _convolution with conv1d and conv2d
  graph_rewrite_helper::replaceConvolutionWithAtenConv(graph);
  replaceConv1dWithConv2d(graph);
}

void transformConv1dToConv2d(script::Module& module) {
  for (auto& method : module.get_methods()) {
    auto graph = method.graph();
    transformConv1dToConv2d(graph);
  }
  for (script::Module m : module.children()) {
    transformConv1dToConv2d(m);
  }
}

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

  std::string conv_2d_transpose_pattern = R"(
      graph(%input, %weight, %bias, %stride:int[], %padding:int[], %dilation:int[],
          %output_padding:int[], %groups:int):
        %r = aten::conv_transpose2d(%input, %weight, %bias, %stride, %padding, %output_padding, %groups, %dilation)
        return (%r) )";

  std::string prepacked_ops_conv2d_transpose_pattern = R"(
    graph(%input, %weight, %bias, %stride:int[], %padding:int[], %dilation:int[], %output_padding:int[], %groups:int):
        %output_min_max : None = prim::Constant()
        %packed_weight_bias = prepacked::conv2d_transpose_clamp_prepack(
            %weight, %bias, %stride, %padding, %output_padding, %dilation, %groups,
            %output_min_max, %output_min_max)
        %r = prepacked::conv2d_transpose_clamp_run(%input, %packed_weight_bias)
        return (%r) )";

  SubgraphRewriter transpose_rewriter;
  transpose_rewriter.RegisterRewritePattern(
      conv_2d_transpose_pattern, prepacked_ops_conv2d_transpose_pattern);
  transpose_rewriter.runOnGraph(graph);
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

void runCanonicalOptimizations(
    script::Module& module,
    const std::unordered_set<std::string>& methods_to_optimize) {
  for (const std::string& method : methods_to_optimize) {
    auto graph = module.get_method(method).graph();
    // Not sure if we have models running on mobile that require loop unrolling.
    // Perhaps language/speech models? Conservatively setting that to false.
    runOptimization(graph, false /* no loop unrolling */);
  }
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

  // Ignore user defined classes for later passes
  ConstantPropagation(graph, true);
}

void FoldPrePackingOps(script::Module& m) {
  PrePackingOpsFilterFn filter_fn = [](const Node* n) -> bool {
    return (
        (n->kind() ==
         Symbol::fromQualString("prepacked::linear_clamp_prepack")) ||
        n->kind() ==
            Symbol::fromQualString("prepacked::conv2d_clamp_prepack") ||
        n->kind() ==
            Symbol::fromQualString(
                "prepacked::conv2d_transpose_clamp_prepack"));
  };
  PrePackingOpsFolder(m, filter_fn, "prepack_folding");
  auto graph = m.get_method("forward").graph();
  // Folding requires a const propagation through user defined classes
  ConstantPropagation(graph, false);
}

script::Module optimizeForMobile(
    const script::Module& m,
    const std::set<MobileOptimizerType>& optimization_blocklist,
    const std::vector<std::string>& preserved_methods,
    const std::vector<std::string>& methods_to_optimize_arg) {
  std::unordered_set<std::string> methods_to_optimize(
      methods_to_optimize_arg.begin(), methods_to_optimize_arg.end());

  // Forward is optimized by default if it exists
  bool optimize_forward = false;
  if (m.find_method("forward")) {
    optimize_forward = true;
    methods_to_optimize.insert("forward");
  }
  TORCH_INTERNAL_ASSERT(
      methods_to_optimize.size() > 0,
      "Model must either define forward, or pass in at least 1 function to optimize");

  // Preserve all functions we intend to optimize
  // Need to keep the optimized_set separate because there are often
  // methods that need to be preserved, but wouldnt need optimization
  // Convert to set and back to remove duplicates
  std::unordered_set<std::string> preserved_set(
      preserved_methods.begin(), preserved_methods.end());
  preserved_set.insert(methods_to_optimize.begin(), methods_to_optimize.end());
  std::vector<std::string> preserved_list(
      preserved_set.begin(), preserved_set.end());

  auto cloned_module = m.clone();
  cloned_module.eval();

  if (!optimization_blocklist.count(MobileOptimizerType::CONV_BN_FUSION) &&
      optimize_forward) {
    cloned_module = FoldConvBatchNorm(cloned_module);
  }

  // Many optimizations require a frozen module, but ConvBatchNorm requires
  // an unfrozen module
  cloned_module = freeze_module(cloned_module, preserved_list);

  if (!optimization_blocklist.count(
          MobileOptimizerType::INSERT_FOLD_PREPACK_OPS) &&
      optimize_forward) {
    insertPrePackedOps(cloned_module);
    cloned_module = freeze_module(cloned_module, preserved_list);
    fusePrePackedLinearConvWithClamp(cloned_module);
    FoldPrePackingOps(cloned_module);
  }

  if (!optimization_blocklist.count(
          MobileOptimizerType::HOIST_CONV_PACKED_PARAMS) &&
      optimize_forward) {
    // freeze again in case it was not done in previous optional passes
    cloned_module = freeze_module(cloned_module, preserved_list);
    HoistConvPackedParams(cloned_module);
    // and freeze yet again to remove the empty QuantizedConv modules
    cloned_module = freeze_module(cloned_module, preserved_list);
  }

  // Run canonical optimizations post freezing
  // since freezing inlines the graph. Otherwise we
  // will have to explicitly call Inlining pass.
  runCanonicalOptimizations(cloned_module, methods_to_optimize);

  if (!optimization_blocklist.count(MobileOptimizerType::REMOVE_DROPOUT)) {
    for (const std::string& method : methods_to_optimize) {
      auto graph = cloned_module.get_method(method).graph();
      // Module must be not be in training mode but optimize calls eval()
      removeDropout(graph);
    }
  }

  if (!optimization_blocklist.count(MobileOptimizerType::FUSE_ADD_RELU)) {
    for (const std::string& method : methods_to_optimize) {
      auto graph = cloned_module.get_method(method).graph();
      FuseAddRelu(graph);
    }
  }
  cloned_module.register_attribute("mobile_optimized", BoolType::get(), true);
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
    const std::set<MobileOptimizerType>& blocklist,
    const std::vector<std::string>& preserved_methods,
    const std::vector<std::string>& methods_to_optimize) {
  TORCH_INTERNAL_ASSERT(
      "Mobile optimization only available with XNNPACK at the moment. "
      "XNNPACK is not enabled. Please build with USE_XNNPACK=1");
  return module;
}

#endif
} // namespace jit
} // namespace torch
