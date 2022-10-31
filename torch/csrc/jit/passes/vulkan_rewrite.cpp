#include <ATen/core/jit_type.h>
#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/ir/subgraph_matcher.h>
#include <torch/csrc/jit/passes/constant_pooling.h>
#include <torch/csrc/jit/passes/dead_code_elimination.h>
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
        %packed_weight_bias = vulkan_prepack::create_linear_context(
            %weight_t, %bias)
        %res = vulkan_prepack::run_linear_context(%input, %packed_weight_bias)
        return (%res))";

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
        %packed_weight_bias = vulkan_prepack::create_conv2d_context(
            %weight, %bias, %stride, %padding, %dilation, %groups,
            %output_min_max, %output_min_max)
        %r = vulkan_prepack::run_conv2d_context(%input, %packed_weight_bias)
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
        %packed_weight_bias = vulkan_prepack::create_tconv2d_context(
            %weight, %bias, %stride, %padding, %output_padding, %dilation, %groups,
            %output_min_max, %output_min_max)
        %res = vulkan_prepack::run_tconv2d_context(%input, %packed_weight_bias)
        return (%res) )";

  SubgraphRewriter transpose_rewriter;
  transpose_rewriter.RegisterRewritePattern(
      conv_2d_transpose_pattern, prepacked_ops_conv2d_transpose_pattern);
  transpose_rewriter.runOnGraph(graph);
}

void transferInputOutputBackends(std::shared_ptr<Graph>& graph) {
  // Move inputs to Vulkan backend
  for (Value* input : graph->inputs()) {
    NamedValue named_input = NamedValue("", input);
    if (named_input.type()->kind() == TypeKind::TensorType) {
      // find the insertion point
      WithInsertPoint ip(input->uses()[0].user->prev());
      Value* replaced_input = graph->insert(
          Symbol::fromQualString("aten::to"), {named_input, "vulkan"});
      // replace the input
      input->replaceAllUsesAfterNodeWith(
          replaced_input->node(), replaced_input);
    }
  }

  // Move outputs to CPU backend
  at::ArrayRef<Value*>&& outputs = graph->outputs();
  for (size_t i = 0; i < outputs.size(); i++) {
    Value* output = outputs[i];
    NamedValue named_output = NamedValue("", output);
    if (named_output.type()->kind() == TypeKind::TensorType) {
      // find the insertion point
      WithInsertPoint ip(output->node()->next());
      Value* replaced_output = graph->insert(
          Symbol::fromQualString("aten::to"), {named_output, "cpu"});
      // replace the output
      graph->block()->replaceOutput(i, replaced_output);
    }
  }

  SubgraphRewriter rewriter;
  rewriter.runOnGraph(graph);
}

void transferInputOutputBackends(script::Module& module) {
  std::shared_ptr<Graph> graph = module.get_methods()[0].graph();
  transferInputOutputBackends(graph);
}

void eliminateDeadCode(script::Module& module) {
  for (auto& method : module.get_methods()) {
    EliminateDeadCode(method.graph());
  }
}

void insertPrePackedGruOp(std::shared_ptr<Graph>& graph) {
  std::string gru_pattern = R"(
      graph(%input.1, %hx.1, %params_cpu:Tensor[], %has_biases:bool, %num_layers:int, %dropout:float, %train:bool, %bidirectional:bool, %batch_first:bool):
        %y.1 : Tensor, %hn.1 : Tensor = aten::gru(%input.1, %hx.1, %params_cpu, %has_biases, %num_layers, %dropout, %train, %bidirectional, %batch_first)
        return (%y.1, %hn.1) )";
  std::string prepacked_ops_pattern = R"(
      graph(%input.1, %hx.1, %params_cpu:Tensor[], %has_biases:bool, %num_layers:int, %dropout:float, %train:bool, %bidirectional:bool, %batch_first:bool):
        %packed_weights_biases = vulkan_prepack::create_gru_context(
            %params_cpu, %has_biases, %num_layers, %dropout, %train, %bidirectional, %batch_first)
        %y.1 : Tensor, %hn.1 : Tensor = vulkan_prepack::run_gru_context(%input.1, %hx.1, %packed_weights_biases)
        return (%y.1, %hn.1) )";

  auto filter = [&](const Match& match,
                    const std::unordered_map<std::string, Value*>& vmap) {
    auto node = match.values_map.at(vmap.at("params_cpu"))->node();
    return node->output()->type()->str() == "Tensor[]";
  };

  SubgraphRewriter gru_rewriter;
  gru_rewriter.RegisterRewritePattern(gru_pattern, prepacked_ops_pattern);
  gru_rewriter.runOnGraph(graph, filter);
}

void insertPrePackedLstmOp(std::shared_ptr<Graph>& graph) {
  std::string lstm_pattern = R"(
      graph(%input.1, %hx:Tensor[], %params_cpu:Tensor[], %has_biases:bool, %num_layers:int, %dropout:float, %train:bool, %bidirectional:bool, %batch_first:bool):
        %y.1 : Tensor, %hn.1 : Tensor, %cn.1 : Tensor = aten::lstm(%input.1, %hx, %params_cpu, %has_biases, %num_layers, %dropout, %train, %bidirectional, %batch_first)
        return (%y.1, %hn.1, %cn.1) )";
  std::string prepacked_ops_pattern = R"(
      graph(%input.1, %hx:Tensor[], %params_cpu:Tensor[], %has_biases:bool, %num_layers:int, %dropout:float, %train:bool, %bidirectional:bool, %batch_first:bool):
        %packed_weights_biases = vulkan_prepack::create_lstm_context(
            %params_cpu, %has_biases, %num_layers, %dropout, %train, %bidirectional, %batch_first)
        %hx.1 : Tensor, %cx.1 : Tensor = prim::ListUnpack(%hx)
        %y.1 : Tensor, %hn.1 : Tensor, %cn.1 : Tensor = vulkan_prepack::run_lstm_context(%input.1, %hx.1, %cx.1, %packed_weights_biases)
        return (%y.1, %hn.1, %cn.1) )";

  auto filter = [&](const Match& match,
                    const std::unordered_map<std::string, Value*>& vmap) {
    auto node = match.values_map.at(vmap.at("hx"))->node();
    return node->output()->type()->str() == "Tensor[]";
  };

  SubgraphRewriter lstm_rewriter;
  lstm_rewriter.RegisterRewritePattern(lstm_pattern, prepacked_ops_pattern);
  lstm_rewriter.runOnGraph(graph, filter);
}

void fuseHardtanhWithPackedOps(std::shared_ptr<Graph>& graph) {
  SubgraphRewriter rewriter;

  std::string conv2d_prepack_run_hardtanh_fused = R"(
    graph(%input, %weight, %bias, %stride:int[], %padding:int[],
          %dilation:int[], %groups:int, %output_min, %output_max, %dummy_min_max):
        %packed_weight_bias : __torch__.torch.classes.vulkan.Conv2dPackedContext = vulkan_prepack::create_conv2d_context(
            %weight, %bias, %stride, %padding, %dilation, %groups,
            %output_min, %output_max)
        %r = vulkan_prepack::run_conv2d_context(%input, %packed_weight_bias)
        return (%r) )";

  std::string conv2d_prepack_run_hardtanh = R"(
    graph(%input, %weight, %bias, %stride:int[], %padding:int[],
          %dilation:int[], %groups:int, %output_min, %output_max, %dummy_min_max):
        %packed_weight_bias = vulkan_prepack::create_conv2d_context(
            %weight, %bias, %stride, %padding, %dilation, %groups,
            %dummy_min_max, %dummy_min_max)
        %conv2d_res = vulkan_prepack::run_conv2d_context(%input, %packed_weight_bias)
        %r = aten::hardtanh(%conv2d_res, %output_min, %output_max)
        return (%r) )";

  rewriter.RegisterRewritePattern(
      conv2d_prepack_run_hardtanh, conv2d_prepack_run_hardtanh_fused);

  std::string conv2d_prepack_run_hardtanh_inplace = R"(
    graph(%input, %weight, %bias, %stride:int[], %padding:int[],
          %dilation:int[], %groups:int, %output_min, %output_max, %dummy_min_max):
        %packed_weight_bias = vulkan_prepack::create_conv2d_context(
            %weight, %bias, %stride, %padding, %dilation, %groups,
            %dummy_min_max, %dummy_min_max)
        %conv2d_res = vulkan_prepack::run_conv2d_context(%input, %packed_weight_bias)
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
        %packed_weight_bias : __torch__.torch.classes.vulkan.Conv2dPackedContext = vulkan_prepack::create_conv2d_context(
            %weight, %bias, %stride, %padding, %dilation, %groups,
            %output_min, %output_max)
        %r = vulkan_prepack::run_conv2d_context(%input, %packed_weight_bias)
        return (%r) )";

  std::string conv2d_prepack_run_relu = R"(
    graph(%input, %weight, %bias, %stride:int[], %padding:int[],
          %dilation:int[], %groups:int, %dummy_min_max):
        %packed_weight_bias = vulkan_prepack::create_conv2d_context(
            %weight, %bias, %stride, %padding, %dilation, %groups,
            %dummy_min_max, %dummy_min_max)
        %conv2d_res = vulkan_prepack::run_conv2d_context(%input, %packed_weight_bias)
        %r = aten::relu(%conv2d_res)
        return (%r) )";

  rewriter.RegisterRewritePattern(
      conv2d_prepack_run_relu, conv2d_prepack_run_relu_fused);

  std::string conv2d_prepack_run_relu_inplace = R"(
    graph(%input, %weight, %bias, %stride:int[], %padding:int[],
          %dilation:int[], %groups:int, %dummy_min_max):
        %packed_weight_bias = vulkan_prepack::create_conv2d_context(
            %weight, %bias, %stride, %padding, %dilation, %groups,
            %dummy_min_max, %dummy_min_max)
        %conv2d_res = vulkan_prepack::run_conv2d_context(%input, %packed_weight_bias)
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
  insertPrePackedGruOp(graph);
  insertPrePackedLstmOp(graph);
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
         Symbol::fromQualString("vulkan_prepack::create_conv2d_context")) ||
        (n->kind() ==
         Symbol::fromQualString("vulkan_prepack::create_tconv2d_context")) ||
        (n->kind() ==
         Symbol::fromQualString("vulkan_prepack::create_linear_context")) ||
        (n->kind() ==
         Symbol::fromQualString("vulkan_prepack::create_gru_context")) ||
        (n->kind() ==
         Symbol::fromQualString("vulkan_prepack::create_lstm_context")));
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
    const std::set<MobileOptimizerType>& optimization_blocklist,
    const std::vector<std::string>& preserved_methods) {
  auto cloned_module = m.clone();
  cloned_module.eval();
  cloned_module = FoldConvBatchNorm(cloned_module);
  vulkanInsertPrePackedOps(cloned_module);
  cloned_module = freeze_module(cloned_module, preserved_methods);
  if (!optimization_blocklist.count(
          MobileOptimizerType::VULKAN_AUTOMATIC_GPU_TRANSFER)) {
    transferInputOutputBackends(cloned_module);
    cloned_module.register_attribute(
        "requires_backend_transfers", BoolType::get(), false);
  }
  vulkanFusePrePackedConvWithClamp(cloned_module);
  vulkanFoldPrePackingOps(cloned_module);
  removeDropout(cloned_module);
  vulkanRemoveMutation(cloned_module);
  // remove duplicated constants
  vulkanRunCanonicalOptimizations(cloned_module);
  eliminateDeadCode(cloned_module);

  cloned_module.register_attribute(
      "optimized_for_vulkan", BoolType::get(), true);
  return cloned_module;
}

} // namespace jit
} // namespace torch
