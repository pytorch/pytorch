#include <ATen/core/jit_type.h>
#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/ir/subgraph_matcher.h>
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

void insertPrePackedBatchNormOp(std::shared_ptr<Graph>& graph) {
  std::string batchnorm_pattern = R"(
    graph(%input, %weight, %bias, %mean, %var, %training, %momentum, %eps, %cudnn_enable):
        %r = aten::batch_norm(%input, %weight, %bias, %mean, %var, %training, %momentum, %eps, %cudnn_enable)
        return (%r))";
  std::string prepacked_ops_pattern = R"(
    graph(%input, %weight, %bias, %mean, %var, %training, %momentum, %eps, %cudnn_enable):
        %op_context : __torch__.torch.classes.vulkan.BatchNormPackedContext = vulkan_prepack::create_batchnorm_context(
            %weight, %bias, %mean, %var, %training, %momentum, %eps, %cudnn_enable)
        %res = vulkan_prepack::run_batchnorm_context(%input, %op_context)
        return (%res))";

  SubgraphRewriter batchnorm_rewriter;
  batchnorm_rewriter.RegisterRewritePattern(
      batchnorm_pattern, prepacked_ops_pattern);
  batchnorm_rewriter.runOnGraph(graph);
}

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

void insertPrePackedLayernormOp(std::shared_ptr<Graph>& graph) {
  std::string layernorm_pattern = R"(
    graph(%input, %normalized_shape, %weight, %bias, %eps, %cudnn_enable):
        %r = aten::layer_norm(%input, %normalized_shape, %weight, %bias, %eps, %cudnn_enable)
        return (%r))";
  std::string prepacked_ops_pattern = R"(
    graph(%input, %normalized_shape, %weight, %bias, %eps, %cudnn_enable):
        %op_context : __torch__.torch.classes.vulkan.LayernormPackedContext = vulkan_prepack::create_layernorm_context(
            %weight, %bias, %eps)
        %res = vulkan_prepack::run_layernorm_context(%input, %normalized_shape, %op_context)
        return (%res))";

  SubgraphRewriter layernorm_rewriter;
  layernorm_rewriter.RegisterRewritePattern(
      layernorm_pattern, prepacked_ops_pattern);
  layernorm_rewriter.runOnGraph(graph);
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

void insertPrePackedConv1dOp(std::shared_ptr<Graph>& graph) {
  graph_rewrite_helper::replaceConvolutionWithAtenConv(graph);

  std::string conv_1d_pattern = R"(
    graph(%input, %weight, %bias, %stride:int[], %padding:int[], %dilation:int[], %groups:int):
        %r = aten::conv1d(%input, %weight, %bias, %stride, %padding, %dilation, %groups)
        return (%r) )";

  std::string prepacked_ops_conv1d_pattern = R"(
    graph(%input, %weight, %bias, %stride:int[], %padding:int[], %dilation:int[], %groups:int):
        %packed_weight_bias = vulkan_prepack::create_conv1d_context(
            %weight, %bias, %stride, %padding, %dilation, %groups)
        %r = vulkan_prepack::run_conv1d_context(%input, %packed_weight_bias)
        return (%r) )";

  SubgraphRewriter rewriter;
  rewriter.RegisterRewritePattern(
      conv_1d_pattern, prepacked_ops_conv1d_pattern);
  rewriter.runOnGraph(graph);
}

void transferInputOutputBackends(std::shared_ptr<Graph>& graph) {
  // Move inputs to Vulkan backend
  for (Value* input : graph->inputs()) {
    NamedValue named_input = NamedValue("", input);
    if (named_input.type()->kind() == TypeKind::TensorType &&
        !input->uses().empty()) {
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

void rewriteQuantizedOps(std::shared_ptr<Graph>& graph) {
  // quantized::add
  std::string quantized_add_pattern = R"(
    graph(%a_quant, %b_quant, %r_scale, %r_zero_point) :
      %res = quantized::add(%a_quant, %b_quant, %r_scale, %r_zero_point)
      return (%res) )";
  std::string vk_quantized_add_pattern = R"(
    graph(%a_quant, %b_quant, %r_scale, %r_zero_point) :
      %res = vulkan_quantized::add(%a_quant, %b_quant, %r_scale, %r_zero_point)
      return (%res) )";

  torch::jit::SubgraphRewriter quantized_add_rewriter;
  quantized_add_rewriter.RegisterRewritePattern(
      quantized_add_pattern, vk_quantized_add_pattern);
  quantized_add_rewriter.runOnGraph(graph);

  // quantized::mul
  std::string quantized_mul_pattern = R"(
    graph(%a_quant, %b_quant, %r_scale, %r_zero_point) :
      %res = quantized::mul(%a_quant, %b_quant, %r_scale, %r_zero_point)
      return (%res) )";
  std::string vk_quantized_mul_pattern = R"(
    graph(%a_quant, %b_quant, %r_scale, %r_zero_point) :
      %res = vulkan_quantized::mul(%a_quant, %b_quant, %r_scale, %r_zero_point)
      return (%res) )";

  torch::jit::SubgraphRewriter quantized_mul_rewriter;
  quantized_mul_rewriter.RegisterRewritePattern(
      quantized_mul_pattern, vk_quantized_mul_pattern);
  quantized_mul_rewriter.runOnGraph(graph);

  // quantized::conv2d
  std::string quantized_conv2d_pattern = R"(
    graph(%a_quant, %packed_params, %r_scale, %r_zero_point) :
      %res = quantized::conv2d(%a_quant, %packed_params, %r_scale, %r_zero_point)
      return (%res) )";
  std::string vk_quantized_conv2d_pattern = R"(
    graph(%a_quant, %packed_params, %r_scale, %r_zero_point):
      %output_min_max : None = prim::Constant()
      %vk_packed_params : __torch__.torch.classes.vulkan.Conv2dPackedContext = vulkan_quantized_prepack::convert_qconv2d_context(
        %packed_params, %output_min_max, %output_min_max)
      %res = vulkan_prepack::run_qconv2d_context(
        %a_quant, %r_scale, %r_zero_point, %vk_packed_params)
      return (%res) )";

  torch::jit::SubgraphRewriter quantized_conv2d_rewriter;
  quantized_conv2d_rewriter.RegisterRewritePattern(
      quantized_conv2d_pattern, vk_quantized_conv2d_pattern);
  quantized_conv2d_rewriter.runOnGraph(graph);

  // quantized::conv2d_relu
  std::string quantized_conv2d_relu_pattern = R"(
    graph(%a_quant, %packed_params, %r_scale, %r_zero_point) :
      %res = quantized::conv2d_relu(%a_quant, %packed_params, %r_scale, %r_zero_point)
      return (%res) )";
  std::string vk_quantized_conv2d_relu_pattern = R"(
    graph(%a_quant, %packed_params, %r_scale, %r_zero_point):
      %output_min: float = prim::Constant[value=0.0]()
      %output_max: None = prim::Constant()
      %vk_packed_params : __torch__.torch.classes.vulkan.Conv2dPackedContext = vulkan_quantized_prepack::convert_qconv2d_context(
        %packed_params, %output_min, %output_max)
      %res = vulkan_prepack::run_qconv2d_context(
        %a_quant, %r_scale, %r_zero_point, %vk_packed_params)
      return (%res) )";

  torch::jit::SubgraphRewriter quantized_conv2d_relu_rewriter;
  quantized_conv2d_relu_rewriter.RegisterRewritePattern(
      quantized_conv2d_relu_pattern, vk_quantized_conv2d_relu_pattern);
  quantized_conv2d_relu_rewriter.runOnGraph(graph);

  // quantized::linear
  std::string quantized_linear_pattern = R"(
    graph(%a_quant, %packed_params, %r_scale, %r_zero_point) :
      %res = quantized::linear(%a_quant, %packed_params, %r_scale, %r_zero_point)
      return (%res) )";
  std::string vk_quantized_linear_pattern = R"(
    graph(%a_quant, %packed_params, %r_scale, %r_zero_point):
      %vk_packed_params : __torch__.torch.classes.vulkan.LinearPackedContext = vulkan_quantized_prepack::convert_linear_context(
        %packed_params)
      %res = vulkan_prepack::run_qlinear_context(
        %a_quant, %r_scale, %r_zero_point, %vk_packed_params)
      return (%res) )";

  torch::jit::SubgraphRewriter quantized_linear_rewriter;
  quantized_linear_rewriter.RegisterRewritePattern(
      quantized_linear_pattern, vk_quantized_linear_pattern);
  quantized_linear_rewriter.runOnGraph(graph);
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
  insertPrePackedLayernormOp(graph);
  insertPrePackedConv2dOp(graph);
  insertPrePackedConv1dOp(graph);
  rewriteQuantizedOps(graph);
  insertPrePackedGruOp(graph);
  insertPrePackedLstmOp(graph);
  insertPrePackedBatchNormOp(graph);
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
         Symbol::fromQualString("vulkan_prepack::create_qconv2d_context")) ||
        (n->kind() ==
         Symbol::fromQualString(
             "vulkan_quantized_prepack::convert_qconv2d_context")) ||
        (n->kind() ==
         Symbol::fromQualString("vulkan_prepack::create_conv1d_context")) ||
        (n->kind() ==
         Symbol::fromQualString(
             "vulkan_quantized_prepack::convert_linear_context")) ||
        (n->kind() ==
         Symbol::fromQualString("vulkan_prepack::create_linear_context")) ||
        (n->kind() ==
         Symbol::fromQualString("vulkan_prepack::create_layernorm_context")) ||
        (n->kind() ==
         Symbol::fromQualString("vulkan_prepack::create_gru_context")) ||
        (n->kind() ==
         Symbol::fromQualString("vulkan_prepack::create_lstm_context")) ||
        (n->kind() ==
         Symbol::fromQualString("vulkan_prepack::create_batchnorm_context")));
  };
  PrePackingOpsFolder(m, filter_fn, "prepack_folding");
}

static void vulkanRemoveMutation(script::Module& module) {
  auto graph = module.get_method("forward").graph();
  RemoveTensorMutation(graph);
}

static void vulkanRunCanonicalOptimizations(script::Module& module) {
  auto graph = module.get_method("forward").graph();
  for (const auto& method : module.get_methods()) {
    auto method_graph = method.graph();
    runOptimization(method_graph, false /* no loop unrolling */);
  }
}

script::Module vulkanOptimizeForMobile(
    const script::Module& m,
    const std::set<MobileOptimizerType>& optimization_blocklist,
    const std::vector<std::string>& preserved_methods) {
  auto cloned_module = m.clone();
  cloned_module.eval();
  cloned_module = FoldConvBatchNorm(cloned_module);
  cloned_module = freeze_module(cloned_module, preserved_methods);
  vulkanInsertPrePackedOps(cloned_module);
  vulkanFusePrePackedConvWithClamp(cloned_module);
  vulkanFoldPrePackingOps(cloned_module);
  removeDropout(cloned_module);
  vulkanRemoveMutation(cloned_module);

  if (!optimization_blocklist.count(
          MobileOptimizerType::VULKAN_AUTOMATIC_GPU_TRANSFER)) {
    transferInputOutputBackends(cloned_module);
    cloned_module.register_attribute(
        "requires_backend_transfers", BoolType::get(), false);
  }

  // remove duplicated constants
  vulkanRunCanonicalOptimizations(cloned_module);
  eliminateDeadCode(cloned_module);

  cloned_module.register_attribute(
      "optimized_for_vulkan", BoolType::get(), true);
  return cloned_module;
}

} // namespace jit
} // namespace torch
