#include <tuple>

#include <torch/csrc/jit/ir/alias_analysis.h>
#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/ir/subgraph_matcher.h>
#include <torch/csrc/jit/passes/constant_pooling.h>
#include <torch/csrc/jit/passes/constant_propagation.h>
#include <torch/csrc/jit/passes/conv2d_batch_norm_folding.h>
#include <torch/csrc/jit/passes/fuse_linear.h>
#include <torch/csrc/jit/passes/graph_rewrite_helper.h>
#include <torch/csrc/jit/passes/subgraph_rewrite.h>

namespace torch {
namespace jit {

using graph_rewrite_helper::computeUpdatedConvWeightAndBias;
using graph_rewrite_helper::ConvBNParameters;
using graph_rewrite_helper::PatternInfo;

namespace {

ConvBNParameters extractConvBNParams(
    Value* conv_weight,
    Value* conv_bias,
    Value* bn_weight,
    Value* bn_bias,
    Value* bn_running_mean,
    Value* bn_running_var,
    Value* eps) {
  ConvBNParameters params;
  params.bn_rm = toIValue(bn_running_mean)->toTensor();
  params.bn_rv = toIValue(bn_running_var)->toTensor();
  params.bn_eps = toIValue(eps)->toDouble();
  auto val = toIValue(bn_weight);
  if (val && !val->isNone()) {
    params.bn_w = val->toTensor();
  } else {
    params.bn_w = at::ones_like(params.bn_rm);
  }
  val = toIValue(bn_bias);
  if (val && !val->isNone()) {
    params.bn_b = val->toTensor();
  } else {
    params.bn_b = at::zeros_like(params.bn_rm);
  }

  params.conv_w = toIValue(conv_weight)->toTensor();
  val = toIValue(conv_bias);
  if (val && !val->isNone()) {
    params.conv_b = val->toTensor();
  } else {
    params.conv_b = at::zeros_like(params.bn_rm);
  }
  return params;
}

void removeBatchNormNodes(std::unordered_set<Node*>& batch_norm_nodes) {
  std::vector<Value*> values_to_remove;
  for (auto n : batch_norm_nodes) {
    // at::ArrayRef<Value*> input_vals = n->inputs();
    // Remove batch norm's weight, bias, mean, variance.
    n->removeAllInputs();
  }
  for (auto n : batch_norm_nodes) {
    n->destroy();
  }
  for (Value* val : values_to_remove) {
    val->node()->removeAllInputs();
    val->node()->destroy();
  }
}

void transformAndCopyConvWeightAndBias(
    const std::shared_ptr<Graph>& graph_ptr,
    Value* conv_weight,
    Value* conv_bias,
    Node* conv_node,
    const ConvBNParameters& params) {
  std::unique_ptr<AliasDb> aliasDb = torch::make_unique<AliasDb>(graph_ptr);
  Graph& graph = *graph_ptr;
  const auto& new_weight_bias = computeUpdatedConvWeightAndBias(params);
  auto weight_tensor = toIValue(conv_weight)->toTensor();
  auto new_weight_val =
      *(tryInsertConstant(graph, std::get<0>(new_weight_bias)));
  Node* weight_node = conv_weight->node();
  conv_weight->replaceAllUsesWith(new_weight_val);
  aliasDb->moveBeforeTopologicallyValid(new_weight_val->node(), conv_node);
  weight_node->removeAllInputs();
  weight_node->destroy();

  auto optional_val = toIValue(conv_bias);
  TORCH_CHECK(optional_val, "Bias for conv2d module must exist");
  auto bias_tensor = std::get<1>(new_weight_bias);
  if (optional_val->isNone()) {
    Value* new_bias_val = *(tryInsertConstant(graph, bias_tensor));
    constexpr size_t conv_bias_index = 2;
    conv_node->replaceInput(conv_bias_index, new_bias_val);
    aliasDb->moveBeforeTopologicallyValid(new_bias_val->node(), conv_node);
  } else {
    TORCH_CHECK(conv_bias->uses().size() == 1,
        "Bias of conv should have only one use.");
    auto orig_bias_tensor = toIValue(conv_bias)->toTensor();
    Value* new_bias_val = *(tryInsertConstant(graph, bias_tensor));
    Node* bias_node = conv_bias->node();
    bias_node->output(0)->replaceAllUsesWith(new_bias_val);
    aliasDb->moveBeforeTopologicallyValid(new_bias_val->node(), conv_node);
    bias_node->removeAllInputs();
    bias_node->destroy();
  }
}

void foldBatchNorm2dInConv2d(std::shared_ptr<Graph>& graph) {
  // Replace _convolution with conv2d
  graph_rewrite_helper::replaceConvolutionWithConv2d(graph);
  std::unordered_set<Node*> nodes_to_delete;

  std::string conv2d_bn_pattern = R"(
    graph(%input, %conv_weight, %conv_bias, %stride:int[],
        %padding:int[], %dilation:int[], %groups:int,
        %bn_weight, %bn_bias, %bn_running_mean, %bn_running_var,
        %training, %momentum, %eps, %cudnn_enabled):
        %conv_out = aten::conv2d(%input, %conv_weight, %conv_bias, %stride,
            %padding, %dilation, %groups)
        %bn_out = aten::batch_norm(%conv_out, %bn_weight, %bn_bias,
            %bn_running_mean, %bn_running_var, %training, %momentum,
            %eps, %cudnn_enabled)
        return (%bn_out) )";
  const PatternInfo& pattern = PatternInfo::parse_from_str(conv2d_bn_pattern);
  const Graph& pattern_graph = *pattern.pattern_graph;
  const auto& vmap = pattern.vmap;
  Value* pattern_conv_weight = vmap.at("conv_weight");
  Value* pattern_conv_bias = vmap.at("conv_bias");
  Value* pattern_bn_weight = vmap.at("bn_weight");
  Value* pattern_bn_bias = vmap.at("bn_bias");
  Value* pattern_bn_running_mean = vmap.at("bn_running_mean");
  Value* pattern_bn_running_var = vmap.at("bn_running_var");
  Value* pattern_bn_eps = vmap.at("eps");
  Value* pattern_training = vmap.at("training");
  Value* conv_out = vmap.at("conv_out");
  Value* bn_out = vmap.at("bn_out");

  const auto& matches = findPatternMatches(pattern_graph, *graph);
  for (const auto& match : matches) {
    Value* matched_conv_weight = match.values_map.at(pattern_conv_weight);
    Value* matched_conv_bias = match.values_map.at(pattern_conv_bias);
    Value* matched_bn_weight = match.values_map.at(pattern_bn_weight);
    Value* matched_bn_bias = match.values_map.at(pattern_bn_bias);
    Value* matched_bn_running_mean =
        match.values_map.at(pattern_bn_running_mean);
    Value* matched_bn_running_var = match.values_map.at(pattern_bn_running_var);
    Value* matched_bn_eps = match.values_map.at(pattern_bn_eps);
    Value* matched_training = match.values_map.at(pattern_training);
    Value* matched_conv_out = match.values_map.at(conv_out);
    Value* matched_bn_out = match.values_map.at(bn_out);

    if (matched_conv_out->uses().size() > 1) {
      continue;
    }

    TORCH_CHECK(
        !(toIValue(matched_training)->toBool()),
        "Training must be set to false for batch norm folding");
    const auto params = extractConvBNParams(
        matched_conv_weight,
        matched_conv_bias,
        matched_bn_weight,
        matched_bn_bias,
        matched_bn_running_mean,
        matched_bn_running_var,
        matched_bn_eps);
    transformAndCopyConvWeightAndBias(
        graph,
        matched_conv_weight,
        matched_conv_bias,
        matched_conv_out->node(),
        params);
    matched_bn_out->replaceAllUsesWith(matched_conv_out);
    nodes_to_delete.insert(matched_bn_out->node());
  }
  removeBatchNormNodes(nodes_to_delete);
}

} // namespace

// This pass depends on freezing to convert attributes to constants.
// It also depends on not having inserted XNNPACK ops since those ops
// pack weights and bias into a custom class.
void FoldConvBatchNorm2dOfFrozenTracedModuleGraph(
    std::shared_ptr<Graph>& graph) {
  ConstantPooling(graph);
  ConstantPropagation(graph);
  graph_rewrite_helper::replaceConvolutionWithConv2d(graph);
  foldBatchNorm2dInConv2d(graph);
  LOG(INFO)
      << "Finished FoldConv2dBatchNorm pass. If you did not get"
         " expected result, you may not have run the freeze_module pass prior\n";
}

} // namespace jit
} // namespace torch
