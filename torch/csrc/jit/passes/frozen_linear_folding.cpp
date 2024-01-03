#include <torch/csrc/jit/ir/constants.h>
#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/passes/dead_code_elimination.h>
#include <torch/csrc/jit/passes/fold_linear_bn.h>
#include <torch/csrc/jit/passes/frozen_linear_folding.h>
#include <torch/csrc/jit/passes/utils/optimization_utils.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#else
#include <ATen/ops/ones_like.h>
#include <ATen/ops/zeros_like.h>
#endif

namespace torch {
namespace jit {

namespace {

using Tensor = at::Tensor;

bool supportedLinearNode(Node* n) {
  if (n->kind() == aten::linear) {
    return true;
  } else {
    return false;
  }
}

bool FoldFrozenLinearBatchnorm(Block* b) {
  bool graph_modified = false;
  for (Node* n : b->nodes()) {
    for (Block* block : n->blocks()) {
      graph_modified |= FoldFrozenLinearBatchnorm(block);
    }

    if (n->kind() == aten::batch_norm &&
        supportedLinearNode(n->inputs().at(0)->node())) {
      auto linear = n->inputs().at(0)->node();
      auto bn = n;

      if (nonConstantParameters(linear) || nonConstantParameters(bn)) {
        continue;
      }

      auto bn_rm_ivalue = bn->namedInput("running_mean");
      auto bn_rv_ivalue = bn->namedInput("running_var");

      // check running_mean and running_var has value, if they are
      // None(track_running_stats=False), skipping the folding path.
      if (bn_rm_ivalue->type() == NoneType::get() &&
          bn_rv_ivalue->type() == NoneType::get()) {
        continue;
      }

      auto bn_rm = constant_as<Tensor>(bn->namedInput("running_mean")).value();
      auto bn_rv = constant_as<Tensor>(bn->namedInput("running_var")).value();
      auto bn_eps = constant_as<double>(bn->namedInput("eps")).value();
      auto linear_w = constant_as<Tensor>(linear->namedInput("weight")).value();

      // implementation taken from torch/nn/utils/fusion.py
      Tensor linear_b;
      if (linear->namedInput("bias")->type() == NoneType::get()) {
        at::ScalarType bias_dtype = bn_rm.scalar_type();
        at::ScalarType weight_dtype = linear_w.scalar_type();
        at::DeviceType weight_device = linear_w.device().type();
        if (weight_device == at::kCUDA &&
            (weight_dtype == at::kHalf || weight_dtype == at::kBFloat16) &&
            bias_dtype == at::kFloat) {
          bias_dtype = weight_dtype;
        }
        linear_b = at::zeros_like(bn_rm, at::TensorOptions().dtype(bias_dtype));
      } else {
        linear_b = constant_as<Tensor>(linear->namedInput("bias")).value();
      }
      Tensor bn_w;
      if (bn->namedInput("weight")->type() == NoneType::get()) {
        bn_w = at::ones_like(bn_rm);
      } else {
        bn_w = constant_as<Tensor>(bn->namedInput("weight")).value();
      }
      Tensor bn_b;
      if (n->namedInput("bias")->type() == NoneType::get()) {
        bn_b = at::zeros_like(bn_rm);
      } else {
        bn_b = constant_as<Tensor>(bn->namedInput("bias")).value();
      }

      LinearBNParameters params;
      params.linear_w = linear_w;
      params.linear_b = linear_b;
      params.bn_rm = bn_rm;
      params.bn_rv = bn_rv;
      params.bn_eps = bn_eps;
      params.bn_w = bn_w;
      params.bn_b = bn_b;
      std::tuple<Tensor, Tensor> out =
          computeUpdatedLinearWeightAndBias(params);
      WithInsertPoint guard(linear);
      auto fused_linear_w = b->owningGraph()->insertConstant(std::get<0>(out));
      auto fused_linear_b = b->owningGraph()->insertConstant(std::get<1>(out));
      auto linear_w_value = linear->namedInput("weight");
      auto linear_b_value = linear->namedInput("bias");

      fused_linear_w->setDebugName(linear_w_value->debugName() + "_fused_bn");
      fused_linear_b->setDebugName(linear_b_value->debugName() + "_fused_bn");

      linear->replaceInputWith(linear_w_value, fused_linear_w);
      linear->replaceInputWith(linear_b_value, fused_linear_b);

      bn->output()->replaceAllUsesWith(linear->output());
      graph_modified = true;
    }
  }
  return graph_modified;
}

} // namespace

bool FoldFrozenLinearBatchnorm(std::shared_ptr<Graph>& graph) {
  bool graph_modified = FoldFrozenLinearBatchnorm(graph->block());
  EliminateDeadCode(graph);
  return graph_modified;
}

} // namespace jit
} // namespace torch
