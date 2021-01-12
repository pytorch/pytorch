#include <torch/csrc/jit/passes/frozen_conv_folding.h>
#include <c10/util/Exception.h>
#include <torch/csrc/jit/ir/constants.h>
#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/jit_log.h>
#include <torch/csrc/jit/passes/dead_code_elimination.h>
#include <torch/csrc/jit/passes/fold_conv_bn.h>

namespace torch {
namespace jit {

using Tensor = at::Tensor;

bool nonConstantParameters(Node* n) {
  for (size_t i = 1; i < n->inputs().size(); i++) {
    if (n->inputs().at(i)->node()->kind() != prim::Constant) {
      return true;
    }
  }
  return false;
}

bool supportedConvNode(Node* n) {
  switch (n->kind()) {
    case aten::conv1d:
    case aten::conv2d:
    case aten::conv3d:
      return true;
    case aten::_convolution: {
      auto transposed_conv =
          constant_as<bool>(n->namedInput("transposed")).value_or(true);
      // dont handle transposed conv yet or not-constant transpose parameter
      return !transposed_conv;
    }
    default:
      return false;
  }
}

void FoldFrozenConvBatchnorm(Block* b) {
  for (Node* n : b->nodes()) {
    for (Block* block : n->blocks()) {
      FoldFrozenConvBatchnorm(block);
    }

    if (n->kind() == aten::batch_norm &&
        supportedConvNode(n->inputs().at(0)->node())) {
      auto conv = n->inputs().at(0)->node();
      auto bn = n;
      if (nonConstantParameters(conv) || nonConstantParameters(bn)) {
        continue;
      }
      if (conv->output()->uses().size() > 1) {
        continue;
      }

      auto bn_rm = constant_as<Tensor>(bn->namedInput("running_mean")).value();
      auto bn_rv = constant_as<Tensor>(bn->namedInput("running_var")).value();
      auto bn_eps = constant_as<double>(bn->namedInput("eps")).value();
      auto conv_w = constant_as<Tensor>(conv->namedInput("weight")).value();

      // implementation taken from torch/nn/utils/fusion.py
      Tensor conv_b;
      if (conv->namedInput("bias")->type() == NoneType::get()) {
        conv_b = at::zeros_like(bn_rm);
      } else {
        conv_b = constant_as<Tensor>(conv->namedInput("bias")).value();
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

      ConvBNParameters params;
      params.conv_w = conv_w;
      params.conv_b = conv_b;
      params.bn_rm = bn_rm;
      params.bn_rv = bn_rv;
      params.bn_eps = bn_eps;
      params.bn_w = bn_w;
      params.bn_b = bn_b;
      std::tuple<Tensor, Tensor> out = computeUpdatedConvWeightAndBias(params);
      WithInsertPoint guard(conv);
      auto fused_conv_w = b->owningGraph()->insertConstant(std::get<0>(out));
      auto fused_conv_b = b->owningGraph()->insertConstant(std::get<1>(out));
      auto conv_w_value = conv->namedInput("weight");
      auto conv_b_value = conv->namedInput("bias");

      fused_conv_w->setDebugName(conv_w_value->debugName() + "_fused_bn");
      fused_conv_b->setDebugName(conv_b_value->debugName() + "_fused_bn");

      conv->replaceInputWith(conv_w_value, fused_conv_w);
      conv->replaceInputWith(conv_b_value, fused_conv_b);

      bn->output()->replaceAllUsesWith(conv->output());
    }
  }
}

void FoldFrozenConvBatchnorm(std::shared_ptr<Graph>& graph) {
  FoldFrozenConvBatchnorm(graph->block());
  EliminateDeadCode(graph);
}

} // namespace jit
} // namespace torch
