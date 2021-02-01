#include <ATen/Utils.h>
#include <c10/core/ScalarType.h>
#include <c10/util/Exception.h>
#include <torch/csrc/jit/ir/constants.h>
#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/jit_log.h>
#include <torch/csrc/jit/passes/constant_propagation.h>
#include <torch/csrc/jit/passes/dead_code_elimination.h>
#include <torch/csrc/jit/passes/fold_conv_bn.h>
#include <torch/csrc/jit/passes/frozen_conv_folding.h>
#include <torch/csrc/jit/tensorexpr/types.h>

namespace torch {
namespace jit {

namespace {

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

bool supportedAddOrSub(Node* n) {
  static const OperatorSet add_set{
      "aten::add.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor",
      "aten::add.Scalar(Tensor self, Scalar other, Scalar alpha=1) -> Tensor",
      // sub is equivalent to add
      "aten::sub.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor",
      "aten::sub.Scalar(Tensor self, Scalar other, Scalar alpha=1) -> Tensor",
  };
  return n->isMemberOf(add_set);
}

// In order to fuse add/sub/mul/div with conv, the dimensions of its
// constant tensor must satisfy the following:
// - with resizing, broadcast to w/ weight/bias tensor shape
// - broadcast to the conv output shape
// It needs to have a shape that can resize to weight/bias
// tensor shape because we need to run the op with the conv
// weights/bias without changing their sizes.
// It needs to broadcast to the conv output shape so that we do
// accidentally change the shape of op output by pre-fusing it
// compared to eager.
// The only dimension value shared by weight/bias/conv output
// is they all contain a dim with value = channels-out. In the
// conv output tensor, this is in the second dimension,
// so the pointwise op tensor may have a second dimension of
// value == channels-out, but all the other dimensions have to be 1
bool opDoesNotBroadCastWithConv(Tensor& op_tensor, Tensor& weight_tensor) {
  if (op_tensor.ndimension() > weight_tensor.ndimension()) {
    return false;
  }
  for (int64_t i = op_tensor.ndimension() - 1; i >= 0; i--) {
    // channels-out dimension == weight_tensor.size(0)
    if (i == 1 && op_tensor.size(i) == weight_tensor.size(0)) {
      continue;
    }
    if (op_tensor.size(i) != 1) {
      return false;
    }
  }
  return true;
}

bool checkConvAndBroadcastingOpPreConditions(Node* conv, Node* op) {
  if (nonConstantParameters(conv) || nonConstantParameters(op)) {
    return false;
  }

  if (conv->output()->uses().size() > 1) {
    return false;
  }

  auto conv_w = constant_as<Tensor>(conv->namedInput("weight")).value();
  Tensor weight_tensor =
      constant_as<Tensor>(conv->namedInput("weight")).value();

  // avoid fusing op that causes type promotion
  // resticting to float avoids int/float difficulties with scalar overload
  if (!weight_tensor.is_floating_point()) {
    return false;
  }

  if (op->inputs().at(1)->type()->cast<TensorType>()) {
    auto op_tensor = constant_as<Tensor>(op->inputs().at(1)).value();
    if (!opDoesNotBroadCastWithConv(op_tensor, weight_tensor)) {
      return false;
    }
    if (!op_tensor.is_floating_point()) {
      return false;
    }
    if (c10::promoteTypes(
            op_tensor.scalar_type(), weight_tensor.scalar_type()) !=
        weight_tensor.scalar_type()) {
      return false;
    }
  }
  return true;
}

Tensor resizeConstantScalarOrTensorToShape(
    Value* v,
    const std::vector<int64_t>& shape,
    at::TensorOptions options) {
  Tensor ret_tensor;
  if (v->type()->cast<TensorType>()) {
    ret_tensor = constant_as<Tensor>(v).value();
  } else {
    ret_tensor = at::zeros(shape, options);
    if (v->type()->cast<IntType>()) {
      ret_tensor.fill_(constant_as<int64_t>(v).value());
    } else {
      ret_tensor.fill_(constant_as<double>(v).value());
    }
  }

  if (ret_tensor.numel() == 1) {
    // expand errors if the shape input has less # dims than the tensor input
    ret_tensor = ret_tensor.reshape({1});
    ret_tensor = ret_tensor.expand(shape);
  } else {
    TORCH_INTERNAL_ASSERT(ret_tensor.numel() == at::prod_intlist(shape));
    ret_tensor = ret_tensor.view(shape);
  }
  return ret_tensor;
}

void FoldFrozenConvAddOrSub(Block* b) {
  for (Node* n : b->nodes()) {
    for (Block* block : n->blocks()) {
      FoldFrozenConvAddOrSub(block);
    }

    if (supportedAddOrSub(n) && supportedConvNode(n->inputs().at(0)->node())) {
      auto conv = n->inputs().at(0)->node();
      auto add_or_div = n;

      if (!checkConvAndBroadcastingOpPreConditions(conv, add_or_div)) {
        continue;
      }

      Tensor weight_tensor =
          constant_as<Tensor>(conv->namedInput("weight")).value();

      Tensor add_or_sub_tensor = resizeConstantScalarOrTensorToShape(
          add_or_div->inputs().at(1),
          {weight_tensor.size(0)},
          weight_tensor.options());
      Tensor bias;
      if (conv->namedInput("bias")->type() == NoneType::get()) {
        bias = at::zeros_like(add_or_sub_tensor);
      } else {
        bias = constant_as<Tensor>(conv->namedInput("bias")).value();
      }

      WithInsertPoint guard(conv);

      add_or_div->replaceInputWith(
          conv->output(), b->owningGraph()->insertConstant(bias));
      add_or_div->replaceInput(
          1, b->owningGraph()->insertConstant(add_or_sub_tensor));

      auto stack_out = runNodeIfInputsAreConstant(add_or_div);
      TORCH_INTERNAL_ASSERT(stack_out && stack_out->size() == 1);
      Tensor fuse_bias = (*stack_out)[0].toTensor();

      auto fused_conv_b = b->owningGraph()->insertConstant(fuse_bias);
      auto conv_b_value = conv->namedInput("bias");

      fused_conv_b->setDebugName(
          conv_b_value->debugName() + "_fused_" +
          add_or_div->kind().toUnqualString());
      conv->replaceInputWith(conv_b_value, fused_conv_b);
      add_or_div->output()->replaceAllUsesWith(conv->output());
      // DCE run after cleans up nodes
    }
  }
}

bool supportedMulOrDiv(Node* n) {
  static const OperatorSet add_set{
      "aten::mul.Tensor(Tensor self, Tensor other) -> Tensor",
      "aten::mul.Scalar(Tensor self, Scalar other) -> Tensor",
      // div is equivalent to mul
      "aten::div.Tensor(Tensor self, Tensor other) -> Tensor",
      "aten::div.Scalar(Tensor self, Scalar other) -> Tensor",
  };
  return n->isMemberOf(add_set);
}

void FoldFrozenConvMulOrDiv(Block* b) {
  for (Node* n : b->nodes()) {
    for (Block* block : n->blocks()) {
      FoldFrozenConvMulOrDiv(block);
    }

    if (supportedMulOrDiv(n) && supportedConvNode(n->inputs().at(0)->node())) {
      auto conv = n->inputs().at(0)->node();
      auto mul_or_div = n;

      if (!checkConvAndBroadcastingOpPreConditions(conv, mul_or_div)) {
        continue;
      }

      Tensor weight_tensor =
          constant_as<Tensor>(conv->namedInput("weight")).value();
      int64_t out_channels = weight_tensor.size(0);

      // We've already verified that the second input has numel == 1 or
      // channels-out resize it to the shape that will broadcast to
      // weight_tensor when the op is run so we dont change weight size
      std::vector<int64_t> weight_compatible_size = {out_channels};
      for (int64_t i = 1; i < weight_tensor.ndimension(); ++i) {
        weight_compatible_size.push_back(1);
      }

      WithInsertPoint guard(conv);

      Tensor mul_tensor = resizeConstantScalarOrTensorToShape(
          mul_or_div->inputs().at(1),
          weight_compatible_size,
          weight_tensor.options());

      // First fold with weight tensor
      mul_or_div->replaceInputWith(
          conv->output(), b->owningGraph()->insertConstant(weight_tensor));
      mul_or_div->replaceInput(1, b->owningGraph()->insertConstant(mul_tensor));

      auto stack_out = runNodeIfInputsAreConstant(mul_or_div);
      TORCH_INTERNAL_ASSERT(stack_out && stack_out->size() == 1);
      Tensor fuse_weight = (*stack_out)[0].toTensor();

      auto fused_conv_weight = b->owningGraph()->insertConstant(fuse_weight);
      auto conv_weight_value = conv->namedInput("weight");

      fused_conv_weight->setDebugName(
          conv_weight_value->debugName() + "_fused_" +
          mul_or_div->kind().toUnqualString());
      conv->replaceInputWith(conv_weight_value, fused_conv_weight);
      mul_or_div->output()->replaceAllUsesWith(conv->output());

      // now fold with bias tensor
      if (conv->namedInput("bias")->type() != NoneType::get()) {
        Tensor bias = constant_as<Tensor>(conv->namedInput("bias")).value();
        // bias is of shape {channels_out}
        auto mul_tensor = resizeConstantScalarOrTensorToShape(
            mul_or_div->inputs().at(1), {out_channels}, bias.options());

        mul_or_div->replaceInput(0, b->owningGraph()->insertConstant(bias));
        mul_or_div->replaceInput(
            1, b->owningGraph()->insertConstant(mul_tensor));

        auto stack_out = runNodeIfInputsAreConstant(mul_or_div);
        TORCH_INTERNAL_ASSERT(stack_out && stack_out->size() == 1);
        Tensor fuse_bias = (*stack_out)[0].toTensor();

        auto fused_conv_bias = b->owningGraph()->insertConstant(fuse_bias);
        auto conv_b_value = conv->namedInput("bias");

        fused_conv_weight->setDebugName(
            conv_b_value->debugName() + "_fused_" +
            mul_or_div->kind().toUnqualString());
        conv->replaceInputWith(conv_b_value, fused_conv_bias);
      }
      // DCE run after cleans up nodes
    }
  }
}

} // namespace

void FoldFrozenConvBatchnorm(std::shared_ptr<Graph>& graph) {
  FoldFrozenConvBatchnorm(graph->block());
  EliminateDeadCode(graph);
}

void FoldFrozenConvAddOrSub(std::shared_ptr<Graph>& graph) {
  FoldFrozenConvAddOrSub(graph->block());
  EliminateDeadCode(graph);
}

void FoldFrozenConvMulOrDiv(std::shared_ptr<Graph>& graph) {
  FoldFrozenConvMulOrDiv(graph->block());
  EliminateDeadCode(graph);
}

} // namespace jit
} // namespace torch
