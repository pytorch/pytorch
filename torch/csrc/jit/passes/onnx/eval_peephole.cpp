#include <torch/csrc/jit/passes/onnx/eval_peephole.h>
#include <torch/csrc/jit/passes/onnx/helper.h>
#include <torch/torch.h>

#include <c10/util/Optional.h>
#include <algorithm>

namespace torch {
namespace jit {

namespace onnx {
using namespace ::c10::onnx;
}

std::vector<at::Tensor> getValues(
    Node* node,
    const ValueToParamPairMap& valsToParamsMap) {
  size_t numInputs = node->inputs().size();
  std::vector<at::Tensor> inputTensorValues;
  inputTensorValues.reserve(numInputs);
  for (auto val : node->inputs()) {
    if (val->node()->kind() == prim::Param) {
      auto itr = valsToParamsMap.find(val);
      if (itr == valsToParamsMap.end()) {
        continue;
      }
      inputTensorValues.push_back(itr->second.second.toTensor());
    } else if (val->node()->kind() == onnx::Constant) {
      inputTensorValues.push_back(val->node()->t(attr::value));
    } else {
      continue;
    }
  }
  return inputTensorValues;
}

// This pass fuses Conv and BatchNorm into Conv node
// Conv and BatchNorm can be fused only if inputs for Batchnorm node:
// scale, bias, mean and var are all tensors of same shape (C) and
// if the size of the first dimension (dim 0) is the same between Conv
// input weight and Batchnorm input scale
static void fuseConvBatchNorm(Block* b, ValueToParamPairMap& valsToParamsMap) {
  for (auto it = b->nodes().begin(), end = b->nodes().end(); it != end; ++it) {
    for (auto* child_block : it->blocks()) {
      fuseConvBatchNorm(child_block, valsToParamsMap);
    }
    if (it->kind() == onnx::Conv) {
      if (it->output()->uses().size() != 1) {
        continue;
      }
      auto bnNode = it->output()->uses()[0].user;
      if (bnNode->kind() != onnx::BatchNormalization) {
        continue;
      }
      auto origconvNode = *it;
      auto epsilon = bnNode->f(attr::epsilon);
      auto w_conv_value = getValues(origconvNode, valsToParamsMap);
      if (w_conv_value.size() < 1 ||
          (origconvNode->inputs().size() == 3 && w_conv_value.size() != 2)) {
        continue;
      }

      auto bn_value = getValues(bnNode, valsToParamsMap);
      if (bn_value.size() != 4) {
        continue;
      }

      auto bn_scale = bn_value[0].clone();
      auto bn_B = bn_value[1].clone();
      auto bn_mean = bn_value[2].clone();
      auto bn_var = bn_value[3].clone();
      auto w_conv = w_conv_value[0].clone();
      at::Tensor b_conv;

      if (!bn_scale.is_floating_point() || !bn_B.is_floating_point() ||
          !bn_mean.is_floating_point() || !bn_var.is_floating_point() ||
          !w_conv.is_floating_point() || bn_scale.dim() != 1 ||
          bn_B.dim() != 1 || bn_mean.dim() != 1 || bn_var.dim() != 1 ||
          !(bn_scale.size(0) == bn_B.size(0)) ||
          !(bn_B.size(0) == bn_mean.size(0)) ||
          !(bn_mean.size(0) == bn_var.size(0)) || !(w_conv.dim() > 2) ||
          !(w_conv.size(0) == bn_scale.size(0))) {
        continue;
      }

      bn_var = bn_var.add(epsilon);
      bn_var = bn_var.sqrt();
      bn_scale = bn_scale.div(bn_var);

      // Calculate weight
      // NOLINTNEXTLINE(clang-diagnostic-sign-compare)
      for (size_t i = 0; i < w_conv.size(0); i++) {
        w_conv[i] = w_conv[i].mul(bn_scale[i]);
      }

      // Calculate bias
      if (origconvNode->inputs().size() == 3) {
        b_conv = w_conv_value[1].clone();
        b_conv = b_conv.sub(bn_mean);
        b_conv = b_conv.mul(bn_scale);
        b_conv = b_conv.add(bn_B);
      } else {
        bn_mean = bn_mean.mul(bn_scale);
        bn_B = bn_B.sub(bn_mean);
        b_conv = bn_B;
      }

      Node* convNode =
          b->owningGraph()->create(onnx::Conv, bnNode->outputs().size());
      for (size_t i = 0; i < convNode->outputs().size(); ++i) {
        convNode->outputs()[i]->copyMetadata(bnNode->outputs()[i]);
      }

      convNode->copyAttributes(*origconvNode);
      convNode->insertBefore(bnNode);
      convNode->addInput(origconvNode->inputs().at(0));

      auto conv_W = b->owningGraph()->addInput();
      valsToParamsMap.insert(
          {conv_W, std::make_pair(conv_W->debugName(), w_conv)});
      conv_W->inferTypeFrom(w_conv);
      convNode->addInput(conv_W);

      auto conv_B = b->addInput();
      valsToParamsMap.insert(
          {conv_B, std::make_pair(conv_B->debugName(), b_conv)});
      conv_B->inferTypeFrom(b_conv);
      convNode->addInput(conv_B);

      bnNode->replaceAllUsesWith(convNode);
      bnNode->removeAllInputs();
      it->removeAllInputs();
      bnNode->destroy();
      it.destroyCurrent();
    }
  }
}

void EvalPeepholeONNX(Block* b, ParamMap& paramsDict) {
  auto valsToParamsMap = buildValueToParamsMap(b, paramsDict);
  fuseConvBatchNorm(b, valsToParamsMap);
  buildParamsMapFromValueToParamsMap(valsToParamsMap, paramsDict);
  return;
}

} // namespace jit
} // namespace torch
