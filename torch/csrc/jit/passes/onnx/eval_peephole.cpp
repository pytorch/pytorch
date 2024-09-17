#include <torch/csrc/jit/jit_log.h>
#include <torch/csrc/jit/passes/onnx/eval_peephole.h>
#include <torch/csrc/jit/passes/onnx/helper.h>
#include <torch/torch.h>

#include <c10/util/irange.h>
#include <algorithm>

namespace torch::jit {

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
// Conv and BatchNorm can be fused only if inputs for BatchNorm node:
// scale, bias, mean and var are all tensors of same shape (C) and
// if the size of the first dimension (dim 0) is the same between Conv
// input weight and BatchNorm input scale.
static void fuseConvBatchNorm(Block* b, ValueToParamPairMap& valsToParamsMap) {
  for (auto it = b->nodes().begin(), end = b->nodes().end(); it != end; ++it) {
    for (auto* child_block : it->blocks()) {
      fuseConvBatchNorm(child_block, valsToParamsMap);
    }
    if (it->kind() == onnx::Conv) {
      auto oldConv = *it;
      if (oldConv->outputs().at(0)->uses().size() != 1) {
        continue;
      }
      auto bnNode = oldConv->outputs().at(0)->uses()[0].user;
      if (bnNode->kind() != onnx::BatchNormalization) {
        continue;
      }

      if (oldConv->outputs().size() !=
          bnNode->outputs().size()) { // BN layer is not in eval mode
        continue;
      }

      auto epsilon = bnNode->f(attr::epsilon);
      auto convInputVals = getValues(oldConv, valsToParamsMap);
      if (convInputVals.empty() ||
          (oldConv->inputs().size() == 3 && convInputVals.size() != 2)) {
        continue;
      }

      auto bnInputVals = getValues(bnNode, valsToParamsMap);
      if (bnInputVals.size() != 4) {
        continue;
      }

      // See
      // https://github.com/onnx/onnx/blob/master/docs/Operators.md#BatchNormalization
      auto bnScale = bnInputVals[0].clone();
      auto bnB = bnInputVals[1].clone();
      auto bnMean = bnInputVals[2].clone();
      auto bnVar = bnInputVals[3].clone();
      // See https://github.com/onnx/onnx/blob/master/docs/Operators.md#Conv
      auto convW = convInputVals[0].clone();
      at::Tensor convB;

      if (!bnScale.is_floating_point() || !bnB.is_floating_point() ||
          !bnMean.is_floating_point() || !bnVar.is_floating_point() ||
          !convW.is_floating_point() || bnScale.dim() != 1 || bnB.dim() != 1 ||
          bnMean.dim() != 1 || bnVar.dim() != 1 ||
          !(bnScale.size(0) == bnB.size(0)) ||
          !(bnB.size(0) == bnMean.size(0)) ||
          !(bnMean.size(0) == bnVar.size(0)) || !(convW.dim() > 2) ||
          !(convW.size(0) == bnScale.size(0))) {
        continue;
      }

      bnVar = bnVar.add(epsilon);
      bnVar = bnVar.sqrt();
      bnScale = bnScale.div(bnVar);

      // Calculate weight
      for (const auto i : c10::irange(convW.size(0))) {
        convW[i] = convW[i].mul(bnScale[i]);
      }

      // Calculate bias
      if (oldConv->inputs().size() == 3) {
        convB = convInputVals[1].clone();
        convB = convB.sub(bnMean);
        convB = convB.mul(bnScale);
        convB = convB.add(bnB);
      } else {
        bnMean = bnMean.mul(bnScale);
        bnB = bnB.sub(bnMean);
        convB = bnB;
      }

      Node* newConv = b->owningGraph()->create(onnx::Conv, 1);
      newConv->outputs().at(0)->copyMetadata(bnNode->outputs().at(0));

      newConv->copyAttributes(*oldConv);
      newConv->insertBefore(bnNode);
      newConv->addInput(oldConv->inputs().at(0));
      newConv->copyMetadata(oldConv);

      auto newConvW = b->owningGraph()->addInput();
      valsToParamsMap.insert(
          {newConvW, std::make_pair(newConvW->debugName(), convW)});
      newConvW->inferTypeFrom(convW);
      newConv->addInput(newConvW);

      auto newConvB = b->owningGraph()->addInput();
      valsToParamsMap.insert(
          {newConvB, std::make_pair(newConvB->debugName(), convB)});
      newConvB->inferTypeFrom(convB);
      newConv->addInput(newConvB);

      bnNode->outputs().at(0)->replaceAllUsesWith(newConv->outputs().at(0));
      bnNode->destroy();
      it.destroyCurrent();
    }
  }
}

void EvalPeepholeONNX(Block* b, ParamMap& paramsDict) {
  auto valsToParamsMap = buildValueToParamsMap(b, paramsDict);
  fuseConvBatchNorm(b, valsToParamsMap);
  buildParamsMapFromValueToParamsMap(valsToParamsMap, paramsDict);
}

void EvalPeepholeONNX(std::shared_ptr<Graph>& g, ParamMap& paramsDict) {
  EvalPeepholeONNX(g->block(), paramsDict);
  GRAPH_DUMP("After EvalPeepholeONNX:", g);
}

} // namespace torch::jit
