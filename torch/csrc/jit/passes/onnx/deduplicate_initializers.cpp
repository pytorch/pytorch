#include <torch/csrc/jit/jit_log.h>
#include <torch/csrc/jit/passes/onnx/deduplicate_initializers.h>
#include <torch/csrc/jit/passes/onnx/helper.h>

#include <c10/util/irange.h>

namespace torch::jit {

namespace onnx {
using namespace ::c10::onnx;
}

void DeduplicateInitializers(
    std::shared_ptr<Graph>& g,
    ValueToParamPairMap& valsToParamsMap,
    bool (*comp)(at::Tensor&, at::Tensor&)) {
  auto is_same_tensor_as = [&valsToParamsMap, comp](Value* v1) {
    return [&valsToParamsMap, v1, comp](Value* v2) {
      if ((valsToParamsMap.find(v1) == valsToParamsMap.end()) ||
          (valsToParamsMap.find(v2) == valsToParamsMap.end())) {
        return false;
      }
      auto iv1 = valsToParamsMap.find(v1)->second.second;
      auto iv2 = valsToParamsMap.find(v2)->second.second;
      if (!iv1.isTensor() || !iv2.isTensor()) {
        return false;
      }
      auto t1 = iv1.toTensor();
      auto t2 = iv2.toTensor();
      return comp(t1, t2);
    };
  };
  std::vector<Value*> uniqueVals;
  std::vector<size_t> inputsIndicesToRemove;
  auto b = g->block();

  for (auto i : c10::irange(b->inputs().size())) {
    auto v = g->inputs().at(i);
    if (valsToParamsMap.find(v) == valsToParamsMap.end()) {
      // Skip model inputs
      continue;
    }
    auto it = std::find_if(
        uniqueVals.begin(), uniqueVals.end(), is_same_tensor_as(v));
    if (it == uniqueVals.end()) {
      uniqueVals.emplace_back(v);
    } else {
      inputsIndicesToRemove.emplace_back(i);
      auto id_node = g->create(onnx::Identity);
      id_node->insertAfter(g->block()->param_node());
      id_node->addInput(*it);
      id_node->output()->copyMetadata(v);
      id_node->copyMetadata(g->block()->param_node());
      v->replaceAllUsesWith(id_node->output());
    }
  }
  for (auto it = inputsIndicesToRemove.rbegin();
       it != inputsIndicesToRemove.rend();
       ++it) {
    valsToParamsMap.erase(g->inputs().at(*it));
    g->eraseInput(*it);
  }
}

bool DeduplicateInitializersByDataPtr(at::Tensor& t1, at::Tensor& t2) {
  return t1.sizes().equals(t2.sizes()) && t1.strides().equals(t2.strides()) &&
      (t1.has_storage() && t2.has_storage() && t1.data_ptr() == t2.data_ptr());
}

bool DeduplicateInitializersByValue(at::Tensor& t1, at::Tensor& t2) {
  if (t1.dtype() != t2.dtype() || !t1.sizes().equals(t2.sizes()) ||
      !t1.strides().equals(t2.strides())) {
    return false;
  }

  if (t1.device() != t2.device()) {
    return t1.to("cpu").equal(t2.to("cpu"));
  }

  return t1.equal(t2);
}

void DeduplicateInitializers(
    std::shared_ptr<Graph>& g,
    std::map<std::string, IValue>& paramsDict,
    bool is_train) {
  auto valsToParamsMap = buildValueToParamsMap(g->block(), paramsDict);
  // ONNX spec does not support parameters with shared memory.
  // This pass de-duplicate those parameters. Training is not affected.
  DeduplicateInitializers(g, valsToParamsMap, DeduplicateInitializersByDataPtr);
  if (!is_train) {
    // More aggressive parameters de-duplication based on tensor values.
    // Producing more compact model for inference.
    // For training, this pass is disabled,
    // because parameters may be updated differently.
    DeduplicateInitializers(g, valsToParamsMap, DeduplicateInitializersByValue);
  }
  buildParamsMapFromValueToParamsMap(valsToParamsMap, paramsDict);
}

} // namespace torch::jit
