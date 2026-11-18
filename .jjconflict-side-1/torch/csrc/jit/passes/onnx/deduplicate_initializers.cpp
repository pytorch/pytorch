#include <torch/csrc/jit/jit_log.h>
#include <torch/csrc/jit/passes/onnx/deduplicate_initializers.h>
#include <torch/csrc/jit/passes/onnx/helper.h>

#include <c10/util/hash.h>
#include <c10/util/irange.h>

#include <functional>
#include <unordered_set>

namespace torch::jit {

namespace onnx {
using namespace ::c10::onnx;
}

struct HashValue {
  HashValue(ValueToParamPairMap& valsToParamsMap, bool compare_by_ptr)
      : valsToParamsMap_(valsToParamsMap), compare_by_ptr_(compare_by_ptr) {}

  size_t operator()(Value* v) const {
    auto t = valsToParamsMap_.find(v)->second.second.toTensor();

    if (compare_by_ptr_) {
      // Hash by metadata + data pointer
      return at::get_hash(
          t.sizes(), t.strides(), t.has_storage() ? t.data_ptr() : 0);
    }

    // Hash by metadata + first element value (if exists). This is a fast
    // approximation of hashing by the whole tensor value, which can be
    // expensive for large tensors.
    double first_elem_double = 0.0;
    int64_t first_elem_int64 = 0;
    uint64_t first_elem_uint64 = 0;

    if (t.numel() > 0 && t.has_storage()) {
      auto scalar = t.reshape(-1)[0].item();

      if (scalar.isFloatingPoint()) {
        first_elem_double = scalar.to<double>();
      } else if (scalar.isIntegral(/*includeBool=*/true)) {
        if (scalar.isUnsigned()) {
          first_elem_uint64 = scalar.to<uint64_t>();
        } else {
          first_elem_int64 = scalar.to<int64_t>();
        }
      }
    }

    return at::get_hash(
        first_elem_double,
        first_elem_int64,
        first_elem_uint64,
        t.sizes(),
        t.strides());
  }

 private:
  ValueToParamPairMap& valsToParamsMap_;
  bool compare_by_ptr_;
};

struct CompareValue {
  CompareValue(std::function<bool(Value*, Value*)> is_same_tensor_as)
      : is_same_tensor_as_(is_same_tensor_as) {}

  bool operator()(Value* v1, Value* v2) const {
    return is_same_tensor_as_(v1, v2);
  }

 private:
  std::function<bool(Value*, Value*)> is_same_tensor_as_;
};

// forward declaration
static bool DeduplicateInitializersByDataPtr(at::Tensor& t1, at::Tensor& t2);

static void DeduplicateInitializers(
    std::shared_ptr<Graph>& g,
    ValueToParamPairMap& valsToParamsMap,
    bool (*comp)(at::Tensor&, at::Tensor&)) {
  auto is_same_tensor_as = [&valsToParamsMap, comp](Value* v1, Value* v2) {
    auto t1 = valsToParamsMap.find(v1)->second.second.toTensor();
    auto t2 = valsToParamsMap.find(v2)->second.second.toTensor();
    return comp(t1, t2);
  };

  bool compare_by_ptr = comp == &DeduplicateInitializersByDataPtr;
  std::unordered_set<Value*, HashValue, CompareValue> uniqueVals(
      0,
      HashValue(valsToParamsMap, compare_by_ptr),
      CompareValue(is_same_tensor_as));
  std::vector<size_t> inputsIndicesToRemove;
  auto b = g->block();

  for (auto i : c10::irange(b->inputs().size())) {
    auto v = g->inputs().at(i);
    auto vals_to_param_it = valsToParamsMap.find(v);

    // Skip parameters without initializers
    if (vals_to_param_it == valsToParamsMap.end()) {
      continue;
    }

    // Skip non-tensors
    if (!vals_to_param_it->second.second.isTensor()) {
      continue;
    }

    auto it = uniqueVals.insert(v);
    if (!it.second) {
      // Same value already exists
      inputsIndicesToRemove.emplace_back(i);
      auto id_node = g->create(onnx::Identity);
      id_node->insertAfter(g->block()->param_node());
      id_node->addInput(*it.first);
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

static bool DeduplicateInitializersByDataPtr(at::Tensor& t1, at::Tensor& t2) {
  return t1.sizes().equals(t2.sizes()) && t1.strides().equals(t2.strides()) &&
      (t1.has_storage() && t2.has_storage() && t1.data_ptr() == t2.data_ptr());
}

static bool DeduplicateInitializersByValue(at::Tensor& t1, at::Tensor& t2) {
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
