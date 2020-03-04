#pragma once

#include <torch/csrc/jit/fuser/common/index_compute.h>

namespace torch {
namespace jit {
namespace fuser {

struct PredicateCompute {
  static bool hasPredicates(const TensorView* tv, const std::vector<Int*> _indices) {
    std::vector<Int*> preds;
    for (auto ind : _indices)
      if (FusionGuard::getCurFusion()->origin(ind) != nullptr)
        return true;
    return false;
  }

  static std::vector<Int*> computePredicates(
      const TensorView* tv,
      const std::vector<Int*> _indices) {
    std::vector<Int*> preds;
    if (!hasPredicates(tv, _indices))
      return preds;

    TensorDomain* root = TransformIter::getRoot(tv->domain());
    TORCH_CHECK(root->size() == _indices.size());
    for (decltype(_indices.size()) i{0}; i < _indices.size(); i++)

      if (FusionGuard::getCurFusion()->origin(_indices[i]) != nullptr) {
        Val* pred = lt(_indices[i], root->axis(i)->size());
        TORCH_CHECK(
            pred->getValType().value() == ValType::Scalar &&
            pred->getDataType().value() == DataType::Int);
        preds.push_back(static_cast<Int*>(pred));
      } else {
        preds.push_back(new Int(1));
      }

    return preds;
  }
};

} // namespace fuser
} // namespace jit
} // namespace torch
