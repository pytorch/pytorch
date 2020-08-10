#include <torch/csrc/jit/codegen/cuda/predicate_compute.h>

#include <torch/csrc/jit/codegen/cuda/arith.h>
#include <torch/csrc/jit/codegen/cuda/fusion.h>
#include <torch/csrc/jit/codegen/cuda/transform_iter.h>

namespace torch {
namespace jit {
namespace fuser {

std::vector<kir::Bool*> PredicateCompute::computePredicates(
    const TensorView* tv,
    const std::vector<Val*>& indices) {
  const std::vector<IterDomain*>& root = tv->getRootDomain();
  TORCH_INTERNAL_ASSERT(root.size() == indices.size());

  bool no_pred_needed = true;
  for (auto id : tv->domain()->domain()) {
    if (id->getOrigin() != nullptr) {
      no_pred_needed = false;
    }
  }

  if (no_pred_needed) {
    return {};
  }

  Val* extent = nullptr;
  std::vector<kir::Bool*> preds;

  for (size_t i = 0; i < indices.size(); i++) {
    const bool zero_ind = indices[i]->isZeroInt();
    const bool simple_ind = indices[i]->getOrigin() == nullptr;

    if (root[i]->isBroadcast()) {
      preds.push_back(new kir::Bool(true));
    } else if (simple_ind && !zero_ind) {
      preds.push_back(new kir::Bool(true));
    } else if (zero_ind) {
      if (extent == nullptr) {
        extent = kir::lowerValue(root[i]->extent());
      } else {
        extent = kir::mulExpr(extent, kir::lowerValue(root[i]->extent()));
      }
    } else {
      auto local_extent = kir::lowerValue(root[i]->extent());
      if (extent != nullptr) {
        local_extent = kir::mulExpr(extent, local_extent);
      }
      auto pred = kir::ltExpr(indices[i], local_extent);
      extent = nullptr;
      TORCH_INTERNAL_ASSERT(
          pred->getValType().value() == ValType::KirScalar &&
          pred->getDataType().value() == DataType::Bool);
      preds.push_back(pred->as<kir::Bool>());
    }
  }
  return preds;
}

} // namespace fuser
} // namespace jit
} // namespace torch
