#include <torch/csrc/jit/codegen/cuda/predicate_compute.h>

#include <torch/csrc/jit/codegen/cuda/arith.h>
#include <torch/csrc/jit/codegen/cuda/fusion.h>
#include <torch/csrc/jit/codegen/cuda/transform_iter.h>

namespace torch {
namespace jit {
namespace fuser {

bool PredicateCompute::hasPredicates(const kir::TensorIndex* ti) {
  std::vector<Bool*> preds;
  for (auto ind : ti->indices())
    if (FusionGuard::getCurFusion()->origin(ind) != nullptr)
      return true;
  return false;
}

std::vector<Bool*> PredicateCompute::computePredicates(
    const kir::TensorIndex* ti) {
  const TensorView* tv = ti->view();
  const std::vector<IterDomain*>& root = tv->getRootDomain();

  std::vector<Bool*> preds;

  bool no_pred_needed = true;
  for (auto id : tv->domain()->domain())
    if (id->getOrigin() != nullptr)
      no_pred_needed = false;

  if (no_pred_needed)
    return preds;

  TORCH_INTERNAL_ASSERT(
      root.size() == ti->nDims(),
      "Predicate compute received mismatched TensorView and TensorIndex.");

  Val* extent = nullptr;

  for (size_t i = 0; i < ti->nDims(); i++) {
    bool zero_ind = ti->index(i)->isZeroInt();
    bool simple_ind = ti->index(i)->getOrigin() == nullptr;

    if (root[i]->isBroadcast()) {
      preds.push_back(new Bool(true));
    } else if (simple_ind && !zero_ind) {
      preds.push_back(new Bool(true));
    } else if (zero_ind) {
      if (extent == nullptr) {
        extent = root[i]->extent();
      } else {
        extent = mul(extent, root[i]->extent());
      }
    } else {
      auto local_extent = root[i]->extent();
      if (extent != nullptr) {
        local_extent = mul(extent, local_extent);
      }
      Val* pred = lt(ti->index(i), local_extent);
      extent = nullptr;
      TORCH_INTERNAL_ASSERT(
          pred->getValType().value() == ValType::Scalar &&
          pred->getDataType().value() == DataType::Bool);
      preds.push_back(pred->as<Bool>());
    }
  }
  return preds;
}

} // namespace fuser
} // namespace jit
} // namespace torch
