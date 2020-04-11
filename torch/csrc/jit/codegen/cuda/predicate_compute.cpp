#include <torch/csrc/jit/codegen/cuda/predicate_compute.h>

#include <torch/csrc/jit/codegen/cuda/arith.h>
#include <torch/csrc/jit/codegen/cuda/fusion.h>
#include <torch/csrc/jit/codegen/cuda/transform_iter.h>

namespace torch {
namespace jit {
namespace fuser {

bool PredicateCompute::hasPredicates(const TensorIndex* ti) {
  std::vector<Int*> preds;
  for (auto ind : ti->indices())
    if (FusionGuard::getCurFusion()->origin(ind) != nullptr)
      return true;
  return false;
}

std::vector<Int*> PredicateCompute::computePredicates(const TensorIndex* ti) {
  std::vector<Int*> preds;
  if (!hasPredicates(ti))
    return preds;
  const TensorView* tv = ti->view();

  TensorDomain* root = tv->getRootDomain();
  TORCH_CHECK(root->size() == ti->size());
  for (decltype(ti->size()) i{0}; i < ti->size(); i++)

    if (FusionGuard::getCurFusion()->origin(ti->index(i)) != nullptr) {
      Val* pred = lt(ti->index(i), root->axis(i)->size());
      TORCH_CHECK(
          pred->getValType().value() == ValType::Scalar &&
          pred->getDataType().value() == DataType::Int);
      preds.push_back(static_cast<Int*>(pred));
    } else {
      preds.push_back(new Int(1));
    }

  return preds;
}

} // namespace fuser
} // namespace jit
} // namespace torch
