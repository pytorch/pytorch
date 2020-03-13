#include <torch/csrc/jit/codegen/cuda/predicate_compute.h>

#include <torch/csrc/jit/codegen/cuda/arith.h>
#include <torch/csrc/jit/codegen/cuda/fusion.h>
#include <torch/csrc/jit/codegen/cuda/transform_iter.h>

namespace torch {
namespace jit {
namespace fuser {

bool PredicateCompute::hasPredicates(
    const TensorView* tv,
    const std::vector<Int*> _indices) {
  std::vector<Int*> preds;
  for (auto ind : _indices)
    if (FusionGuard::getCurFusion()->origin(ind) != nullptr)
      return true;
  return false;
}

std::vector<Int*> PredicateCompute::computePredicates(
    const TensorView* tv,
    const std::vector<Int*> _indices) {
  std::vector<Int*> preds;
  if (!hasPredicates(tv, _indices))
    return preds;

  TensorDomain* root = tv->getRootDomain();
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

} // namespace fuser
} // namespace jit
} // namespace torch