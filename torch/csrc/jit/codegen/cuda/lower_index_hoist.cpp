#include <torch/csrc/jit/codegen/cuda/iter_visitor.h>
#include <torch/csrc/jit/codegen/cuda/kernel_ir_dispatch.h>
#include <torch/csrc/jit/codegen/cuda/lower2device.h>
#include <torch/csrc/jit/codegen/cuda/lower_magic_zero.h>

#include <torch/csrc/jit/codegen/cuda/lower_index_hoist.h>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

namespace {

// Find the outer-most loop nest that contains all the dependencies of `index`.
int64_t findOutermostPosWithSatisfiedDependency(
    Val* index,
    const std::vector<kir::ForLoop*>& loops) {
  auto def = index->definition();

  // If `index` is not computed from other values, then it must either be a loop
  // variable or something like named scalar or constant
  if (def == nullptr) {
    // Check if `index` is a loop variable
    for (auto i : c10::irange(loops.size())) {
      auto loop = loops.at(i);
      // We skip trivial loop here because it is never materialized, so its loop
      // variable is accessible everywhere. For example, if the trivial loop is
      // a un-concretized broadcast, the its loop variable is a constant 0. If
      // the trivial loop is thread parallelized, then its loop variable will be
      // threadIdx.x, which is also accessible on all scopes.
      if (loop->isTrivial()) {
        continue;
      }
      if (loop->index()->sameAs(index)) {
        return i;
      }
    }
    // If no loop found, then `index` would could only depend on constants and
    // trivial loop variables, for example:
    // index = blockIdx.x * 256 + threadIdx.x
    // For this case, return -1, which indicates that the computation of this
    // index should be placed at top-level exprs.
    return -1;
  }

  int64_t pos = -1;

  for (auto v : def->inputs()) {
    pos = std::max(pos, findOutermostPosWithSatisfiedDependency(v, loops));
  }

  return pos;
}

// Get the key for `common_index_map_`
kir::ForLoop* getLoopAtPos(
    const std::vector<kir::ForLoop*>& loops,
    int64_t position) {
  // position < 0 refers to the top level exprs (no corresponding loop)
  if (position < 0) {
    return nullptr;
  }
  return loops.at(position);
}

// Get the position of the innermost non-trivial loop
int64_t getInnermostNonTrivialLoop(const std::vector<kir::ForLoop*>& loops) {
  int64_t position = -1;
  for (auto i : c10::irange(loops.size())) {
    if (!loops.at(i)->isTrivial()) {
      position = i;
    }
  }
  return position;
}

// Check if in the definition of from, there is a subexpression equivalent to
// reference. If found, then return this subexpression.
Val* findRefAsSubexprOf(Val* from, Val* reference) {
  if (from->sameAs(reference)) {
    return from;
  }
  auto def = from->definition();
  if (def != nullptr) {
    for (auto input : def->inputs()) {
      auto common_subexpr = findRefAsSubexprOf(input, reference);
      if (common_subexpr != nullptr) {
        return common_subexpr;
      }
    }
  }
  return nullptr;
}

} // namespace

std::pair<Val*, bool> CommonIndexMap::hoistIndexImpl(
    Val* index,
    const std::vector<kir::ForLoop*>& loops,
    int64_t parent_pos,
    bool is_given) {
  if (index->isA<TensorView>() || index->isA<kir::TensorIndex>()) {
    // Current implementation of index hoisting does not have data flow
    // analysis. It assumes that allocating all indices at the beginning of a
    // for loop satisfies all data dependency. However, when the index is an
    // expression of a tensor, this approach will fail. For this case we just
    // return a true for the second return value which will help us to make sure
    // that we don't insert it into `common_index_map_` so that we won't
    // consider it as a reusing opportunity.
    return {index, true};
  }

  auto def = index->definition();
  if (def == nullptr || index->isConstScalar()) {
    return {index, false};
  }

  auto my_pos = findOutermostPosWithSatisfiedDependency(index, loops);
  auto my_loop = getLoopAtPos(loops, my_pos);

  // Check if `index` is already computed. If yes, just reuse it and return.
  auto existing_subexpr = reuseIndexIfAlreadyComputed(index, my_loop);
  if (existing_subexpr != nullptr) {
    return {existing_subexpr, false};
  }

  // Recursively hoist all the producers of `index`
  bool changed = false; // if any of the inputs is replaced by an existing val
  bool has_tensor_dependency = false;
  std::vector<Val*> inputs;
  for (auto input : def->inputs()) {
    auto hoist = hoistIndexImpl(input, loops, my_pos);
    inputs.emplace_back(hoist.first);
    if (hoist.second) {
      has_tensor_dependency = true;
    }
    if (inputs.back() != input) {
      changed = true;
    }
  }

  // If any of the inputs is replaced, then create a new expression whose inputs
  // are replaced with hoisted input
  if (changed) {
    index = IrBuilder::newScalar(*index->getDataType());
    TORCH_INTERNAL_ASSERT(def->outputs().size() == 1);
    auto create_fn = def->newObjectFunc();
    create_fn(index->container(), inputs, {index}, def->attributes());
  }

  // hoist subexpression to outer loop. If `index` depends on a tensor, then we
  // should never insert it into `common_index_map_`, because we can not
  // allocate it at the beginning of the loop. If `index` is the given index to
  // the public `hoistIndex`, then we should always insert it into
  // `common_index_map_` so that future `index` could consider reusing it. If
  // `index` is a subexpression of the given index, then we insert it into
  // `common_index_map_` only if it can be hoisted to outer loops.
  if (!has_tensor_dependency && (is_given || my_pos < parent_pos)) {
    common_index_map_[my_loop].emplace_back(index);
    if (my_pos < parent_pos) {
      hoisted_or_reused_.emplace(index);
    }
  }
  return {index, has_tensor_dependency};
}

Val* CommonIndexMap::hoistIndex(
    Val* index,
    const std::vector<kir::ForLoop*>& loops) {
  if (isOptionDisabled(DisableOption::IndexHoist)) {
    return index;
  }
  return hoistIndexImpl(index, loops, getInnermostNonTrivialLoop(loops), true)
      .first;
}

Val* CommonIndexMap::reuseIndexIfAlreadyComputed(
    Val* index,
    kir::ForLoop* loop) {
  // Find if loop already contain `index`.
  auto it = common_index_map_.find(loop);
  if (it != common_index_map_.end()) {
    auto& indices = it->second;
    for (auto it = indices.begin(); it != indices.end(); it++) {
      auto idx = *it;
      auto common_subexpr = findRefAsSubexprOf(idx, index);
      if (common_subexpr != nullptr) {
        if (common_subexpr != idx) {
          // If the reuse is a subexpression instead of the complete
          // expression, we split this subexpression out and allocate it
          // separately.
          indices.insert(it, common_subexpr);
        }
        hoisted_or_reused_.emplace(common_subexpr);
        return common_subexpr;
      }
    }
  }
  return nullptr;
}

std::vector<Val*> CommonIndexMap::getHoistedIndices(kir::ForLoop* loop) const {
  // In codegen, parallel type group may not be generated as a for loop, so
  // don't allocate in this loop
  if (loop != nullptr && loop->isGroup()) {
    return {};
  }
  std::vector<Val*> result;
  auto it = common_index_map_.find(loop);
  if (it != common_index_map_.end()) {
    for (auto v : it->second) {
      if (hoisted_or_reused_.count(v) > 0) {
        result.emplace_back(v);
      }
    }
  }
  return result;
}

namespace {

// Inserts allocations of hoisted indices
class CommonIndexInserter : private kir::ExprMutator {
 public:
  static std::vector<Expr*> run(
      const std::vector<Expr*>& exprs,
      const CommonIndexMap& common_indices) {
    CommonIndexInserter inserter(exprs, common_indices);
    return std::move(inserter.exprs_);
  }

 private:
  CommonIndexInserter(
      const std::vector<Expr*>& exprs,
      const CommonIndexMap& common_index_map)
      : common_index_map_(common_index_map) {
    IrVisitor::handle(exprs);
    maybeInsertAllocation(nullptr);
    mutate();
  }

  void maybeInsertAllocation(kir::ForLoop* loop) {
    kir::Scope* scope = nullptr;
    if (loop != nullptr) {
      scope = &loop->body();
    }
    for (auto index : common_index_map_.getHoistedIndices(loop)) {
      // Make the type of the hoisted index be the index type of the
      // kernel, which can be either int64_t or int. Not very clean,
      // but this seems to be the quickest way to use the index type
      // as we don't have a scalar IR node for the index type.
      if (isIntegralType(*index->getDataType())) {
        index->resolveIndexDtype();
      }

      auto alloc = IrBuilder::create<kir::Allocate>(
          index, MemoryType::Local, GpuLower::current()->kernel()->oneVal());
      const auto index_def = index->definition();
      TORCH_INTERNAL_ASSERT(
          index_def != nullptr,
          "Hoisted index must have a definition. ",
          index->toString());

      auto ref = (scope == nullptr ? exprs_.at(0) : scope->exprs().at(0));
      registerInsertBefore(ref, alloc, scope);
      registerInsertBefore(ref, index_def, scope);
    }
  }

  using kir::ExprMutator::handle;

  void handle(kir::ForLoop* loop) final {
    maybeInsertAllocation(loop);
    kir::ExprMutator::handle(loop);
  }

 private:
  const CommonIndexMap& common_index_map_;
};

} // namespace

std::vector<Expr*> allocateCommonIndices(const std::vector<Expr*>& exprs) {
  return CommonIndexInserter::run(exprs, GpuLower::current()->commonIndexMap());
}

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
