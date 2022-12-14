#include <torch/csrc/jit/codegen/cuda/expr_simplifier.h>
#include <torch/csrc/jit/codegen/cuda/iter_visitor.h>
#include <torch/csrc/jit/codegen/cuda/kernel_ir_dispatch.h>
#include <torch/csrc/jit/codegen/cuda/lower2device.h>
#include <torch/csrc/jit/codegen/cuda/lower_magic_zero.h>

#include <torch/csrc/jit/codegen/cuda/lower_scalar_hoist.h>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

namespace {

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

// Find the outer-most loop nest that contains all the dependencies of `value`.
int64_t findOutermostPosWithSatisfiedDependency(
    Val* value,
    const std::vector<kir::ForLoop*>& loops) {
  // We don't recursively look into tensor indexing to find its dependency.
  // Instead, we always assume tensors to have dependency on all loop variables
  // and prefer to put it at the innermost loop.
  if (value->isOneOf<TensorView, kir::TensorIndex>()) {
    return getInnermostNonTrivialLoop(loops);
  }

  auto def = value->definition();

  // If `value` is not computed from other values, then it must either be a loop
  // variable or something like named scalar or constant
  if (def == nullptr) {
    // Check if `value` is a loop variable
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
      if (loop->index()->sameAs(value)) {
        return i;
      }
    }
    // If no loop found, then `value` would could only depend on constants and
    // trivial loop variables, for example:
    // value = blockIdx.x * 256 + threadIdx.x
    // For this case, return -1, which indicates that the computation of this
    // value should be placed at top-level exprs.
    return -1;
  }

  int64_t pos = -1;

  for (auto v : def->inputs()) {
    pos = std::max(pos, findOutermostPosWithSatisfiedDependency(v, loops));
  }

  return pos;
}

// Get the key for `common_scalar_map_`
kir::ForLoop* getLoopAtPos(
    const std::vector<kir::ForLoop*>& loops,
    int64_t position) {
  // position < 0 refers to the top level exprs (no corresponding loop)
  if (position < 0) {
    return nullptr;
  }
  return loops.at(position);
}

// Check if in the definition of from, there is a subexpression equivalent to
// reference. If found, then return this subexpression.
Val* findRefAsSubexprOf(Val* from, Val* reference, bool exact) {
  if (exact) {
    if (from == reference) {
      return from;
    }
  } else {
    if (from->sameAs(reference)) {
      return from;
    }
  }
  auto def = from->definition();
  if (def != nullptr) {
    for (auto input : def->inputs()) {
      auto common_subexpr = findRefAsSubexprOf(input, reference, exact);
      if (common_subexpr != nullptr) {
        return common_subexpr;
      }
    }
  }
  return nullptr;
}

} // namespace

std::pair<Val*, bool> CommonScalarMap::hoistScalarImpl(
    Val* value,
    const std::vector<kir::ForLoop*>& loops,
    std::vector<Val*>& seen_subexprs,
    int64_t parent_pos,
    bool is_given) {
  if (value->isA<TensorView>() || value->isA<kir::TensorIndex>()) {
    // Current implementation of value hoisting does not have data flow
    // analysis. It assumes that allocating all indices at the beginning of a
    // for loop satisfies all data dependency. However, when the value is an
    // expression of a tensor, this approach will fail. For this case we just
    // return a true for the second return value which will help us to make sure
    // that we don't insert it into `common_scalar_map_` so that we won't
    // consider it as a reusing opportunity.
    return {value, true};
  }

  auto def = value->definition();
  if (def == nullptr || value->isConstScalar()) {
    return {value, false};
  }

  auto my_pos = findOutermostPosWithSatisfiedDependency(value, loops);
  auto my_loop = getLoopAtPos(loops, my_pos);

  // Check if `value` is already computed. If yes, just reuse it and return.
  if (auto existing_subexpr = reuseScalarIfAlreadyComputed(value, my_loop)) {
    return {existing_subexpr, false};
  }
  for (auto existing_subexpr : seen_subexprs) {
    if (value->sameAs(existing_subexpr)) {
      common_scalar_map_[my_loop].emplace_back(existing_subexpr);
      hoisted_or_reused_.emplace(existing_subexpr);
      return {existing_subexpr, false};
    }
  }

  // Recursively hoist all the producers of `value`
  bool changed = false; // if any of the inputs is replaced by an existing val
  bool has_tensor_dependency = false;
  std::vector<Val*> inputs;
  for (auto input : def->inputs()) {
    auto hoist = hoistScalarImpl(input, loops, seen_subexprs, my_pos);
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
    value = IrBuilder::newScalar(*value->getDataType());
    TORCH_INTERNAL_ASSERT(def->outputs().size() == 1);
    auto create_fn = def->newObjectFunc();
    create_fn(value->container(), inputs, {value}, def->attributes());
  }

  // hoist subexpression to outer loop. If `value` depends on a tensor, then we
  // should never insert it into `common_scalar_map_`, because we can not
  // allocate it at the beginning of the loop. If `value` is the given value to
  // the public `hoistScalar`, then we should always insert it into
  // `common_scalar_map_` so that future `value` could consider reusing it. If
  // `value` is a subexpression of the given value, then we insert it into
  // `common_scalar_map_` only if it can be hoisted to outer loops.
  if (!has_tensor_dependency && (is_given || my_pos < parent_pos)) {
    common_scalar_map_[my_loop].emplace_back(value);
    if (my_pos < parent_pos) {
      hoisted_or_reused_.emplace(value);
    }
  }
  seen_subexprs.emplace_back(value);
  return {value, has_tensor_dependency};
}

namespace {

std::list<ValInfo> getLoopIndices(const std::vector<kir::ForLoop*>& loops) {
  std::list<ValInfo> variables;
  for (auto loop : loops) {
    if (loop->isTrivial()) {
      if (loop->iter_domain()->isThread()) {
        variables.push_front({loop->index()});
      }
    } else {
      variables.push_back({loop->index()});
    }
  }
  return variables;
}

} // namespace

Val* CommonScalarMap::hoistScalar(
    Val* value,
    const std::vector<kir::ForLoop*>& loops) {
  if (isOptionDisabled(DisableOption::IndexHoist)) {
    return value;
  }
  value = simplifyExpr(value, getLoopIndices(loops));
  std::vector<Val*> seen_subexprs;
  return hoistScalarImpl(
             value,
             loops,
             seen_subexprs,
             getInnermostNonTrivialLoop(loops),
             true)
      .first;
}

Val* CommonScalarMap::reuseScalarIfAlreadyComputed(
    Val* value,
    kir::ForLoop* loop) {
  // Find if loop already contain `value`.
  auto it = common_scalar_map_.find(loop);
  if (it != common_scalar_map_.end()) {
    auto& indices = it->second;
    for (auto it = indices.begin(); it != indices.end(); it++) {
      auto idx = *it;
      auto common_subexpr = findRefAsSubexprOf(idx, value, false);
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

std::vector<Val*> CommonScalarMap::getHoistedScalars(kir::ForLoop* loop) const {
  // In codegen, parallel type group may not be generated as a for loop, so
  // don't allocate in this loop
  if (loop != nullptr && loop->isGroup()) {
    return {};
  }
  std::vector<Val*> result;
  auto it = common_scalar_map_.find(loop);
  if (it != common_scalar_map_.end()) {
    for (auto v : it->second) {
      if (hoisted_or_reused_.count(v) > 0) {
        result.emplace_back(v);
      }
    }
  }
  return result;
}

void CommonScalarMap::initialize(const std::vector<Expr*> exprs) {
  // We only hoist scalars not depending on tensors. In lowered expressions, all
  // these scalars are computed in top level scope.
  TORCH_INTERNAL_ASSERT(
      common_scalar_map_.empty(),
      "CommonScalarMap used before initialization.");
  for (auto expr : exprs) {
    if (lower_utils::isScalarExpr(expr) && expr->outputs().size() == 1) {
      common_scalar_map_[nullptr].emplace_back(expr->output(0));
    } else if (ir_utils::isTvOp(expr)) {
      // We only try to reuse scalar expressions placed at the beginning of the
      // top level scope. For example, if I have
      // i1 = i0 + 1
      // i2 = T0.size[0] * T0.size[1]
      // i3 = address(T0)
      // i4 = i3 + 1
      // ....
      // Then we only consider `i1 = i0 + 1` and `i2 = T0.size[0] * T0.size[1]`
      // as valid reuse-opportunity. The barrier between valid and invalid is
      // the first tensor operation. The reason why we have this barrier is
      // because the `reorderExprsForComputeAt` always place scalar expressions
      // at the beginning if possible. If a scalar expression is placed after a
      // tensor expression, then this scalar operation must have dependency on
      // some tensor. For this case, we just giveup reusing because this helps
      // us to keep the analysis simple.
      break;
    }
  }
}

namespace {

// Check if the given `value` is already allocated (and computed in full) or
// computed partially. If yes, then return the position of the existing
// computation for the first return value, else return -1 for the first return
// value. The second return value is a boolean indicating if the given `value`
// is already fully computed.
std::pair<int64_t, bool> findAllocPointFromDataDependency(
    const std::vector<Expr*>& exprs, // ordered expressions in the scope where
                                     // the hoisted value should be inserted
    Val* value) {
  int64_t pos = -1;
  for (auto i : c10::irange(exprs.size())) {
    for (auto o : exprs[i]->outputs()) {
      if (value == o) {
        return {i, true};
      }
      auto subexpr = findRefAsSubexprOf(value, o, true);
      if (subexpr != nullptr) {
        pos = i;
      }
    }
  }
  return {pos, false};
}

// Inserts allocations of hoisted indices
class CommonIndexInserter : private kir::ExprMutator {
 public:
  static std::vector<Expr*> run(
      const std::vector<Expr*>& exprs,
      const CommonScalarMap& common_indices) {
    CommonIndexInserter inserter(exprs, common_indices);
    return std::move(inserter.exprs_);
  }

 private:
  CommonIndexInserter(
      const std::vector<Expr*>& exprs,
      const CommonScalarMap& common_scalar_map)
      : common_scalar_map_(common_scalar_map) {
    IrVisitor::handle(exprs);
    maybeInsertAllocation(nullptr);
    mutate();
  }

  void maybeInsertAllocation(kir::ForLoop* loop) {
    kir::Scope* scope = nullptr;
    if (loop != nullptr) {
      scope = &loop->body();
    }
    const auto& exprs = (scope == nullptr ? exprs_ : scope->exprs());
    int64_t alloc_point = -1;
    Expr* insert_ref = nullptr;
    for (auto value : common_scalar_map_.getHoistedScalars(loop)) {
      auto existing_alloc_info = findAllocPointFromDataDependency(exprs, value);
      // If this value has already been fully computed, then don't insert
      // duplicate allocation and computation.
      if (existing_alloc_info.second) {
        continue;
      }
      // If a partial computation is found, then move the allocation point to
      // after the partial computation, so that the result of this partial
      // computation can be used.
      if (existing_alloc_info.first > alloc_point) {
        alloc_point = existing_alloc_info.first;
        insert_ref = exprs[alloc_point];
      }
      // Make the type of the hoisted value be the value type of the
      // kernel, which can be either int64_t or int. Not very clean,
      // but this seems to be the quickest way to use the value type
      // as we don't have a scalar IR node for the value type.
      if (isIntegralType(*value->getDataType())) {
        value->resolveIndexDtype();
      }

      auto alloc = IrBuilder::create<kir::Allocate>(
          value, MemoryType::Local, GpuLower::current()->kernel()->oneVal());
      const auto def = value->definition();
      TORCH_INTERNAL_ASSERT(
          def != nullptr,
          "Hoisted value must have a definition. ",
          value->toString());
      if (insert_ref == nullptr) {
        registerInsertBefore(exprs.at(0), alloc, scope);
      } else {
        registerInsertAfter(insert_ref, alloc, scope);
      }
      registerInsertAfter(alloc, def, scope);
      insert_ref = def;
    }
  }

  using kir::ExprMutator::handle;

  void handle(kir::ForLoop* loop) final {
    maybeInsertAllocation(loop);
    kir::ExprMutator::handle(loop);
  }

 private:
  const CommonScalarMap& common_scalar_map_;
};

} // namespace

std::vector<Expr*> allocateCommonScalars(const std::vector<Expr*>& exprs) {
  return CommonIndexInserter::run(
      exprs, GpuLower::current()->commonScalarMap());
}

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
