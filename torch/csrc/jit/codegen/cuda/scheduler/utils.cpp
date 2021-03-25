#include <torch/csrc/jit/codegen/cuda/scheduler/utils.h>

#include <torch/csrc/jit/codegen/cuda/arith.h>
#include <torch/csrc/jit/codegen/cuda/compute_at_map.h>
#include <torch/csrc/jit/codegen/cuda/ir_iostream.h>
#include <torch/csrc/jit/codegen/cuda/ir_utils.h>
#include <torch/csrc/jit/codegen/cuda/iter_visitor.h>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {
namespace scheduler_utils {
std::vector<int> reductionAxes(TensorView* tv) {
  size_t n_dims = tv->nDims();
  std::vector<int> reduction_axes;
  for (size_t i = 0; i < n_dims; i++) {
    if (tv->axis(i)->isReduction()) {
      reduction_axes.emplace_back(i);
    }
  }
  return reduction_axes;
}

// Merge all reduction to the right side and returns total number of
// reduction axes
size_t mergeReduction(TensorView* tv) {
  int prev_i = -1;
  size_t num_merged = 0;
  for (int i = static_cast<int>(tv->nDims()) - 1; i >= 0; i--) {
    if (!tv->axis(i)->isReduction()) {
      continue;
    }
    if (prev_i == -1) {
      prev_i = i;
    } else {
      tv->merge(i, prev_i);
      prev_i = i;
      num_merged++;
    }
  }
  if (prev_i == 0) {
    tv->reorder({{prev_i, -1}});
  }

  return prev_i == -1 ? 0 : num_merged + 1;
}

// merge all non-reduction axes to the left side and returns total number of
// iteration axes
size_t mergeNonReduction(TensorView* tv) {
  int prev_i = -1;
  size_t num_merged = 0;
  for (int i = static_cast<int>(tv->nDims()) - 1; i >= 0; i--) {
    if (tv->axis(i)->isReduction()) {
      continue;
    }
    if (prev_i == -1) {
      prev_i = i;
    } else {
      tv->merge(i, prev_i);
      prev_i = i;
      num_merged++;
    }
  }
  if (prev_i != 0) {
    tv->reorder({{prev_i, 0}});
  }

  return prev_i == -1 ? 0 : num_merged + 1;
}

int log2_ceil(int value) {
  int log2_value = 0;
  while ((1 << log2_value) < value) {
    ++log2_value;
  }
  return log2_value;
}

void scheduleReductionComputeAt(
    TensorView* red_tv,
    TensorView* red_tv_rf,
    const std::vector<TensorView*>& outs_of_red) {
  if (!outs_of_red.empty()) {
    red_tv->computeAt(outs_of_red[0], -1);
  }
  if (red_tv_rf != nullptr) {
    red_tv_rf->computeAt(red_tv, -1);
  }
}

TensorView* rfactorHelper(TensorView* red_tv, const std::vector<int>& axes) {
  TORCH_INTERNAL_ASSERT(red_tv->definition() != nullptr);
  const bool is_welford = red_tv->definition()->isA<WelfordOp>();
  if (!is_welford) {
    return red_tv->rFactor(axes);
  }
  auto welford = red_tv->definition()->as<WelfordOp>();
  auto w_var = welford->outVar()->as<TensorView>();
  auto w_avg = welford->outAvg()->as<TensorView>();
  auto w_n = welford->outN()->as<TensorView>();

  WelfordResult rtvs = red_tv->rFactor(axes, w_var, w_avg, w_n);

  // TODO: this can be more generic, using avg because
  //      WelfordOp::out() returns the avg
  return rtvs.avg;
}

bool canDuplicate(const Expr* expr) {
  return expr->outputs().size() == 1 && expr->output(0)->isA<TensorView>() &&
      (expr->getExprType().value() == ExprType::BinaryOp ||
       expr->getExprType().value() == ExprType::UnaryOp ||
       expr->getExprType().value() == ExprType::TernaryOp ||
       expr->getExprType().value() == ExprType::BroadcastOp);
}

bool isConstantAllocation(const TensorView* tv) {
  if (!tv->hasComputeAt()) {
    // We cannot determine allocation size without computeAt structure.
    // Assume Non-Constant Allocation
    return false;
  }

  bool constant_allocation = true;
  auto domain = tv->domain()->domain();
  for (size_t axis = tv->getComputeAtPosition(); axis < domain.size(); ++axis) {
    if (!domain[axis]->isBroadcast() && !domain[axis]->isReduction() &&
        !domain[axis]->isParallelized()) {
      constant_allocation &= domain[axis]->extent()->isConstScalar();
    }
  }
  return constant_allocation;
}

//! Find all TensorViews that require duplication to avoid recompute
//! computeAt error when applying inline ComputeAt
std::vector<TensorView*> findTensorViewsToDuplicate(
    Fusion* fusion,
    const std::vector<TensorView*>& other_tv) {
  std::vector<TensorView*> duplicate_tv;
  // Initialize stack with any pointwise op with multiple usages
  // Find any pointwise definition expressions via depth-first search (DFS)
  std::vector<TensorView*> stack;
  for (auto tensor : other_tv) {
    if (tensor->uses().size() > 1 && !fusion->hasOutput(tensor)) {
      stack.push_back(tensor);
    }
  }

  std::unordered_set<StmtNameType> visited;
  while (!stack.empty()) {
    auto tensor = stack.back();
    stack.pop_back();

    if (visited.find(tensor->name()) == visited.end()) {
      auto def_expr = tensor->definition();
      if (canDuplicate(def_expr)) {
        duplicate_tv.push_back(tensor);

        for (auto input_tv :
             ir_utils::filterByType<TensorView>(def_expr->inputs())) {
          if (!input_tv->isFusionInput() && !input_tv->isFusionOutput() &&
              !isConstantAllocation(input_tv)) {
            stack.push_back(input_tv);
          }
        }
      }
    }
    visited.insert(tensor->name());
  }

  // sort TensorViews in descending order
  std::sort(
      duplicate_tv.begin(),
      duplicate_tv.end(),
      [](TensorView* left, TensorView* right) {
        return left->name() > right->name();
      });
  return duplicate_tv;
}

bool canComputeAtInline(TensorView* tv) {
  auto uses = tv->uses();
  if (uses.size() == 1) {
    Expr* expr = *uses.begin();
    TensorView* consumer = expr->output(0)->as<TensorView>();
    bool optional_inline =
        !tv->hasBroadcast() && tv->nDims() == consumer->nDims();
    bool required_inline = !isConstantAllocation(tv);
    return optional_inline || required_inline;
  }
  return false;
}

//! Find all TensorViews that require inline ComputeAt
//! to avoid non-static allocation error
std::vector<TensorView*> findTensorViewsToComputeAtInline(
    Fusion* fusion,
    const std::vector<TensorView*>& tensors) {
  std::vector<TensorView*> computeAt_inline_tv;
  for (auto tv : tensors) {
    if (!fusion->hasInput(tv) && !fusion->hasOutput(tv)) {
      if (tv->getMemoryType() == MemoryType::Local && canComputeAtInline(tv)) {
        computeAt_inline_tv.push_back(tv);
      }
    }
  }
  return computeAt_inline_tv;
}

//! Place all cache TensorViews in Shared Memory
//! All point-wise TensorViews inherit shared memory from their parents
void setupSharedMemory(
    Fusion* fusion,
    const std::vector<TensorView*>& cache_tv) {
  std::vector<TensorView*> stack(cache_tv.begin(), cache_tv.end());
  while (!stack.empty()) {
    auto tensor = stack.back();
    stack.pop_back();
    if (!fusion->hasOutput(tensor) && !fusion->hasInput(tensor)) {
      tensor->setMemoryType(MemoryType::Shared);
      for (auto expr : tensor->uses()) {
        if (canDuplicate(expr)) {
          auto output = expr->output(0)->as<TensorView>();
          stack.push_back(output);
        }
      }
    }
  }
}

// TODO: Review this. Seems we should be using a root map here, or we should
// simply be replaying all tensors as a reduction tv.
void organizeAxes(
    const std::vector<TensorView*>& reduction_tv,
    const std::vector<TensorView*>& all_tv) {
  // Determine merged reduction axis position
  auto findMergedReductionAxis = [](TensorView* reduction_tv) {
    int merged_reduction_axis = -1;
    auto domain = reduction_tv->domain()->domain();
    for (size_t axis = 0; axis < domain.size(); ++axis) {
      if (domain[axis]->isReduction()) {
        TORCH_INTERNAL_ASSERT(merged_reduction_axis == -1);
        merged_reduction_axis = axis;
      }
    }
    return merged_reduction_axis;
  };

  auto first_reduction_tv = reduction_tv.front();
  const size_t kRootNumberOfDims = first_reduction_tv->getRootDomain().size();
  auto root_domain = first_reduction_tv->getRootDomain();
  int merged_reduction_axis = -1;

  // Find reduction axes positions
  std::vector<int> reduction_axes;
  for (size_t axis = 0; axis < root_domain.size(); ++axis) {
    if (root_domain[axis]->isReduction()) {
      reduction_axes.push_back(axis);
    }
  }

  // Coalese reduction axes together
  for (auto tv : all_tv) {
    const size_t kOuterAxis = reduction_axes.front();
    if (tv->getRootDomain().size() == kRootNumberOfDims) {
      for (size_t idx = 0; idx < reduction_axes.size() - 1; ++idx) {
        size_t inner_axis = reduction_axes[idx + 1] - idx;
        tv->merge(kOuterAxis, inner_axis);
      }
    }
  }

  // Coalese non-reduction axes together divided by merged reduction axis
  // Flatten input into [Outer, Reduction, Inner]
  merged_reduction_axis = findMergedReductionAxis(first_reduction_tv);
  const int kBeforeReductionAxis = merged_reduction_axis - 1;
  const int kAfterReductionAxis = merged_reduction_axis + 1;
  const size_t kNumberOfDims = first_reduction_tv->nDims();
  for (auto tv : all_tv) {
    if (tv->getRootDomain().size() == kRootNumberOfDims) {
      for (int idx = 0; idx < kBeforeReductionAxis; ++idx) {
        tv->merge(0, 1);
      }
      for (size_t idx = kAfterReductionAxis; idx < kNumberOfDims - 1; ++idx) {
        tv->merge(kAfterReductionAxis, kAfterReductionAxis + 1);
      }
    }
  }

  // Move reduction axes to the inner-most position
  merged_reduction_axis = findMergedReductionAxis(first_reduction_tv);
  const size_t kInnerMostAxis = first_reduction_tv->domain()->nDims() - 1;
  if (merged_reduction_axis != int(kInnerMostAxis)) {
    for (auto tv : all_tv) {
      tv->reorder(
          {{merged_reduction_axis, kInnerMostAxis},
           {kInnerMostAxis, merged_reduction_axis}});
    }
  }
}

// If tv is broadcasted (used in a broadcast op) return that op, otherwise
// return nullptr
Expr* isBroadcasted(TensorView* tv) {
  auto uses = tv->uses();
  if (uses.size() == 1) {
    auto expr = *uses.begin();
    bool is_broadcasted = expr->getExprType().value() == ExprType::BroadcastOp;
    return (is_broadcasted) ? expr : nullptr;
  }
  return nullptr;
};

// If tv is casted (used in a cast op) return that op, otherwise return nullptr
Expr* isCasted(TensorView* tv) {
  auto uses = tv->uses();
  if (uses.size() == 1) {
    auto expr = *uses.begin();
    bool is_casted = expr->getExprType().value() == ExprType::UnaryOp &&
        expr->as<UnaryOp>()->getUnaryOpType() == UnaryOpType::Cast;
    return (is_casted) ? expr : nullptr;
  }
  return nullptr;
};

void handleCastBroadcastInput(Fusion* fusion, TensorView* input) {
  TORCH_INTERNAL_ASSERT(fusion->hasInput(input));

  auto castOp_expr = isCasted(input);
  if (castOp_expr != nullptr) {
    auto castOp_tv = castOp_expr->output(0)->as<TensorView>();
    auto broadcast_expr = isBroadcasted(castOp_tv);
    if (broadcast_expr != nullptr) {
      auto broadcast_tv = broadcast_expr->output(0)->as<TensorView>();
      castOp_tv->computeAt(broadcast_tv, -1);
    }
  }
}

void cacheInputs(
    Fusion* fusion,
    const ReductionParams& rparams,
    const std::vector<TensorView*>& reduction_tv,
    std::vector<TensorView*>& other_tv) {
  if (rparams.fastest_dim) {
    const bool kHasOuterAxis = reduction_tv.front()->nDims() > 1;
    if (rparams.persistent_kernel && kHasOuterAxis) {
      // Fusion input castOp replaces cache_after
      // Determine if there are any casts or broadcast on fusion
      // inputs
      const auto& in_tv = ir_utils::filterByType<TensorView>(fusion->inputs());
      for (const auto input : in_tv) {
        if (input->getRootDomain().size() > 1) {
          // If pseudo-cache, skip cache after
          bool hasBroadcast = isBroadcasted(input) != nullptr;
          bool hasCast = isCasted(input) != nullptr;
          if (!hasBroadcast && !hasCast) {
            other_tv.push_back(input->cache_after());
          }
        }
      }
    }
  }
}

namespace {

std::vector<TensorView*> uniqueEntries(
    const std::vector<TensorView*>& tv_deuqe) {
  std::vector<TensorView*> unique_entries;
  std::unordered_set<TensorView*> inserted;
  for (auto tv_entry : tv_deuqe) {
    if (inserted.emplace(tv_entry).second) {
      unique_entries.emplace_back(tv_entry);
    }
  }
  return unique_entries;
}

} // namespace

std::vector<TensorView*> producerTvsOf(TensorView* tv) {
  auto producer_vals =
      ir_utils::filterByType<TensorView>(tv->definition()->inputs());
  return uniqueEntries({producer_vals.begin(), producer_vals.end()});
}

std::vector<TensorView*> consumerTvsOf(TensorView* tv) {
  std::vector<TensorView*> consumer_tvs;
  for (auto use_expr : tv->uses()) {
    auto outputs = ir_utils::filterByType<TensorView>(use_expr->outputs());
    consumer_tvs.insert(consumer_tvs.end(), outputs.begin(), outputs.end());
  }
  return uniqueEntries(consumer_tvs);
}

std::vector<TensorView*> producerTvsOf(const std::vector<TensorView*>& tvs) {
  std::vector<TensorView*> all_producer_tvs;
  for (auto tv : tvs) {
    auto producer_tvs = producerTvsOf(tv);
    all_producer_tvs.insert(
        all_producer_tvs.end(), producer_tvs.begin(), producer_tvs.end());
  }

  return uniqueEntries(all_producer_tvs);
}

std::vector<TensorView*> consumerTvsOf(const std::vector<TensorView*>& tvs) {
  std::vector<TensorView*> all_consumer_tvs;
  for (auto tv : tvs) {
    auto consumer_tvs = consumerTvsOf(tv);
    all_consumer_tvs.insert(
        all_consumer_tvs.end(), consumer_tvs.begin(), consumer_tvs.end());
  }

  return uniqueEntries(all_consumer_tvs);
}

void parallelizeAllLike(
    TensorView* reference_tv,
    const std::vector<TensorView*>& all_tvs) {
  FusionGuard fg(reference_tv->fusion());

  auto ca_loop_map = ComputeAtMap(ComputeAtMap::MappingMode::LOOP);
  ca_loop_map.build(FusionGuard::getCurFusion());
  for (auto id : reference_tv->domain()->domain()) {
    ca_loop_map.getConcreteMappedID(id)->parallelize(id->getParallelType());
  }

  for (auto tv : all_tvs) {
    if (tv->isFusionInput()) {
      continue;
    }
    for (size_t i = 0; i < tv->domain()->domain().size(); i++) {
      tv->axis(i)->parallelize(
          ca_loop_map.getConcreteMappedID(tv->axis(i))->getParallelType());
    }
  }
}

void computeAtInputs(TensorView* consumer, int pos, ComputeAtMode mode) {
  auto inp_vals = IterVisitor::getInputsTo({consumer});
  auto inp_tvs = ir_utils::filterByType<TensorView>(inp_vals);
  for (auto inp_tv : inp_tvs) {
    inp_tv->computeAt(consumer, pos, mode);
  }
}

void computeWithOutputs(TensorView* producer, int pos, ComputeAtMode mode) {
  auto out_vals = DependencyCheck::getAllOutputsOf({producer});
  auto out_tvs = ir_utils::filterByType<TensorView>(out_vals);
  for (auto out_tv : out_tvs) {
    producer->computeWith(out_tv, pos, mode);
  }
}

std::vector<TensorView*> allTvs(Fusion* fusion) {
  auto used_vals = DependencyCheck::getAllValsBetween(
      {fusion->inputs().begin(), fusion->inputs().end()}, fusion->outputs());

  auto used_tvs = ir_utils::filterByType<TensorView>(used_vals);
  return uniqueEntries({used_tvs.begin(), used_tvs.end()});
}

} // namespace scheduler_utils
} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
