#include <torch/csrc/jit/codegen/cuda/lower_predicate_elimination.h>

#include <torch/csrc/jit/codegen/cuda/arith.h>
#include <torch/csrc/jit/codegen/cuda/expr_evaluator.h>
#include <torch/csrc/jit/codegen/cuda/instrumentation.h>
#include <torch/csrc/jit/codegen/cuda/ir_iostream.h>
#include <torch/csrc/jit/codegen/cuda/ir_utils.h>
#include <torch/csrc/jit/codegen/cuda/lower2device.h>
#include <torch/csrc/jit/codegen/cuda/lower_shift.h>
#include <torch/csrc/jit/codegen/cuda/lower_utils.h>
#include <torch/csrc/jit/codegen/cuda/predicate_compute.h>
#include <torch/csrc/jit/codegen/cuda/transform_iter.h>
#include <torch/csrc/jit/codegen/cuda/transform_replay.h>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

namespace {

// Warp primitives are currently limited to un-predicated usage,
//   predicating these ops will require extra steps to ensure that
//   the whole warp will get the same value.
void assertOnWarpOps(const Expr* expr) {
  TORCH_INTERNAL_ASSERT(
      !expr->isA<MmaOp>(),
      "Mma op: cannot eliminate predicate for mma op, tiling not valid. ",
      expr->toString());
}

} // namespace

namespace {

class PredicateAnalyzer : public OptOutDispatch {
 public:
  //! Checks if a predicate is needed to avoid out-of-bound accesses.
  //!
  //! Due to the way we allocate local-memory tensors, there should
  //! never be out-of-bound accesses with consumer tensors when allocated on
  //! local memory. However, accessing producer tensors still may
  //! result in out-of-bound as they are replayed as consumers.
  static bool needsPredicate(TensorView* producer, TensorView* consumer) {
    // Both tensors must be on local memory. Global tensors must be
    // predicated as allocation is done based on root domains. Smem
    // and local tensors are allocated based on leaf domains, however,
    // smem tensors are parallelized, which is highly likely, the size
    // of the parallelized axis is the actual size of the axis, not
    // the number of threads. Since the number of threads can be
    // larger than the axis size, it's not safe to skip predication

    // Check that parallel dimension will not generate out of bound index
    if (!(producer->getMemoryType() == MemoryType::Local &&
          consumer->getMemoryType() == MemoryType::Local)) {
      return true;
    }

    auto pairwise_map = PairwiseRootDomainMap(producer, consumer);
    auto c2p =
        BestEffortReplay::replayPasC(producer, consumer, -1, pairwise_map)
            .getReplay();

    PredicateAnalyzer analyzer(c2p);

    for (auto id : consumer->domain()->domain()) {
      if (analyzer.needsPredicate(id)) {
        return true;
      }
    }

    return false;
  }

 private:
  PredicateAnalyzer(const std::unordered_map<IterDomain*, IterDomain*>& c2p_map)
      : c2p_map_(c2p_map) {}

  // Returns true if no out-of-bound accesses could occur with a
  // producer
  bool needsPredicate(IterDomain* consumer_id) {
    needs_predicate_ = false;
    handle(consumer_id);
    return needs_predicate_;
  }

  void handle(IterDomain* consumer_id) override {
    // The traversal should have ended if needs_predicate_ was true
    TORCH_INTERNAL_ASSERT(!needs_predicate_);

    // If consumer_id is not going to be materialized as a loop (e.g.,
    // broadcast), no need to predicate
    if (consumer_id->isBroadcast() ||
        GpuLower::current()->trivialReductionInfo().isDerived(consumer_id)) {
      return;
    }

    // If the producer has a matching domain, it should not cause
    // out-of-bound accesses
    if (c2p_map_.find(consumer_id) != c2p_map_.end()) {
      return;
    }

    // If no definition exists, stop traversing
    if (consumer_id->definition() == nullptr) {
      return;
    }

    OptOutDispatch::handle(consumer_id->definition());
  }

  // If it splits the input axis evenly, proceeds to check the input
  // axis. Otherwise, we can't skip predication as it might cause
  // out-bound accesses with the producer tensor
  void handle(Split* split) override {
    auto factor = split->factor()->getInt();
    if (!factor.has_value()) {
      needs_predicate_ = true;
      return;
    }

    ExpressionEvaluator ee(split->fusion());
    const auto in_extent = ee.evaluate(split->in()->extent());

    if (!in_extent.has_value() || ((in_extent.value() % factor.value()) != 0)) {
      needs_predicate_ = true;
      return;
    }

    handle(split->in());
  }

  void handle(Merge* merge) override {
    handle(merge->inner());
    if (needs_predicate_) {
      return;
    }
    handle(merge->outer());
  }

 private:
  //! BestEffort map from consumer IDs to producer IDs
  const std::unordered_map<IterDomain*, IterDomain*>& c2p_map_;
  bool needs_predicate_ = false;
};

class PredicateChcker : public IterVisitor {
 public:
  static bool needsPredicate(
      Expr* expr,
      const std::unordered_set<const Expr*>& non_predicated_exprs) {
    if (!ir_utils::isTvOp(expr)) {
      return false;
    }

    PredicateChcker checker(non_predicated_exprs);
    checker.handle(expr);
    return checker.needs_predicate_;
  }

 private:
  PredicateChcker(const std::unordered_set<const Expr*>& non_predicated_exprs)
      : non_predicated_exprs_(non_predicated_exprs) {}

  using IterVisitor::handle;

  void handle(Expr* expr) final {
    needs_predicate_ = predicateIntDiv(expr) ||
        predicateMisalignedVectorize(expr) || predicateShift(expr) ||
        predicateProducerConsumerPair(expr) ||
        predicateNonDivisibleRootDomains(expr) ||
        predicateNonDivisibleSplit(expr);

    if (needs_predicate_) {
      return;
    }

    // Check ExprType-specific conditions
    IterVisitor::handle(expr);
  }

  // All "predicateXYZ" functions return true if an expr needs to be
  // predicated.

  // Always predicate integer division and related ops as we don't
  // know what values are in the out-of-bound region and they may
  // cause exceptions
  bool predicateIntDiv(Expr* expr) const {
    auto dt = expr->outputs()[0]->getDataType().value();
    return (
        (dt == DataType::Int || dt == DataType::Int32) &&
        expr->isA<BinaryOp>() &&
        (expr->as<BinaryOp>()->getBinaryOpType() == BinaryOpType::Div ||
         expr->as<BinaryOp>()->getBinaryOpType() == BinaryOpType::Mod ||
         expr->as<BinaryOp>()->getBinaryOpType() == BinaryOpType::Remainder ||
         expr->as<BinaryOp>()->getBinaryOpType() == BinaryOpType::CeilDiv));
  }

  // Skip if MisalignedVectorize is involved for now. This could be
  // relaxed.
  bool predicateMisalignedVectorize(Expr* expr) const {
    std::vector<const std::vector<Val*>*> inputs_and_outputs = {
        &(expr->inputs()), &(expr->outputs())};
    for (const auto& inputs_or_outputs : inputs_and_outputs) {
      for (auto tv : ir_utils::filterByType<TensorView>(*inputs_or_outputs)) {
        if (std::any_of(
                tv->domain()->domain().begin(),
                tv->domain()->domain().end(),
                [](IterDomain* axis) {
                  return axis->getParallelType() ==
                      ParallelType::MisalignedVectorize;
                })) {
          return true;
        }
      }
    }
    return false;
  }

  // Shift is not supported yet.
  bool predicateShift(Expr* expr) const {
    auto& halo_info = GpuLower::current()->haloInfo();
    auto input_tvs = ir_utils::filterByType<TensorView>(expr->inputs());
    return halo_info.needsShiftPredicate(expr) ||
        std::any_of(input_tvs.begin(), input_tvs.end(), [&](auto input_tv) {
             return input_tv->definition() != nullptr &&
                 halo_info.needsShiftPredicate(input_tv->definition());
           });
  }

  // Predicates the expression if any producer-consumer pair of the
  // expression needs to be predicated
  bool predicateProducerConsumerPair(Expr* expr) const {
    for (auto output : ir_utils::filterByType<TensorView>(expr->outputs())) {
      for (auto input : ir_utils::filterByType<TensorView>(expr->inputs())) {
        if (PredicateAnalyzer::needsPredicate(input, output)) {
          return true;
        }
      }
    }
    return false;
  }

  // An index can exceed the logical extent of the indexed domain if
  // it's split. It can cause a reduction op to reduce the same value
  // multiple times. Even a pointwise op can be a problem if the
  // consumer is an alias of the producer. This check excludes such
  // expressions from predicate elimination.
  //
  // This is not an issue if the index includes a zero domain (as defined in
  // index_compute.cpp), the extent is calculated by multiplying the
  // split output domains, so it never cross the domain boundary.
  // So, if a root domain is split and none of its descendants is a
  // zero domain, the expr needs to be predicated. See
  // FusionPredicateElimination6 for a concrete example.
  //
  // It would be also possible to avoid register aliasing instead of
  // giving up predicate elimination. Since this condition should be
  // rather uncommon, either would be fine as long as correctness is
  // provided.
  bool predicateNonDivisibleRootDomains(Expr* expr) const {
    for (auto output : ir_utils::filterByType<TensorView>(expr->outputs())) {
      const auto all_exprs = DependencyCheck::getAllExprsBetween(
          {output->getMaybeRFactorDomain().begin(),
           output->getMaybeRFactorDomain().end()},
          {output->domain()->domain().begin(),
           output->domain()->domain().end()});
      std::unordered_set<Val*> split_root;
      std::copy_if(
          output->getMaybeRFactorDomain().begin(),
          output->getMaybeRFactorDomain().end(),
          std::inserter(split_root, split_root.end()),
          [&](auto rf_root) {
            if (rf_root->isBroadcast() ||
                GpuLower::current()->trivialReductionInfo().isDerived(
                    rf_root)) {
              return false;
            }
            for (Expr* use : rf_root->uses()) {
              if (std::find(all_exprs.begin(), all_exprs.end(), use) ==
                  all_exprs.end()) {
                continue;
              }
              return use->isA<Split>();
            }
            return false;
          });
      // If no root domain is split, no need to predicate
      if (split_root.empty()) {
        continue;
      }
      TORCH_INTERNAL_ASSERT(
          output->getMemoryType() == MemoryType::Local,
          "Local memory tensor is assumed: ",
          output->toString());
      std::vector<Val*> zero_leaf_ids;
      for (const auto i : c10::irange(output->nDims())) {
        auto leaf_id = output->axis(i);
        if (i < output->getComputeAtPosition() || leaf_id->isThread() ||
            leaf_id->isMma()) {
          zero_leaf_ids.push_back(leaf_id);
        }
      }
      if (zero_leaf_ids.empty()) {
        return true;
      }
      const auto vals =
          DependencyCheck::getAllValsBetween(split_root, zero_leaf_ids);
      if (std::any_of(
              split_root.begin(),
              split_root.end(),
              [&vals](auto split_root_id) {
                return std::find(vals.begin(), vals.end(), split_root_id) ==
                    vals.end();
              })) {
        return true;
      }
    }
    return false;
  }

  // Always predicate if non-divisible split is found. It may be
  // possible to make it less conservative.
  // See FusionPredicateElimination7 for a concrete example.
  bool predicateNonDivisibleSplit(Expr* expr) const {
    const auto& non_divisible_split_info =
        GpuLower::current()->nonDivisibleSplitInfo();
    for (auto output : ir_utils::filterByType<TensorView>(expr->outputs())) {
      if (non_divisible_split_info.splitsToPredicate().find(output) !=
          non_divisible_split_info.splitsToPredicate().end()) {
        return true;
      }
    }
    return false;
  }

  // If this is a reduction, and if we omit the predicate for the
  // input, the input may have a garbabe value, which must not be used
  // for this reduction. However, it is still legal to omit its
  // predicate when: 1) the predicate of the input is not omitted and
  // 2) the input can be initialized to the init value of this
  // reduction. When the input is the output of another reduciton, the
  // input is initialized to the init value of the reduction, so the
  // two reductions must use the same init value.
  // See FusionPredicateElimination3 and FusionPredicateElimination4
  // for concrete examples.
  void handle(ReductionOp* rop) final {
    auto input = rop->inputs()[0]->as<TensorView>();
    auto input_def = input->definition();
    // When input_def is null, input must be an input to the fusion,
    // so that must be allocated on global memory. Since we don't omit
    // predication for expressions involving global memory, this
    // should never occur.
    TORCH_INTERNAL_ASSERT(
        input_def != nullptr, "Inconsistent input found: ", input);

    // The input needs to be initialized to the init value to omit
    // the predicate, so if the input has its own init value, i.e.,
    // produced by another reduction, they must use the same init
    // value.
    Val* input_init = ir_utils::getReductionInitValOf(input);
    if (input_init != nullptr && !rop->init()->sameAs(input_init)) {
      needs_predicate_ = true;
      return;
    }

    // If input is not predicated, out-of-bound value may be
    // overwritten by a garbage value. However, it doesn't matter if
    // the input is also produced by another reduction. If the preceding
    // reduction omits the predicate, it means its input must be
    // initialized to its init value, so no predicate should be
    // needed in both of the two reduction ops if they use the same
    // init value, which is guaranteed by the avove check, and the
    // same reduction op.
    if (auto input_def_rop = dynamic_cast<ReductionOp*>(input_def)) {
      if (rop->getReductionOpType() != input_def_rop->getReductionOpType() &&
          non_predicated_exprs_.find(input_def) !=
              non_predicated_exprs_.end()) {
        needs_predicate_ = true;
        return;
      }
    } else if (
        non_predicated_exprs_.find(input_def) != non_predicated_exprs_.end()) {
      needs_predicate_ = true;
      return;
    }
  }

  // Welford. See FusionPredicateElimination5.
  void handle(WelfordOp* wop) final {
    for (const auto i : c10::irange(3)) {
      auto init = wop->getInitVals()[i];

      // Welford input can be a scalar. Predicate is required unless
      // the scalar value is equal to the init value.
      auto input = wop->inputs().at(i);
      if (input->isScalar()) {
        if (!input->sameAs(init)) {
          needs_predicate_ = true;
          return;
        }
        continue;
      }

      auto input_tv = dynamic_cast<TensorView*>(input);
      TORCH_INTERNAL_ASSERT(input_tv != nullptr);

      auto input_def = input->definition();

      // When input_def is null, input must be an input to the fusion,
      // so that must be allocated on global memory. Since we don't omit
      // predication for expressions involving global memory, this
      // should never occur.
      TORCH_INTERNAL_ASSERT(
          input_def != nullptr, "Inconsistent input found: ", input);

      // The input needs to be initialized to the init value to omit
      // the predicate, so if the input has its own init value, i.e.,
      // produced by another reduction, they must use the same init
      // value.
      Val* input_init = ir_utils::getReductionInitValOf(input_tv);
      if (input_init != nullptr && !init->sameAs(input_init)) {
        needs_predicate_ = true;
        return;
      }

      // If input is not predicated, out-of-bound value may be
      // overwritten by a garbage value. However, it doesn't matter if
      // the input is also produced by another welford.
      if (!input_def->isA<WelfordOp>() &&
          non_predicated_exprs_.find(input_def) !=
              non_predicated_exprs_.end()) {
        needs_predicate_ = true;
      }
    }
  }

  // Similar to the above reduction constraint but for MMA
  void handle(MmaOp* mma) final {
    for (auto input : ir_utils::filterByType<TensorView>(mma->inputs())) {
      auto input_def = input->definition();
      TORCH_INTERNAL_ASSERT(
          input_def != nullptr, "Inconsistent input found: ", input);

      Val* input_init = ir_utils::getReductionInitValOf(input);
      if (input_init != nullptr && !mma->init()->sameAs(input_init)) {
        needs_predicate_ = true;
        return;
      }

      if (non_predicated_exprs_.find(input_def) !=
          non_predicated_exprs_.end()) {
        needs_predicate_ = true;
        return;
      }
    }
  }

 private:
  const std::unordered_set<const Expr*>& non_predicated_exprs_;
  bool needs_predicate_ = false;
};

} // namespace

bool PredicateElimination::needsPredicate(Expr* expr) const {
  return PredicateChcker::needsPredicate(expr, non_predicated_exprs_);
}

void PredicateElimination::handle(Expr* expr) {
  if (!ir_utils::isTvOp(expr)) {
    return;
  }

  if (needsPredicate(expr)) {
    assertOnWarpOps(expr);
    return;
  }

  non_predicated_exprs_.insert(expr);

  // Ensure all inputs have some values set at the out-of-bound
  // regions
  for (const auto i : c10::irange(expr->inputs().size())) {
    auto input = dynamic_cast<TensorView*>(expr->inputs()[i]);
    if (input == nullptr) {
      continue;
    }
    auto input_def = input->definition();
    // When input_def is null, input must be an input to the fusion,
    // so that must be allocated on global memory. Since we don't omit
    // predication for expressions involving global memory, this
    // should never occur.
    std::stringstream ss;
    ss << input;
    TORCH_INTERNAL_ASSERT(
        input_def != nullptr, "Inconsistent input found: ", ss.str());

    // If input is an output of reduction, it should be fully
    // initialied as it's allocated on local memory.
    if (input_def->isA<ReductionOp>() || input_def->isA<WelfordOp>()) {
      continue;
    } else if (expr->isA<ReductionOp>()) {
      setReductionInitValue(input, expr->as<ReductionOp>()->init());
      continue;
    } else if (auto wop = dynamic_cast<WelfordOp*>(expr)) {
      Val* init = wop->getInitVals().at(i);
      setReductionInitValue(input, init);
      continue;
    }
    if (expr->isA<MmaOp>()) {
      setReductionInitValue(input, expr->as<MmaOp>()->init());
      continue;
    } else if (
        non_predicated_exprs_.find(input_def) != non_predicated_exprs_.end()) {
      // If an input does not need a predicate either, then it should
      // have some value, so no need to set a default value
      continue;
    } else {
      // Make sure input is initialized
      setDefaultInitValue(input);
    }
  }
}

bool PredicateElimination::setDefaultInitValue(TensorView* tv) {
  auto it = init_value_map_.find(tv);
  // If there's already a mapping for tv, it should be mapped to a
  // zero val or a reduction init. Either case, no need to modify
  // the existing mapping.
  if (it == init_value_map_.end()) {
    init_value_map_.insert({tv, nullptr});
  }
  return true;
}

bool PredicateElimination::setReductionInitValue(
    TensorView* tv,
    Val* reduction_init) {
  TORCH_INTERNAL_ASSERT(tv != nullptr);

  auto it = init_value_map_.find(tv);
  if (it == init_value_map_.end()) {
    init_value_map_.insert({tv, reduction_init});
    return true;
  }

  auto existing_val = it->second;
  if (existing_val == nullptr) {
    // If the existing mapping returns nullptr, it means that a
    // default init was set before. Overwrite with the reduction
    // init val.
    init_value_map_[tv] = reduction_init;
    return true;
  } else if (existing_val->sameAs(reduction_init)) {
    return true;
  } else {
    TORCH_INTERNAL_ASSERT(
        false,
        "Incosistent setting of initialization value for t",
        tv->name(),
        ". Prev: ",
        existing_val,
        ", New: ",
        reduction_init);
    return false;
  }
}

bool PredicateElimination::canOmitPredicate(const Expr* expr) const {
  // Predicate elimination can be disabled with
  // PYTORCH_NVFUSER_DISABLE_PREDICATE_ELIMINATION
  if (disablePredicateElimination()) {
    assertOnWarpOps(expr);
    return false;
  }

  TORCH_INTERNAL_ASSERT(expr != nullptr);
  const auto out_tv = ir_utils::getTvOutput(expr);
  TORCH_INTERNAL_ASSERT(out_tv != nullptr, "Not a tensor expression");
  // No need to predicate local tensors to which a scalar is assigned
  if (out_tv->getMemoryType() == MemoryType::Local) {
    if (auto uop = dynamic_cast<const UnaryOp*>(expr)) {
      if (uop->getUnaryOpType() == UnaryOpType::Set && uop->in()->isScalar()) {
        return true;
      }
    }
  }
  if (non_predicated_exprs_.find(expr) != non_predicated_exprs_.end()) {
    return true;
  }

  assertOnWarpOps(expr);
  return false;
}

Val* PredicateElimination::getInitValue(TensorView* tv) const {
  auto it = init_value_map_.find(tv);
  if (it == init_value_map_.end()) {
    return nullptr;
  }
  auto init_val = it->second;
  if (init_val == nullptr) {
    // No reduction restriction. Just use zero
    return GpuLower::current()->kernel()->zeroVal();
  } else {
    return init_val;
  }
}

void PredicateElimination::build(Fusion* fusion) {
  traverseFrom(fusion, fusion->outputs());
}

std::string PredicateElimination::toString() const {
  std::stringstream ss;
  ss << "Tensors that do not need predication:";
  for (auto expr : non_predicated_exprs_) {
    for (auto out : expr->outputs()) {
      TORCH_INTERNAL_ASSERT(out->isA<TensorView>());
      ss << " T" << out->name();
    }
  }
  ss << "\n";
  ss << "Init values:";
  for (auto kv : init_value_map_) {
    ss << " T" << kv.first->name() << "->";
    if (kv.second == nullptr) {
      ss << "<default(0)>";
    } else {
      ss << kv.second;
    }
  }
  ss << "\n";
  return ss.str();
}

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
