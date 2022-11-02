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
      !ir_utils::isLdMatrixOp(expr),
      "Predicate elimination: cannot eliminate pred for ldmatrix, use exact parallel dims",
      expr->toString());
  TORCH_INTERNAL_ASSERT(
      !expr->isA<MmaOp>(),
      "Mma op: cannot eliminate predicate for mma op, tiling not valid. ",
      expr->toString());
}

} // namespace

namespace {

// Utility to check if the scheduled domain of the given
//   TensorView represent an exact shared mem access, meaning
//   that all the thread parallel dimensions on the leaf nodes
//   are exact so that the shared mem read/write would not
//   run out of bound because of thread over-subscription.
bool isExactParallelSharedMemAccess(TensorView* tv) {
  auto& parallel_dimension_map = GpuLower::current()->parallelDimensionMap();
  for (auto id : tv->domain()->domain()) {
    if (id->isThreadDim()) {
      auto ptype = id->getParallelType();
      // Need to predicate to avoid out of bound access
      //  because of over-subscribed block size.
      if (!parallel_dimension_map.isExact(ptype)) {
        return false;
      }
    }
  }
  return true;
}

class PredicateAnalyzer : public OptOutDispatch {
 public:
  //! Checks if a predicate is needed to avoid out-of-bound accesses.
  //!
  //! Due to the way we allocate local-memory tensors, there should
  //! never be out-of-bound accesses with consumer tensors when allocated on
  //! local memory. However, accessing producer tensors still may
  //! result in out-of-bound as they are replayed as consumers.
  static bool needsPredicate(TensorView* producer, TensorView* consumer) {
    // Both tensors must be on local or shared memory. Global tensors must be
    // predicated as allocation is done based on root domains. Smem
    // and local tensors are allocated based on leaf domains.
    // However, smem tensors are parallelized, which is highly likely, the size
    // of the parallelized axis is the actual size of the axis, not
    // the number of threads. This is currently actively checked to avoid
    // out of bound shared mem access by out of bound threads.
    if (producer->getMemoryType() == MemoryType::Global ||
        consumer->getMemoryType() == MemoryType::Global) {
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
        predicateSharedMemAccess(expr) || predicateProducerConsumerPair(expr) ||
        predicateNonDivisibleRootDomains(expr) ||
        predicateNonDivisibleSplit(expr) || predicateExpandReduce(expr);

    // A cp.async op would need a predicate for either the global
    //  input or its shared mem output, or both.
    // Due to the WAR discussed in [Predicate Inversion for CpAsync],
    //  we currently cannot support use cases where both the gmem read
    //  and the smem write need to be predicated.
    // Adding a check here would make the exclusion of such case as precise as
    //  possible and avoid duplication of predicateSharedMemAccess
    //  logic. But this part along with [Predicate Inversion for CpAsync]
    //  should be cleaned up all together when we extend predicate/masking
    //  logic to cover this usage.
    TORCH_INTERNAL_ASSERT(
        !(ir_utils::isCpAsyncOp(expr) && predicateSharedMemAccess(expr)),
        "predicate removal: unsupported use case of cp.async");

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

  // If we're reducing an expanded domain, we need to be careful to predicate it
  // or we could end up reducing a broadcasted value too many times.
  bool predicateExpandReduce(Expr* expr) const {
    if (!ir_utils::isReductionOp(expr)) {
      return false;
    }
    auto tv_inputs = ir_utils::getTvs(expr->inputs());
    TORCH_INTERNAL_ASSERT(
        tv_inputs.size() > 0,
        "Should never have a reduction op without a tensor view input.");
    bool found_expand = false;
    for (auto tv_input : tv_inputs) {
      found_expand |= std::any_of(
          tv_input->getMaybeRFactorDomain().begin(),
          tv_input->getMaybeRFactorDomain().end(),
          [](IterDomain* id) { return id->hasExpandedExtent(); });
    }

    if (!found_expand) {
      return false;
    }

    auto tv_outputs = ir_utils::getTvs(expr->outputs());
    if (expr->isA<WelfordOp>() && tv_inputs.size() != tv_outputs.size()) {
      tv_outputs = std::vector<TensorView*>(tv_inputs.size(), tv_outputs[0]);
    }

    TORCH_INTERNAL_ASSERT(
        tv_outputs.size() == tv_inputs.size(),
        "Was expecting matching number of inputs and outputs for expression: ",
        expr->toString());

    for (auto i : c10::irange(tv_inputs.size())) {
      const auto root_p2c =
          PairwiseRootDomainMap(tv_inputs[i], tv_outputs[i])
              .mapProducerToConsumer(
                  tv_inputs[i]->domain(), tv_outputs[i]->domain());
      for (auto entry : root_p2c) {
        auto p_id = entry.first;
        auto c_id = entry.second;
        if (p_id->hasExpandedExtent() && c_id->isReduction()) {
          return true;
        }
      }
    }
    return false;
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
    auto halo_info = GpuLower::current()->haloInfo();
    auto input_tvs = ir_utils::filterByType<TensorView>(expr->inputs());
    return halo_info->needsShiftPredicate(expr) ||
        std::any_of(input_tvs.begin(), input_tvs.end(), [&](auto input_tv) {
             return input_tv->definition() != nullptr &&
                 halo_info->needsShiftPredicate(input_tv->definition());
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

  bool predicateSharedMemAccess(Expr* expr) const {
    // This is initial step to gradually remove predicates around
    //  sharedmem access in suitable situations.
    // Using an additional variable to track the predicate-on reasons
    //  when the predicate around shared mem cannot be removed.
    for (auto consumer : ir_utils::filterByType<TensorView>(expr->outputs())) {
      for (auto producer : ir_utils::filterByType<TensorView>(expr->inputs())) {
        if (producer->getMemoryType() == MemoryType::Shared ||
            consumer->getMemoryType() == MemoryType::Shared) {
          if (needSharedMemPredicate(producer, consumer)) {
            return true;
          }
        }
      }
    }

    return false;
  }

  // Check for conditions where the predicate cannot be removed
  //  when either producer or consumer is in shared memory.
  bool needSharedMemPredicate(TensorView* producer, TensorView* consumer)
      const {
    // Indexing is based on consumer leaf ids so check the consumer.

    // If consumer schedule contains in-exact thread parallel
    //  dimensions, need to predicate against out of bound
    //  shared memory access by out of bound threads.
    if (!isExactParallelSharedMemAccess(consumer)) {
      return true;
    }

    // TODO: This is directed WAR on FusionPersistentNormLocalShared.
    //  This use case along with other previous issues motivate a
    //   joint optimization of predicate removal and buffer reuse.
    // In this particular case:
    //   __shared__ T0 [10], T1[10]
    //   for i in ...
    //      if(pred)
    //        T1[i] = T0[i] + ...  // exp0
    //      T2 = 0;              // init for exp1
    //      if(pred)
    //        T2 = T1 ...        // exp1
    //  If we remove pred around expr1, as the way the pred removal
    //    pass is set up, the init for expr will be pushed up to
    //    initialize T1 instead.
    //  However if we initialize T1, the code will look like:
    //  for i in ...
    //    T1[i] = 0;
    //  for i in ...
    //    if(pred)
    //      T1[i] = T0[i] + ...
    //  Note that we'd be able to reuse buffer of T0 for T1 but
    //    if we initialze T1 we cannot do that and thus the
    //    kernel would not fit in smaller devices.
    if (producer->getMemoryType() == MemoryType::Shared) {
      if (auto producer_def = producer->definition()) {
        if (std::any_of(
                producer_def->inputs().begin(),
                producer_def->inputs().end(),
                [](Val* val) {
                  if (auto tv = ir_utils::getTv(val)) {
                    return tv->getMemoryType() == MemoryType::Shared;
                  }
                  return false;
                })) {
          // Disable shared memory producers that is a consumer
          //  of another shared memory tensor. The initialization would
          //  break potential opportunity to re-use shared mem buffer.
          return true;
        }
      }
    }

    for (auto id : consumer->domain()->domain()) {
      // TODO: (Enable in a follow up)
      //  smem predicate removal with init would break unroll and unswitch,
      //  eg. as in issue 1133, so disabling this removal pattern for now.
      if (id->getParallelType() == ParallelType::Unroll ||
          id->getParallelType() == ParallelType::Unswitch) {
        return true;
      }

      // TODO: （Enable in a follow up)
      //  This cannot yet be removed since smem initialization needs to be
      //  handled specially, e.g. as in smem_reduce test. Will be able to
      //  lift this one once the generic pred removal pass with fusion
      //  traversal is ready.
      auto consumer_def = consumer->definition();
      if (ir_utils::isReductionOp(consumer_def)) {
        if (producer->getMemoryType() == MemoryType::Shared) {
          return true;
        }
      }
    }

    return false;
  }

  // Utility to find the leaf iterdomains of the given
  //   tensor view that will be treated as "zero loops"
  //   in the indexing pass.
  // For details on zero loops, see indexMapFromTV in
  //  lower index pass.
  std::vector<Val*> getZeroLeafIds(const TensorView* tv) const {
    TORCH_INTERNAL_ASSERT(
        tv->getMemoryType() == MemoryType::Local ||
            tv->getMemoryType() == MemoryType::Shared,
        "Local or shared memory tensor is assumed: ",
        tv->toString());
    bool is_shared_mem = tv->getMemoryType() == MemoryType::Shared;
    std::vector<Val*> zero_leaf_ids;
    for (const auto i : c10::irange(tv->nDims())) {
      auto leaf_id = tv->axis(i);
      if (is_shared_mem && leaf_id->isThreadDim()) {
        // Thread parallel axes on shared mem are never
        //  zero loops as each thread owns its share
        //  of the shared mem space.
        continue;
      }
      if (
          // Non-thread parallel dimension on the left
          //  of CA axes are zero loops.
          i < tv->getComputeAtPosition() ||
          // Parallel axes on local mem is zero loop.
          // Grid axes on shared mem is zero loop.
          leaf_id->isThread() ||
          // Mma axes, similar to vectorization, are
          //  implicit in hardware intrinsics, and thus
          //  will be treated as a zero loop.
          leaf_id->isMma()) {
        zero_leaf_ids.push_back(leaf_id);
      }
    }

    return zero_leaf_ids;
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
      const auto zero_leaf_ids = getZeroLeafIds(output);
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
        input_def != nullptr, "Inconsistent input found: ", input->toString());

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
    // init value, which is guaranteed by the above check, and the
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
          input_def != nullptr,
          "Inconsistent input found: ",
          input->toString());

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
      if (!input_def->isA<WelfordOp>() && !input_def->isA<GroupedWelfordOp>() &&
          non_predicated_exprs_.find(input_def) !=
              non_predicated_exprs_.end()) {
        needs_predicate_ = true;
        return;
      }
    }
  }

  void handle(GroupedReductionOp* grouped_rop) final {
    for (const auto i : c10::irange(grouped_rop->numExprs())) {
      auto input = grouped_rop->input(i)->as<TensorView>();
      auto input_def = input->definition();
      // When input_def is null, input must be an input to the fusion,
      // so that must be allocated on global memory. Since we don't omit
      // predication for expressions involving global memory, this
      // should never occur.
      TORCH_INTERNAL_ASSERT(
          input_def != nullptr,
          "Inconsistent input found: ",
          input->toString());

      // The input needs to be initialized to the init value to omit
      // the predicate, so if the input has its own init value, i.e.,
      // produced by another reduction, they must use the same init
      // value.
      Val* input_init = ir_utils::getReductionInitValOf(input);
      if (input_init != nullptr &&
          !grouped_rop->initVal(i)->sameAs(input_init)) {
        needs_predicate_ = true;
        return;
      }

      // If input is not predicated, out-of-bound value may be
      // overwritten by a garbage value. However, it doesn't matter if
      // the input is also produced by another reduction. If the preceding
      // reduction omits the predicate, it means its input must be
      // initialized to its init value, so no predicate should be
      // needed in both of the two reduction ops if they use the same
      // init value, which is guaranteed by the above check, and the
      // same reduction op.
      if (auto input_def_rop = dynamic_cast<ReductionOp*>(input_def)) {
        if (grouped_rop->getReductionOpType(i) !=
                input_def_rop->getReductionOpType() &&
            non_predicated_exprs_.find(input_def) !=
                non_predicated_exprs_.end()) {
          needs_predicate_ = true;
          return;
        }
      } else if (
          auto input_def_grouped_rop =
              dynamic_cast<GroupedReductionOp*>(input_def)) {
        auto input_index_as_output =
            input_def_grouped_rop->getExprIndexOfOutput(input);
        if (grouped_rop->getReductionOpType(i) !=
                input_def_grouped_rop->getReductionOpType(
                    input_index_as_output) &&
            non_predicated_exprs_.find(input_def) !=
                non_predicated_exprs_.end()) {
          needs_predicate_ = true;
          return;
        }
      } else if (
          non_predicated_exprs_.find(input_def) !=
          non_predicated_exprs_.end()) {
        needs_predicate_ = true;
        return;
      }
    }
  }

  void handle(GroupedWelfordOp* grouped_wop) final {
    for (const auto expr_idx : c10::irange(grouped_wop->numExprs())) {
      for (const auto val_idx : c10::irange(3)) {
        auto init = grouped_wop->initVals().at(expr_idx).get(val_idx);

        // Welford input can be a scalar. Predicate is required unless
        // the scalar value is equal to the init value.
        auto input = grouped_wop->inputVals().at(expr_idx).get(val_idx);
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
            input_def != nullptr,
            "Inconsistent input found: ",
            input->toString());

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
        // the input is also produced by another reduction op as it
        // must be initialized and its initialized value is already
        // found to be equal to the initil value of this op.
        if (!input_def->isA<WelfordOp>() &&
            !input_def->isA<GroupedWelfordOp>() &&
            non_predicated_exprs_.find(input_def) !=
                non_predicated_exprs_.end()) {
          needs_predicate_ = true;
          return;
        }
      }
    }
  }

  // Similar to the above reduction constraint but for MMA
  void handle(MmaOp* mma) final {
    for (auto input : ir_utils::filterByType<TensorView>(mma->inputs())) {
      auto input_def = input->definition();
      TORCH_INTERNAL_ASSERT(
          input_def != nullptr,
          "Inconsistent input found: ",
          input->toString());

      Val* input_init = ir_utils::getReductionInitValOf(input);
      if (input_init != nullptr && !mma->init()->sameAs(input_init)) {
        needs_predicate_ = true;
        return;
      }

      if (non_predicated_exprs_.find(input_def) !=
          non_predicated_exprs_.end()) {
        // If producer of mma is non_predicated and initialized
        //  with the same value. The mma should not need a
        //  predicate. In fact this is the only way we can
        //  use mma at the moment since we could not predicate
        //  mma ops without guaranteeing warp uniform results.
        auto input_init =
            GpuLower::current()->predicateElimination().getInitValue(input);

        // TODO:
        //   clean up this to support more generic prolog fusion.
        //   Will need additional analysis passes on initialization
        //    propagation and further predicate placement on top.
        // More TODO:
        //  Even when producer is initialized, it is still generally
        //   not safe to remove predicate around reduction ops if the
        //   producer is not predicated.
        //  On the other side, we do have patterns like ldmatrix->mma where
        //   both producer and consumer cannot be safely predicated without
        //   guaranteeing warp uniform results.
        //  This is currently a WAR and relies on validation pass to exclude
        //   complex prolog patterns in mma based matmul kernels. Will
        //   definitely need to revisit and build out predicate and
        //   initialization analysis pass to better handle this case.
        if (input_init != nullptr && !input_init->sameAs(mma->init())) {
          // This is a WAR at the moment. We would need to propagate
          //  initialization information from PredicateElimination
          //  pass to most accurately detect if the input is
          //  initialized correctly.
          // This could also be fixed when we have the traversal
          //  based predicate elimination and initialization pass
          //  ready. Would be easy to clean up this part at that point.
          needs_predicate_ = true;
          return;
        }
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
    TORCH_INTERNAL_ASSERT(
        input_def != nullptr, "Inconsistent input found: ", input->toString());

    // If input is an output of reduction, it should be fully
    // initialied as it's allocated on local memory.
    if (ir_utils::isReductionOp(input_def)) {
      continue;
    }

    if (expr->isA<ReductionOp>()) {
      setReductionInitValue(input, expr->as<ReductionOp>()->init());
      continue;
    } else if (expr->isA<GroupedReductionOp>()) {
      setReductionInitValue(input, expr->as<GroupedReductionOp>()->initVal(i));
      continue;
    } else if (auto wop = dynamic_cast<WelfordOp*>(expr)) {
      Val* init = wop->getInitVals().at(i);
      setReductionInitValue(input, init);
      continue;
    } else if (expr->isA<MmaOp>()) {
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
        existing_val->toString(),
        ", New: ",
        reduction_init->toString());
    return false;
  }
}

bool PredicateElimination::canOmitPredicate(const Expr* expr) const {
  // Predicate elimination can be disabled with
  // PYTORCH_NVFUSER_DISABLE=predicate_elimination
  if (isOptionDisabled(DisableOption::PredicateElimination)) {
    assertOnWarpOps(expr);
    return false;
  }

  TORCH_INTERNAL_ASSERT(expr != nullptr);
  const auto out_tv = ir_utils::getTvOutput(expr);
  TORCH_INTERNAL_ASSERT(out_tv != nullptr, "Not a tensor expression");

  if (ir_utils::isTensorScalarFillOp(expr)) {
    if (out_tv->getMemoryType() == MemoryType::Local) {
      // Filling a local tensor with scalar shouldn't
      //   need any predicate currently.
      return true;
    } else if (out_tv->getMemoryType() == MemoryType::Shared) {
      // A shared memory initialization should be same except
      //  that we'd need a predicate to guard against out of
      //  bound access by out of inexact threads.
      return isExactParallelSharedMemAccess(out_tv);
    }
  }

  if (non_predicated_exprs_.find(expr) != non_predicated_exprs_.end()) {
    return true;
  }

  assertOnWarpOps(expr);
  return false;
}

void PredicateElimination::propagateRemovalInfo(
    const Expr* from,
    const Expr* to) {
  if (non_predicated_exprs_.count(from)) {
    non_predicated_exprs_.insert(to);
  }
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
  traverseTo(fusion, fusion->outputs());
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
