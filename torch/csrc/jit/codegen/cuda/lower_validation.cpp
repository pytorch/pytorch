#include <torch/csrc/jit/codegen/cuda/lower_validation.h>

#include <torch/csrc/jit/codegen/cuda/expr_evaluator.h>
#include <torch/csrc/jit/codegen/cuda/instrumentation.h>
#include <torch/csrc/jit/codegen/cuda/ir_iostream.h>
#include <torch/csrc/jit/codegen/cuda/ir_utils.h>
#include <torch/csrc/jit/codegen/cuda/iter_visitor.h>
#include <torch/csrc/jit/codegen/cuda/kernel_ir_printer.h>
#include <torch/csrc/jit/codegen/cuda/lower2device.h>
#include <torch/csrc/jit/codegen/cuda/lower_utils.h>
#include <torch/csrc/jit/codegen/cuda/transform_replay.h>
#include <torch/csrc/jit/codegen/cuda/type.h>

#include <limits>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

namespace {

//! A parallel type validation pass to make sure all the outputs of
//!   welford ops are parallelized the same way. Will infer and modify serial
//!   parallel types if other output/s are parallelized, so that
//!   user wouldn't have to specify the same parallelization
//!   3 times. Will throw if conflicts are detected, i.e.
//!   TIDx vs BIDx etc.
class ValidateParallelType : public IterVisitor {
 public:
  static void validate(Fusion* fusion) {
    ValidateParallelType VPT;
    VPT.traverse(fusion);
  }

 private:
  using IterVisitor::handle;
  // Parallelize id1 and id0 consistently if one is serial and the other isn't
  void convertIterDomain(IterDomain* id0, IterDomain* id1) {
    const auto ptype0 = id0->getParallelType();
    const auto ptype1 = id1->getParallelType();

    if (ptype0 == ParallelType::Vectorize ||
        ptype1 == ParallelType::Vectorize) {
      auto other_type = ptype0 == ParallelType::Vectorize ? ptype1 : ptype0;
      TORCH_INTERNAL_ASSERT(
          other_type == ParallelType::Vectorize ||
              (!isParallelTypeThreadDim(other_type) &&
               !isParallelTypeBlockDim(other_type)),
          "Vectorize type was parallelized inconsistently in. ",
          "Detected during promoting parallel types.");
      return;
    }

    if (ptype0 != ptype1) {
      TORCH_CHECK(
          ptype0 == ParallelType::Serial || ptype1 == ParallelType::Serial,
          "Error promoting parallel types");
      if (ptype0 == ParallelType::Serial) {
        id0->parallelize(ptype1);
      }
      if (ptype1 == ParallelType::Serial) {
        id1->parallelize(ptype0);
      }
    }
  }

  void handle(WelfordOp* wop) override {
    auto out_avg = wop->outAvg()->as<TensorView>();
    auto out_var = wop->outVar()->as<TensorView>();
    auto out_n = wop->outN()->as<TensorView>();
    TORCH_INTERNAL_ASSERT(out_avg->nDims() == out_var->nDims());
    TORCH_INTERNAL_ASSERT(out_avg->nDims() == out_n->nDims());
    for (size_t i = 0; i < out_avg->nDims(); i++) {
      // TODO: can be cleaner.
      convertIterDomain(out_avg->axis(i), out_var->axis(i));
      convertIterDomain(out_avg->axis(i), out_n->axis(i));
      convertIterDomain(out_n->axis(i), out_var->axis(i));
    }
  }
};

// Make sure all IterDomains are only used for a unique
// TensorView. Several mappings from IterDomains are
// created during lowering, which relies on the unique usage of
// IterDomains.
void validateIterDomainUsage(Fusion* fusion) {
  FUSER_PERF_SCOPE("GpuLower::Lower::validateIterDomainUse");
  FusionGuard fg(fusion);

  auto used_vals = fusion->usedMathVals();
  std::unordered_map<IterDomain*, TensorView*> domain_use_map;

  for (auto tv : ir_utils::filterByType<TensorView>(used_vals)) {
    std::unordered_set<Val*> root_domains;
    std::copy(
        tv->getRootDomain().begin(),
        tv->getRootDomain().end(),
        std::inserter(root_domains, root_domains.begin()));

    std::vector<Val*> leaf_domains;
    std::copy(
        tv->domain()->domain().begin(),
        tv->domain()->domain().end(),
        std::back_inserter(leaf_domains));

    auto all_domain_vals =
        DependencyCheck::getAllValsBetween(root_domains, leaf_domains);

    for (auto id : ir_utils::filterByType<IterDomain>(all_domain_vals)) {
      auto it = domain_use_map.find(id);
      TORCH_INTERNAL_ASSERT(
          it == domain_use_map.end(),
          "Multiple use of ",
          id,
          " detected.",
          " Used in both TV",
          tv->name(),
          " and TV",
          it->second->name());
      domain_use_map.insert({id, tv});
    }
  }
}

} // namespace

void validateIr(Fusion* fusion) {
  FUSER_PERF_SCOPE("GpuLower::Lower::validateIr");

  FusionGuard fg(fusion);

  fusion->validateInputs();

  // Convert all input broadcast iterdomains to strided
  for (auto tv : ir_utils::filterByType<TensorView>(fusion->inputs())) {
    for (auto id : tv->getMaybeRFactorDomain()) {
      if (id->isBroadcast()) {
        id->toStridedBroadcast();
      }
    }
  }

  // Convert all output broadcast iterdomains to strided
  for (auto tv : ir_utils::filterByType<TensorView>(fusion->outputs())) {
    for (auto id : tv->getMaybeRFactorDomain()) {
      if (id->isBroadcast()) {
        id->toStridedBroadcast();
      }
    }
  }

  // Validate Parallelization
  ValidateParallelType::validate(fusion);

  validateIterDomainUsage(fusion);
}

namespace {

// Check contiguity for all root domains associated with Misaligned Vectorize
// ParallelType
void checkContiguity(
    const std::unordered_set<IterDomain*>& domains,
    TensorView* tv) {
  TORCH_INTERNAL_ASSERT(tv->getMemoryType() == MemoryType::Global);

  for (size_t idx = 0; idx < tv->getRootDomain().size(); ++idx) {
    auto root = tv->getRootDomain()[idx];
    if (domains.find(root) != domains.end()) {
      TORCH_INTERNAL_ASSERT(
          !root->isBroadcast(),
          "Misaligned vectorization prohibits merging broadcast domains.",
          "Issue found in, ",
          tv);
      TORCH_INTERNAL_ASSERT(
          tv->domain()->contiguity()[idx],
          "Cannot merge non-contiguous root domains with misaligned vectorization.",
          "Issue found in, ",
          tv);
    }
  }
}

// Check all root iter domains in consumer that are present in domain, making
// sure they're contiguous. Map these domains to producer and make sure they are
// also contiguous in producer. Producer-consumer relationship is assumed to be
// through a set operation.
void checkContiguity(
    const std::unordered_set<IterDomain*>& domains,
    TensorView* consumer,
    TensorView* producer) {
  // This seems not quite right, shouldn't we be able to reverse this?
  TORCH_INTERNAL_ASSERT(consumer->getMemoryType() == MemoryType::Local);
  TORCH_INTERNAL_ASSERT(producer->getMemoryType() == MemoryType::Global);

  auto root_c2p =
      PairwiseRootDomainMap(producer, consumer)
          .mapConsumerToProducer(consumer->domain(), producer->domain());

  std::unordered_map<IterDomain*, bool> producer_domain_contiguity;
  for (size_t idx = 0; idx < producer->getRootDomain().size(); ++idx) {
    auto root = producer->getRootDomain()[idx];
    auto contiguity = producer->domain()->contiguity()[idx];
    producer_domain_contiguity.insert({root, contiguity});
  }

  for (auto consumer_root : consumer->getRootDomain()) {
    if (domains.find(consumer_root) != domains.end()) {
      auto producer_root = root_c2p[consumer_root];
      TORCH_INTERNAL_ASSERT(
          producer_domain_contiguity.find(producer_root) !=
          producer_domain_contiguity.end());

      TORCH_INTERNAL_ASSERT(
          !consumer_root->isBroadcast() || !producer_root->isBroadcast(),
          "Misaligned vectorization prohibits merging broadcast domains.",
          "Issue found in, ",
          consumer);

      TORCH_INTERNAL_ASSERT(root_c2p.find(consumer_root) != root_c2p.end());

      TORCH_INTERNAL_ASSERT(
          producer_domain_contiguity[producer_root],
          "Cannot merge non-contiguous root domains with misaligned vectorization.",
          "Issue found in, ",
          consumer);
    }
  }
}

class VectorizeValidator : public OptInDispatch {
 private:
  // Initially, vectorized_id is the IterDomain with Vectorize ParallelType
  // After processing all merge and split operations,
  // vectorized_id is the corresponding root domain
  VectorizeValidator(IterDomain* vectorized_id)
      : vectorized_id_(vectorized_id) {}

  using OptInDispatch::handle;

  void handle(Split* s) final {
    if (s->outer() == vectorized_id_) {
      is_valid = false;
    } else if (s->inner() == vectorized_id_) {
      vectorized_id_ = s->in();
    }
    domains_.insert(s->outer());
    domains_.insert(s->inner());
  }

  void handle(Merge* m) final {
    if (m->out() == vectorized_id_) {
      if (m->inner()->isBroadcast() && !m->outer()->isBroadcast()) {
        vectorized_id_ = m->outer();
      } else {
        vectorized_id_ = m->inner();
      }
    }
    domains_.insert(m->outer());
    domains_.insert(m->inner());
  }

 private:
  std::unordered_set<IterDomain*> domains_;
  IterDomain* vectorized_id_ = nullptr;
  bool is_valid = true;

 public:
  static void validate(TensorView* tv) {
    // Make sure there's only one vectorized ID
    IterDomain* v_id = nullptr;
    bool misaligned_vectorize = false;
    for (auto id : tv->domain()->domain()) {
      if (id->getParallelType() == ParallelType::Vectorize ||
          id->getParallelType() == ParallelType::MisalignedVectorize) {
        TORCH_INTERNAL_ASSERT(
            v_id == nullptr,
            "Found two vectorized domains in ",
            tv,
            " only one is allowed.");
        v_id = id;
        misaligned_vectorize =
            id->getParallelType() == ParallelType::MisalignedVectorize;
      }
    }

    // If no vectorized id's found simply return;
    if (v_id == nullptr) {
      return;
    }

    auto fusion = FusionGuard::getCurFusion();

    TORCH_CHECK(
        v_id->extent()->isConstScalar(),
        "Vectorizing a domain requires a constant size.");

    ExpressionEvaluator const_expr_eval(fusion);

    auto vector_size_optional = const_expr_eval.evaluate(v_id->extent());

    TORCH_CHECK(
        vector_size_optional.has_value(),
        "Could not evaluate constant value bound to vectorized dim.");

    auto vector_size = ((int64_t)dataTypeSize(tv->getDataType().value())) *
        vector_size_optional.value();

    // Allow half2, float2, float4 and same sized vtypes.
    std::array<int64_t, 4> allowed_vector_sizes = {2, 4, 8, 16}; // NOLINT

    TORCH_CHECK(
        std::find(
            allowed_vector_sizes.begin(),
            allowed_vector_sizes.end(),
            vector_size) != allowed_vector_sizes.end(),
        "Tried to vectorize a dim resulting in a word size of ",
        vector_size,
        " however, vector sizes only upto and including 16 bytes are supported.");

    auto replay_exprs = ExprSort::getExprs(fusion, {v_id});

    VectorizeValidator validator(v_id);

    for (auto expr_it = replay_exprs.rbegin(); expr_it != replay_exprs.rend();
         ++expr_it) {
      auto expr = *expr_it;
      validator.handle(expr);
    }

    TORCH_CHECK(
        validator.is_valid,
        "Invalid vectorized pattern found, vectorization iter domains must be descendants of inner-most dimension.",
        "Issue found in, ",
        tv,
        "\n");

    if (misaligned_vectorize) {
      if (tv->getMemoryType() == MemoryType::Global) {
        checkContiguity(validator.domains_, tv);
      } else if (
          tv->definition()->getExprType() == ExprType::UnaryOp &&
          tv->definition()->as<UnaryOp>()->getUnaryOpType() ==
              UnaryOpType::Set) {
        auto input = tv->definition()->input(0);
        TORCH_INTERNAL_ASSERT(input->isA<TensorView>());
        auto input_tv = input->as<TensorView>();
        checkContiguity(validator.domains_, tv, input_tv);
      }
    }

    TORCH_INTERNAL_ASSERT(validator.vectorized_id_ != nullptr);

    // TODO: Contiguity is based on root domain not rfactor. Seems this
    // generally doesn't cause problems, though contiguity should be on rfactor
    // domain as that's the domain we index on.
    IterDomain* last_root_dim = nullptr;
    int last_root_dim_pos = -1;
    for (size_t i = tv->getRootDomain().size(); i > 0; i--) {
      auto r_id = tv->getRootDomain()[i - 1];
      if (r_id->isReduction() || r_id->isBroadcast()) {
        continue;
      }
      last_root_dim = r_id;
      last_root_dim_pos = (int)i - 1;
      break;
    }

    if (last_root_dim == nullptr) {
      // Should never get here, but that would mean there are no concrete dims,
      // so we should be fine.
      return;
    }

    TORCH_CHECK(
        last_root_dim == validator.vectorized_id_ &&
            tv->domain()->contiguity()[last_root_dim_pos],
        "Vectorized dim has to be from a contiguous inner most position: ",
        tv,
        "\n");
  }
};

} // namespace

void validateVectorize(Fusion* fusion) {
  FUSER_PERF_SCOPE("GpuLower::Lower::validateVectorize");
  FusionGuard fg(fusion);

  auto used_vals = fusion->usedMathVals();

  std::unordered_set<TensorView*> used_tvs;

  for (auto val : used_vals) {
    if (ir_utils::isTV(val)) {
      used_tvs.emplace(val->as<TensorView>());
    }
  }

  for (auto tv : used_tvs) {
    bool has_vectorize_dim = false;
    bool has_misaligned_vectorize_dim = false;

    for (size_t i = 0; i < tv->nDims(); i++) {
      IterDomain* id = tv->axis(i);
      IterDomain* concrete_id =
          GpuLower::current()->caParallelMap().getConcreteMappedID(id);

      auto ptype = concrete_id->getParallelType();

      if (ptype == ParallelType::Vectorize) {
        // If we want to do this check up front we would have to do 2 things:
        // (1) Check that the tensor view with vectorize being set on it is
        // getting set outside the local compute at position
        // (2) Check any producers of the tensor view with vectorize being set
        // on it to make sure their compute at position isn't to the right of
        // the vectorize dim.
        TORCH_INTERNAL_ASSERT(
            i >= tv->getComputeAtPosition(),
            "IterDomains to the left of the compute at point cannot be vectorized: ",
            tv,
            "\n");
        has_vectorize_dim = true;
      }

      if (concrete_id->getParallelType() == ParallelType::MisalignedVectorize) {
        TORCH_INTERNAL_ASSERT(
            !tv->hasComputeAt() ||
                tv->getComputeAtPosition() == tv->nDims() - 1,
            "Only allow misaligned vectorization in the -2 computeAt position.");
        TORCH_INTERNAL_ASSERT(
            tv->getMemoryType() == MemoryType::Local ||
                tv->getMemoryType() == MemoryType::Global,
            "Only allow misaligned vectorization between global and local memory.");
        has_misaligned_vectorize_dim = true;
      }
    }
    if (has_vectorize_dim) {
      TORCH_INTERNAL_ASSERT(
          tv->definition() == nullptr ||
              (tv->definition()->isA<UnaryOp>() &&
               tv->definition()->as<UnaryOp>()->getUnaryOpType() ==
                   UnaryOpType::Set),
          "Vectorized accesses cannot be inline with computation, they are only supported with a Set operation.",
          "TensorView: ",
          tv);
    }
    if (has_vectorize_dim || has_misaligned_vectorize_dim) {
      VectorizeValidator::validate(tv);
    }
  }
}

void validateParallelize(Fusion* fusion) {
  FUSER_PERF_SCOPE("GpuLower::Lower::validateParallelize");
  FusionGuard fg(fusion);

  const auto& par_map = GpuLower::current()->caParallelMap();
  const auto& loop_map = GpuLower::current()->caLoopMap();
  const auto& index_map = GpuLower::current()->caIndexMap();
  const auto& pred_map = GpuLower::current()->threadPredMap();

  auto exprs = ExprSort::getExprs(fusion);

  for (auto expr : exprs) {
    if (!ir_utils::isTVOp(expr)) {
      continue;
    }
    for (auto producer : ir_utils::filterByType<TensorView>(expr->inputs())) {
      // Parallelization on input tensors have no effect.
      if (producer->isFusionInput()) {
        continue;
      }
      const auto parallel_bcast_doms =
          pred_map.getParallelBroadcastDomains(producer);
      ParallelTypeBitmap pt_map;
      for (size_t i = 0; i < producer->nDims(); ++i) {
        // If a producer axis is threaded, either with threadIdx or
        // blockIdx, there must be a mapped consumer axis with the
        // same ParallelType. An exception is when the producer is
        // allocated on shared memory and its parallelized with
        // threadIdx. In that case, there is no parallelization
        // constraint on the consumer as syncthreads will be inserted
        // when necessary.
        auto producer_axis = producer->axis(i);
        auto producer_ptype =
            par_map.getConcreteMappedID(producer_axis)->getParallelType();
        if (!isParallelTypeThread(producer_ptype)) {
          continue;
        }
        // Each ParallelType can be used only once.
        TORCH_INTERNAL_ASSERT(
            !pt_map.get(producer_ptype),
            "Multiple use of ",
            producer_ptype,
            " in tensor t",
            producer->name(),
            ": ",
            producer);
        pt_map.set(producer_ptype, true);
        // When the producer axis is a broadcast, it is not really
        // parallelized unless thread-predicated
        if (producer_axis->isBroadcast() && parallel_bcast_doms.none()) {
          continue;
        }
        // No constraint on the consumer tensor when the producer
        // axis is parallelized with threadIdx and allocates on
        // shared memory
        if (isParallelTypeThreadDim(producer_ptype) &&
            producer->getMemoryType() == MemoryType::Shared) {
          continue;
        }
        // There should be also nothing to validate when the producer
        // axis is reduction.
        if (producer_axis->isReduction()) {
          continue;
        }
        // There must be a consumer axis that uses the same indexing
        // with the same parallel type as the producer axis. The index
        // map is used to to find such an axis. In addition, even when
        // no mapped axis is found in the index map, but when an
        // mapped axis exists in the loop map, the producer and
        // consumer axes may still use the same indexing. That only
        // happens when the producer is derived from a root axis that
        // is an input to any leaf CA axes. In such a case, the axis
        // in the reference tensor that maps to
        // the producer axis is created based on the consumer, so both
        // the producer and consumer axes should have the same
        // indexing. See issue #995 as well as the
        // FusionValidateParallelize6 test for a concrete example.
        for (auto consumer :
             ir_utils::filterByType<TensorView>(expr->outputs())) {
          auto it = std::find_if(
              consumer->domain()->domain().begin(),
              consumer->domain()->domain().end(),
              [&](IterDomain* consumer_axis) {
                return index_map.areMapped(producer_axis, consumer_axis) ||
                    (loop_map.areMapped(producer_axis, consumer_axis) &&
                     ir_utils::derivedFromRootCAAxes(producer, producer_axis));
              });
          TORCH_INTERNAL_ASSERT(
              it != consumer->domain()->domain().end(),
              "Inconsistent parallelization found between TV",
              producer->name(),
              " (",
              producer,
              ") and TV",
              consumer->name(),
              "(",
              consumer,
              "). ",
              "TV",
              consumer->name(),
              " does not have a matching axis for parallelized producer axis, ",
              producer_axis,
              ". CA Map: ",
              loop_map.toString());
          auto consumer_axis = *it;
          auto consumer_ptype =
              par_map.getConcreteMappedID(consumer_axis)->getParallelType();
          TORCH_INTERNAL_ASSERT(
              producer_ptype == consumer_ptype,
              "Inconsistent parallelization found between TV",
              producer->name(),
              " (",
              producer,
              ") and TV",
              consumer->name(),
              "(",
              consumer,
              "). "
              "Producer axis, ",
              producer_axis,
              " is parallelized with ",
              stringifyThread(producer_ptype),
              ", but the parallel type of its matching consumer axis, ",
              consumer_axis,
              " is ",
              stringifyThread(consumer_ptype),
              ".");
        }
      }
    }
  }
}

namespace {

// Backward propagation of partial ranges from outputs to
// inputs. Necessary to determine required ranges to compute.
//
// Example:
//  tv0: [0:N]
//  tv1: shift(tv0, {1}) -> [1:N]
//  tv2: shift(tv0, {-1}) -> [0:N-1]
//  tv3: tv1 + tv2 -> [1:N-1]
//
// In this case, the valid range of tv3 starts at 1 and ends at
// N-1. This means that not all of the values of tv1 and tv2 are
// actually necessary. Specifically, tv1[0] and tv2[N-1] aren't used
// for tv3. This function calculates the required minimum range of
// each tensor that needs to be computed.
std::unordered_map<IterDomain*, std::pair<int64_t, int64_t>> getLiveRangeOffsets(
    Fusion* fusion) {
  auto exprs = ExprSort::getExprs(fusion);

  std::unordered_map<IterDomain*, std::pair<int64_t, int64_t>> map;

  ExpressionEvaluator ee(fusion);

  for (auto it = exprs.rbegin(); it != exprs.rend(); ++it) {
    auto expr = *it;
    for (auto consumer : ir_utils::filterByType<TensorView>(expr->outputs())) {
      for (auto consumer_root : consumer->getRootDomain()) {
        auto consumer_start_offset = ee.evaluate(consumer_root->start());
        auto consumer_stop_offset = ee.evaluate(consumer_root->stopOffset());
        TORCH_INTERNAL_ASSERT(
            consumer_start_offset.has_value(),
            "Can't evaluate start value of ",
            consumer_root->start());
        TORCH_INTERNAL_ASSERT(
            consumer_stop_offset.has_value(),
            "Can't evaluate stop value of ",
            consumer_root->stopOffset());
        auto it = map.find(consumer_root);
        if (it == map.end() || consumer->isFusionOutput()) {
          // No range set for this root domain, which means this
          // consumer_tensor is an output tensor or the consumer_root
          // domain is a reduction domain. In either case, the
          // required range is simply defined by the start and stop
          // offsets of the root domain.
          // Also, when consumer is an output, even if it's not
          // terminating, the range to compute must not be affected by
          // how it's used by its consumers because an output tensor
          // is visible to outside of the fusion.
          map.insert(
              {consumer_root,
               {consumer_start_offset.value(), consumer_stop_offset.value()}});
        } else {
          // When the range of this root domain is already set, it
          // must be set by its consumers. Make sure the required
          // range by the consumers is covered by the defined range of
          // this root domain.
          auto& consumer_range = it->second;
          TORCH_INTERNAL_ASSERT(
              consumer_start_offset.value() <= consumer_range.first);
          TORCH_INTERNAL_ASSERT(
              consumer_stop_offset.value() <= consumer_range.second);
        }
      }

      // Propagate the range information from consumers to the
      // produces. Note that the effect on the range by shift and
      // gather is not considered here but taken care by halo regions.
      for (auto producer : ir_utils::filterByType<TensorView>(expr->inputs())) {
        auto c2p =
            PairwiseRootDomainMap(producer, consumer)
                .mapConsumerToProducer(consumer->domain(), producer->domain());
        for (auto consumer_root : consumer->getRootDomain()) {
          auto producer_it = c2p.find(consumer_root);
          if (producer_it == c2p.end()) {
            continue;
          }
          auto producer_root = producer_it->second;
          auto& consumer_range = map.at(consumer_root);
          const std::pair<int64_t, int64_t> init_range{
              std::numeric_limits<int64_t>::max(),
              std::numeric_limits<int64_t>::max()};
          auto& producer_range =
              map.insert({producer_root, init_range}).first->second;
          producer_range.first =
              std::min(producer_range.first, consumer_range.first);
          producer_range.second =
              std::min(producer_range.second, consumer_range.second);
        }
      }
    }
  }

  return map;
}

// Make sure that a partial split with split_offset does not violate
// the required range defined by domain_offset. Suppose checking the
// start side of a root domain. Only positions at split_offset or
// larger are going to be computed, and all positions starting at
// domain_offset must be computed, thus split_offset must be smaller
// or equal to domain_offset. The same condition must hold for the end
// side of the domain.
//
// In order to validate this condition, the split offset is assumed to
// be a statically known constant value. This is not a hard
// requirement, but otherwise a runtime check would be needed.
void validateSplit(
    Val* split_offset,
    int64_t domain_offset,
    const std::string& err_msg_prefix) {
  ExpressionEvaluator ee(split_offset->fusion());

  TORCH_INTERNAL_ASSERT(split_offset->isA<Int>());
  auto split_offset_value = ee.evaluate(split_offset);
  TORCH_INTERNAL_ASSERT(
      split_offset_value.has_value(),
      err_msg_prefix,
      ": Unknown offset of split: ",
      split_offset);

  TORCH_INTERNAL_ASSERT(
      split_offset_value.value() <= domain_offset,
      err_msg_prefix,
      ": Split offset is larger than the domain offset.",
      " Split offset: ",
      split_offset_value.value(),
      ". Domain offset: ",
      domain_offset);
}

} // namespace

void validatePartialSplit(Fusion* fusion) {
  FUSER_PERF_SCOPE("GpuLower::Lower::validatePartialSplit");
  FusionGuard fg(fusion);

  // If a root domain is partially split, only the sub range defined
  // by the start and stop offsets of the partial split is
  // computed. That sub range must cover the required range of the
  // domain. So, the first thing to do is to determine the required
  // minimum range of each root domain. Then, check if any partial
  // split could result in a smaller range than the required range.

  // Compute the required range of each root domain
  auto range_info = getLiveRangeOffsets(fusion);

  for (auto tv : ir_utils::allTvs(fusion)) {
    auto exprs = ir_utils::historyOf(tv);
    for (auto split : ir_utils::filterByType<Split>(exprs)) {
      // When the start and stop offsets are not zero, make sure the
      // range defined by the split includes the required range to
      // compute. If both of the split offsets are zero, this
      // condition is obviously true. Also, this validation only needs
      // to be done with root domains. Since the start and stop
      // offsets of non-root domains must be just zero, they are
      // skipped at this point.
      if (split->startOffset()->isZeroInt() &&
          split->stopOffset()->isZeroInt()) {
        continue;
      }
      auto root_domain = split->in();
      std::stringstream err_msg_prefix;
      err_msg_prefix << "Error with " << root_domain << " in T" << tv->name();
      TORCH_INTERNAL_ASSERT(range_info.find(root_domain) != range_info.end());
      const auto& valid_range = range_info.at(root_domain);
      // Check the start offset. If it's zero, no validation regarding
      // the required range can occur.
      if (!split->startOffset()->isZeroInt()) {
        validateSplit(
            split->startOffset(), valid_range.first, err_msg_prefix.str());
      }
      // Same for the stop offset.
      if (!split->stopOffset()->isZeroInt()) {
        validateSplit(
            split->stopOffset(), valid_range.second, err_msg_prefix.str());
      }
    }
  }
}

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
