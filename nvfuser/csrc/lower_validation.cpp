#include <lower_validation.h>

#include <contiguity.h>
#include <expr_evaluator.h>
#include <instrumentation.h>
#include <ir_iostream.h>
#include <ir_utils.h>
#include <iter_visitor.h>
#include <lower2device.h>
#include <lower_utils.h>
#include <transform_iter.h>
#include <transform_replay.h>
#include <type.h>

#include <ATen/cuda/CUDAContext.h>
#include <limits>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

namespace {

//! Validate multiple output tensors of the same expression, i.e.,
//! siblings, have valid domains and parallel types. Since siblings
//! are placed in the same loop nest, they must be parallelized the
//! same way. Will infer and modify serial parallel types if other
//! output/s are parallelized, so that user wouldn't have to specify
//! the same parallelization 3 times. Will throw if conflicts are
//! detected, i.e. TIDx vs BIDx etc.
class ValidateSiblings : public IterVisitor {
 public:
  static void validate(Fusion* fusion) {
    ValidateSiblings validator;
    validator.traverse(fusion);
  }

 private:
  using IterVisitor::handle;

  void handle(Expr* expr) final {
    if (!ir_utils::isTvOp(expr) || expr->outputs().size() < 2) {
      IterVisitor::handle(expr);
      return;
    }

    auto ref_output = expr->outputs().at(0)->as<TensorView>();
    auto ref_ndims = ref_output->nDims();
    const auto& ref_root = ref_output->getRootDomain();
    std::unordered_map<IterDomain*, IterDomain*> id_map;

    for (const auto sibling :
         ir_utils::filterByType<TensorView>(expr->outputs())) {
      if (ref_output == sibling) {
        continue;
      }

      TORCH_INTERNAL_ASSERT(
          sibling->nDims() == ref_ndims,
          "Mismatched dimensionality detected. Expr: ",
          expr->toString(),
          "Ref output: ",
          ref_output->toString(),
          ". Sibling: ",
          sibling->toString());

      for (const auto i : c10::irange(ref_ndims)) {
        validateParallelTypes(ref_output->axis(i), sibling->axis(i));
      }

      for (const auto i : c10::irange(ref_root.size())) {
        id_map[ref_root[i]] = sibling->getRootDomain().at(i);
      }

      BestEffortReplay replay(
          sibling->domain()->domain(), ref_output->domain()->domain(), id_map);
      for (const auto i : c10::irange(ref_ndims)) {
        auto it = replay.getReplay().find(ref_output->axis(i));
        TORCH_INTERNAL_ASSERT(
            it != replay.getReplay().end(),
            "Matching sibling ID not found. Expr: ",
            expr->toString(),
            "Ref ID: ",
            ref_output->axis(i)->toString());
        auto sibling_id = it->second;
        TORCH_INTERNAL_ASSERT(
            sibling->axis(i) == sibling_id,
            "Invalid matching sibling ID detected. Expr: ",
            expr->toString(),
            "Sibling ID: ",
            sibling_id->toString());
      }
    }
  }

  // Parallelize id1 and id0 consistently if one is serial and the other isn't
  void validateParallelTypes(IterDomain* id0, IterDomain* id1) {
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

  // Validate Parallelization
  ValidateSiblings::validate(fusion);

  validateIterDomainUsage(fusion);
}

namespace {

// Check contiguity for all root domains associated with Misaligned Vectorize
// ParallelType
void checkContiguity(
    const std::unordered_set<IterDomain*>& domains,
    TensorView* tv) {
  TORCH_INTERNAL_ASSERT(tv->getMemoryType() == MemoryType::Global);

  for (const auto idx : c10::irange(tv->getRootDomain().size())) {
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
  for (const auto idx : c10::irange(producer->getMaybeRFactorDomain().size())) {
    auto root = producer->getMaybeRFactorDomain()[idx];
    auto contiguity = producer->domain()->contiguity()[idx];
    producer_domain_contiguity.insert({root, contiguity});
  }

  for (auto consumer_root : consumer->getMaybeRFactorDomain()) {
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

  // For the producer tensor, it's indexed first by transformed like
  // the consumer. So, to find its contig merged domain, use the
  // consumer TensorDomain with the producer contiguity info.
  static std::vector<bool> mapProducerContiguity(
      TensorView* producer_tv,
      TensorView* consumer_tv) {
    const auto c2p = PairwiseRootDomainMap(producer_tv, consumer_tv)
                         .mapConsumerToProducer(
                             consumer_tv->domain(), producer_tv->domain());

    std::vector<bool> producer_contiguity;

    for (auto consumer_root_id : consumer_tv->getRootDomain()) {
      auto producer_root_id = c2p.at(consumer_root_id);
      auto producer_root_it = std::find(
          producer_tv->getMaybeRFactorDomain().begin(),
          producer_tv->getMaybeRFactorDomain().end(),
          producer_root_id);
      TORCH_INTERNAL_ASSERT(
          producer_root_it != producer_tv->getMaybeRFactorDomain().end());
      auto producer_root_id_offset = std::distance(
          producer_tv->getMaybeRFactorDomain().begin(), producer_root_it);
      producer_contiguity.push_back(
          producer_tv->domain()->contiguity().at(producer_root_id_offset));
    }

    return producer_contiguity;
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

    // If no vectorized ids found simply return. If vectorized access is
    // broadcast, it won't generate an actual vector instruction, so can safely
    // be ignore
    if (v_id == nullptr || v_id->isBroadcast()) {
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

    auto replay_exprs = DependencyCheck::getAllExprsBetween(
        {tv->getMaybeRFactorDomain().begin(),
         tv->getMaybeRFactorDomain().end()},
        {v_id});

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

    // Contiguity is based on rfactor domain.
    IterDomain* last_root_dim = nullptr;
    int last_root_dim_pos = -1;
    for (size_t i = tv->getMaybeRFactorDomain().size(); i > 0; i--) {
      auto r_id = tv->getMaybeRFactorDomain()[i - 1];
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

    // Save info required to lowering and runtime validation
    auto consumer_word_size_it =
        GpuLower::current()->vectorizedAccesses().find(tv);
    if (consumer_word_size_it !=
        GpuLower::current()->vectorizedAccesses().end()) {
      consumer_word_size_it->second = std::max(
          (int)vector_size_optional.value(), consumer_word_size_it->second);
    } else {
      GpuLower::current()->vectorizedAccesses().emplace(
          tv, (int)vector_size_optional.value());
    }
    auto producer_tv = tv->definition()->inputs().at(0)->as<TensorView>();
    auto producer_word_size_it =
        GpuLower::current()->vectorizedAccesses().find(producer_tv);
    if (producer_word_size_it !=
        GpuLower::current()->vectorizedAccesses().end()) {
      producer_word_size_it->second = std::max(
          (int)vector_size_optional.value(), producer_word_size_it->second);
    } else {
      GpuLower::current()->vectorizedAccesses().emplace(
          producer_tv, (int)vector_size_optional.value());
    }

    VectorizedSetInfo vectorized_set_info;
    vectorized_set_info.consumer_tv = tv;
    vectorized_set_info.producer_tv = producer_tv;
    // Note that VectorizedSetInfo is about each instance of
    // vectorized set operations, so the word size is the size of this
    // specific vectorized set.
    vectorized_set_info.word_size = (int)vector_size_optional.value();
    vectorized_set_info.vectorized_leaf_id = v_id;
    vectorized_set_info.vectorized_root_id = validator.vectorized_id_;
    // For aligned vectorize, the extent of a vectorized domain must
    // be divisible by the vector word size. The domain is usually
    // just one of the root domains, but can be a merged domain of
    // contiguous domains. Those domains are saved in
    // VectorizedSetInfo.contig_root_ids at the time of indexing.
    GpuLower::current()->vectorizedSetInfo().emplace_back(vectorized_set_info);
  }
};

} // namespace

// Uses ContigIDs to find root contig domains that a vectorized domain
// depends on. As ContigIDs depends on HaloInfo, this must be done
// after HaloInfo is created.
void validateAndCollectVectorizeInfo(Fusion* fusion) {
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

    for (const auto i : c10::irange(tv->nDims())) {
      IterDomain* id = tv->axis(i);
      IterDomain* concrete_id =
          GpuLower::current()->caMap()->getConcreteMappedID(
              id, IdMappingMode::LOOP);

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
                   UnaryOpType::Set) ||
              tv->definition()->isA<LoadStoreOp>(),
          "Vectorized accesses cannot be inline with computation, they are only supported with a Set operation.",
          "TensorView: ",
          tv);
    }
    // Validate the vectorized domain maps to the innermost domain of
    // tv. Note that we don't need to validate its producer tv as
    // both Vectorize and MisalignedVectorize can only be used with
    // UnaryOp::Set.
    if (has_vectorize_dim || has_misaligned_vectorize_dim) {
      VectorizeValidator::validate(tv);
    }
  }
}

namespace {

void fillVectorizedContigRootDomains(
    const TensorView* tv,
    const ContigIDs& contig_finder,
    IterDomain* vectorized_root_id,
    VectorizedSetInfo& info) {
  const auto& root_dom = tv->getMaybeRFactorDomain();

  // Find the root domains that are dependency of the merged contig
  // domain.

  auto consumer_indexed_it =
      contig_finder.rootToIndexedID().find(vectorized_root_id);
  TORCH_INTERNAL_ASSERT(
      consumer_indexed_it != contig_finder.rootToIndexedID().end(),
      "Contiguity information not found for root domain: ",
      vectorized_root_id->toString());
  auto consumer_indexed_id = consumer_indexed_it->second;

  // Actual indexed root domains for this root domain. If
  // contig merge is done, multiple root domains are included.
  std::unordered_set<IterDomain*> indexed_root_ids;

  if (consumer_indexed_id == vectorized_root_id) {
    // Indexed domain is equal to the root domain, meaning no contig
    // merge is involved.
    indexed_root_ids.insert(vectorized_root_id);
  } else {
    auto consumer_within_contig_it =
        contig_finder.withinContigIDs().find(consumer_indexed_id);
    TORCH_INTERNAL_ASSERT(
        consumer_within_contig_it != contig_finder.withinContigIDs().end());
    const auto& within_ids = consumer_within_contig_it->second;
    std::copy_if(
        root_dom.begin(),
        root_dom.end(),
        std::inserter(indexed_root_ids, indexed_root_ids.end()),
        [&](IterDomain* root_id) {
          return within_ids.find(root_id) != within_ids.end();
        });
  }

  // Store the contig merged root domains. If it is already set, pick
  // the smaller one as it is used for validating divisibility of the
  // merged extent.
  if (info.contig_root_ids.empty() ||
      indexed_root_ids.size() < info.contig_root_ids.size()) {
    info.contig_root_ids = indexed_root_ids;
  }
}

} // namespace

void fillConsumerVectorizedContigRootDomains(
    const TensorView* consumer_tv,
    const ContigIDs& contig_finder) {
  auto& info_vector = GpuLower::current()->vectorizedSetInfo();
  auto it = std::find_if(
      info_vector.begin(), info_vector.end(), [&consumer_tv](auto& info) {
        return info.consumer_tv == consumer_tv;
      });
  if (it == info_vector.end()) {
    return;
  }

  VectorizedSetInfo& info = *it;

  // info.vectorized_root_id is validated at this point to be the
  // last concrete root domain in consumer.
  auto consumer_root_id = info.vectorized_root_id;

  fillVectorizedContigRootDomains(
      consumer_tv, contig_finder, consumer_root_id, info);
}

void fillProducerVectorizedContigRootDomains(
    const TensorView* producer_tv,
    const TensorView* consumer_tv,
    const std::unordered_map<IterDomain*, IterDomain*>& c2p_map,
    const ContigIDs& contig_finder) {
  auto& info_vector = GpuLower::current()->vectorizedSetInfo();
  auto it = std::find_if(
      info_vector.begin(),
      info_vector.end(),
      [&producer_tv, &consumer_tv](auto& info) {
        return info.consumer_tv == consumer_tv &&
            info.producer_tv == producer_tv;
      });
  if (it == info_vector.end()) {
    return;
  }

  VectorizedSetInfo& info = *it;

  // info.vectorized_root_id is validated at this point to be the
  // last concrete root domain in consumer.
  auto consumer_root_id = info.vectorized_root_id;

  auto root_id = c2p_map.at(consumer_root_id);

  fillVectorizedContigRootDomains(producer_tv, contig_finder, root_id, info);
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
  auto exprs = StmtSort::getExprs(fusion);

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
               {consumer_start_offset->as<int64_t>(),
                consumer_stop_offset->as<int64_t>()}});
        } else {
          // When the range of this root domain is already set, it
          // must be set by its consumers. Make sure the required
          // range by the consumers is covered by the defined range of
          // this root domain.
          auto& consumer_range = it->second;
          TORCH_INTERNAL_ASSERT(
              consumer_start_offset->as<int64_t>() <= consumer_range.first);
          TORCH_INTERNAL_ASSERT(
              consumer_stop_offset->as<int64_t>() <= consumer_range.second);
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
    auto exprs = StmtSort::getExprs(
        tv->fusion(),
        {tv->domain()->domain().begin(), tv->domain()->domain().end()});
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

namespace {

//! Utility to make sure targeted gpu capability is
//!  higher than provided major.minor.
void validateMinimumArch(int major, int minor) {
  // Skip checking arch if disabled.
  if (isOptionDisabled(DisableOption::ArchCheck)) {
    return;
  }

  auto prop = at::cuda::getCurrentDeviceProperties();
  TORCH_INTERNAL_ASSERT(prop->major >= major);
  if (prop->major == major) {
    TORCH_INTERNAL_ASSERT(prop->minor >= minor);
  }
}

//! Validates that the operand and result tensors
//!  of mma ops are swizzled and also validates
//!  specialization of tidx as lane id.
void validateMmaTensors(MmaOp* mma) {
  bool tidx_validated = false;
  std::vector<TensorView*> to_validate = {
      mma->inA()->as<TensorView>(),
      mma->inB()->as<TensorView>(),
      mma->out()->as<TensorView>()};

  for (auto tv : to_validate) {
    for (auto id : tv->domain()->domain()) {
      auto ptype = id->getParallelType();
      if (ptype == ParallelType::TIDx) {
        TORCH_INTERNAL_ASSERT(
            id->isMmaSwizzled(),
            "TIDx for mma input/output must be set by WarpMmaSwizzler",
            id,
            tv);
        if (!tidx_validated) {
          // Check that TIDx is exact lane_id
          const auto& paralel_dim_map =
              GpuLower::current()->parallelDimensionMap();
          TORCH_INTERNAL_ASSERT(
              paralel_dim_map.isExact(ptype) &&
                  paralel_dim_map.get(ptype)->isConstInt() &&
                  paralel_dim_map.get(ptype)->evaluateInt() ==
                      at::cuda::warp_size(),
              "TIDx is reserved for lane id in mma kernels, and it needs to be exactly a warp");
          tidx_validated = true;
        }
      }
    }
  }

  // Note: this check will be relaxed in a follow up.
  auto validate_operand = [](const TensorView* tv) {
    TORCH_INTERNAL_ASSERT(
        tv->getMemoryType() == MemoryType::Local,
        "Only supporting register input for mma ops, up to sm80 all mma ops have to take register inputs.");

    TORCH_INTERNAL_ASSERT(
        std::all_of(
            tv->domain()->domain().begin() + tv->getComputeAtPosition(),
            tv->domain()->domain().end(),
            [](IterDomain* id) {
              return id->isMmaSwizzled() ||
                  // MMA instructions can only take inputs from registers,
                  //  so we always assume mma op inputs are located on
                  //  registers.
                  // Currently requiring that serial ids on the right of the
                  //  CA axis are constant sized to ensure early detection of
                  //  invalid mma schedules.
                  ((id->isBroadcast() || id->extent()->isConstInt()) &&
                   id->getParallelType() == ParallelType::Serial);
            }),
        "All id's on the right of CA pos needs to be mma-swizzled by WarpMmaSwizzler\n",
        tv);
  };

  validate_operand(mma->inA()->as<TensorView>());
  validate_operand(mma->inB()->as<TensorView>());

  // Additionally validate that mma is not directly taking a double buffered
  //  register input as the double buffer indexing is currently not compatible
  //  with fragment iteration. Would need to require a cache stage in this case.
  TORCH_INTERNAL_ASSERT(
      !mma->inA()->as<TensorView>()->isDoubleBuffered(),
      "MMA op cannot directly take double buffered register input, put a set stage before.");
  TORCH_INTERNAL_ASSERT(
      !mma->inB()->as<TensorView>()->isDoubleBuffered(),
      "MMA op cannot directly take double buffered register input, put a set stage before.");
}

//! Note and TODO:
//!   Currently relying on ldmatrix to
//!     obtain the correct data layout for turing/ampere
//!     mma's.
//!   This restriction will eventually not
//!    be necessary once the scatter swizzle is ready.
void validateTuringMmaInput(TensorView* tv) {
  // Pattern matching here to make sure LDMatrix is the right format.
  //  Format is done through swizzling in the scheduling and
  //  we check that swizzling to make sure it's correctly setup for LDMatrix.
  //  We could in theory support patterns LDMatrix doesn't support,
  //  but that would also mean the MMA isn't supported and
  //  so we would have to lower to something completely different.

  // MemCpy async is a more generic utility that we can use.
  // Currently only allowed input paths are:
  //  ldmatrix -> mma or
  //  ldmatrix -> broadcast -> mma
  // We actually wouldn't want too much flexibility here since
  //  this path is very perf critical. But the check itself
  //  can be made cleaner once we have the correct swizzle
  //  labeling.
  // The most generic support would involve build out to
  //  support any pointwise ops that does not change the
  //  datalayout.
  auto tv_def = tv->definition();
  TORCH_INTERNAL_ASSERT(tv_def);
  if (tv_def->isA<BroadcastOp>()) {
    tv_def = tv_def->input(0)->definition();
  }
  TORCH_INTERNAL_ASSERT(tv_def);
  TORCH_INTERNAL_ASSERT(ir_utils::isLdMatrixOp(tv_def));
}

// Output of ldmatrix is swizzled with the mma format, so it
//  currently should not be fused with any pointwise ops. This
//  check is to protect against these cases.
// This would also not be needed once scatter swizzle ready, should
//  just become a swizzle format check if we wanted to fuse ldmatrix
//  with any op other than mma.
void validateLdMatrixOutput(TensorView* tv) {
  const auto& out_uses = tv->fusion()->unordered_uses(tv);
  if (out_uses.empty()) {
    return;
  }
  // TODO: restricting to single use pipelines for now which
  //  is true to matmul mainloop. This Could be relaxed to
  //  support more complex mma usage.
  TORCH_INTERNAL_ASSERT(out_uses.size() == 1);
  auto out_use = *(out_uses.begin());

  if (out_use->isA<BroadcastOp>()) {
    validateLdMatrixOutput(out_use->output(0)->as<TensorView>());
    return;
  }

  TORCH_INTERNAL_ASSERT(
      out_use->isA<MmaOp>(),
      "validateLdMatrixOutput: currently only supports single mma use for ldmatrix",
      out_use);
}

// Checks that the memory ops are supported on the targeted GPU
void validateArchMemoryOp(LoadStoreOp* ldst) {
  switch (ldst->opType()) {
    case LoadStoreOpType::LdMatrix:
    case LoadStoreOpType::LdMatrixTranspose:
      validateMinimumArch(7, 5);
      validateLdMatrixOutput(ldst->out()->as<TensorView>());
      return;
    case LoadStoreOpType::CpAsync:
      validateMinimumArch(8, 0);
      return;
    default:
      return;
  }
}

} // namespace

//! Validate data format and GPU arch compatibility of scheduled
//!  mma operators on the fusion.
void validateMma(Fusion* fusion) {
  auto exprs = StmtSort::getExprs(fusion);

  for (auto expr : exprs) {
    if (auto mma = dynamic_cast<MmaOp*>(expr)) {
      validateMmaTensors(mma);

      switch (mma->options().macro) {
        case MmaOptions::MacroType::Volta_16_16_4:
          validateMinimumArch(7, 0);
          break;
        case MmaOptions::MacroType::Turing_16_8_16:
        case MmaOptions::MacroType::Turing_16_16_16:
          validateMinimumArch(7, 5);

          // Check that operands come from ldmatrix, can be
          //  relaxed once swizzles can be labeled on iterdomains.
          validateTuringMmaInput(mma->inA()->as<TensorView>());
          validateTuringMmaInput(mma->inB()->as<TensorView>());
          break;
        case MmaOptions::MacroType::Ampere_16_8_16:
        case MmaOptions::MacroType::Ampere_16_16_16:
          validateMinimumArch(8, 0);

          // Check that operands come from ldmatrix, can be
          //  relaxed once swizzles can be labeled on iterdomains.
          validateTuringMmaInput(mma->inA()->as<TensorView>());
          validateTuringMmaInput(mma->inB()->as<TensorView>());
          break;
        default:
          TORCH_INTERNAL_ASSERT(false, "validate mma: unsupported macro");
          break;
      }
    }
    if (auto ldst = dynamic_cast<LoadStoreOp*>(expr)) {
      validateArchMemoryOp(ldst);
    }
  }
}

namespace {

// Utility function to validate a loop swizzle:
//  1. Throws an error if any output of the swizzle is not in leaf_domain set.
//  2. Warns if any output of the swizzle is not the concrete id of the loop
//  map.
// The second case would make the codegen ignore this swizzle, as if it was not
// there at all.
void validateLoopSwizzle(
    Expr* swizzle_expr,
    std::unordered_set<IterDomain*>& leaf_domains) {
  for (auto out_id :
       ir_utils::filterByType<IterDomain>(swizzle_expr->outputs())) {
    TORCH_INTERNAL_ASSERT(
        leaf_domains.count(out_id),
        "Loop swizzle can only be direct producer of leaf domains.");
    if (GpuLower::current()->caMap()->getConcreteMappedID(
            out_id, IdMappingMode::LOOP) != out_id) {
      TORCH_WARN_ONCE("Ignored loop swizzle :", swizzle_expr->toString());
    }
  }
}

} // namespace

void validateSwizzle(Fusion* fusion) {
  auto used_vals = fusion->usedMathVals();
  for (auto tv : ir_utils::filterByType<TensorView>(used_vals)) {
    if (tv->hasSwizzleOp()) {
      std::unordered_set<IterDomain*> tv_leaf_domain_set(
          tv->domain()->domain().begin(), tv->domain()->domain().end());

      // Make sure no swizzle op is inlined:
      auto inlined_swizzles = ir_utils::getAllSwizzlesBetween(
          tv->getMaybeRFactorDomain(),
          {tv->domain()->domain().begin(),
           tv->domain()->domain().begin() + tv->getComputeAtPosition()});

      auto not_inlined_swizzles = ir_utils::getAllSwizzlesBetween(
          tv->getMaybeRFactorDomain(),
          {tv->domain()->domain().begin() + tv->getComputeAtPosition(),
           tv->domain()->domain().end()});

      // Check inlined swizzles: only loop swizzles can be inlined currently
      //  as inlining data swizzles would require addtional support of unswizzle
      //  operator, which currently doesn't have important use cases.
      for (auto swizzle_expr : inlined_swizzles) {
        TORCH_INTERNAL_ASSERT(
            swizzle_expr->as<Swizzle2D>()->swizzleMode() == SwizzleMode::Loop,
            "Only support inlining loop swizzles");
        validateLoopSwizzle(swizzle_expr, tv_leaf_domain_set);
      }

      std::unordered_set<Expr*> inlined_swizzle_set(
          inlined_swizzles.begin(), inlined_swizzles.end());

      // Check not inlined swizzles:
      //  Apply the loop swizzle check when it applies, and
      // also make sure that the no swizzle is also in inlined_swizzle set.
      // The latter would mean that one output of the swizzle is inlined while
      //  the other is not. Such case will not be supported.
      for (auto swizzle_expr : not_inlined_swizzles) {
        TORCH_INTERNAL_ASSERT(
            !inlined_swizzle_set.count(swizzle_expr),
            "Cannot partially inline across swizzle domains.",
            swizzle_expr->toString());
        if (swizzle_expr->as<Swizzle2D>()->swizzleMode() == SwizzleMode::Loop) {
          validateLoopSwizzle(swizzle_expr, tv_leaf_domain_set);
        }
      }
    }
  }
}

void validateAndConvertIterDomainGrouping(Fusion* fusion) {
  for (auto tv : ir_utils::allTvs(fusion)) {
    bool is_grouped = false;
    for (const auto id_idx : c10::irange(tv->nDims())) {
      const auto id = tv->axis(id_idx);
      auto ptype = GpuLower::current()
                       ->caMap()
                       ->getConcreteMappedID(id, IdMappingMode::LOOP)
                       ->getParallelType();
      if (ptype != ParallelType::Group) {
        // Not a grouped ID
        continue;
      }

      // Remember if a grouped ID is found
      is_grouped = true;

      // Grouping only makes sense for the normal iteration type
      TORCH_CHECK(
          id->getIterType() == IterType::Iteration,
          "Invalid use of ParallelType::Group.",
          " Grouping of ",
          id->getIterType(),
          " is not allowed. ",
          tv->toString());

      // Extent must be static
      TORCH_CHECK(
          id->extent()->getInt().has_value(),
          "Invalid use of ParallelType::Group.",
          " IterDomain must have a static extent: ",
          id->toString());

      // The CA position must be left of any grouped ID
      TORCH_CHECK(
          tv->getComputeAtPosition() <= id_idx,
          "Invalid use of ParallelType::Group.",
          " ComputeAt position must be left of grouped IDs: ",
          tv->toString());

      // Similarly, the produce-at position must be left of any grouped ID
      TORCH_CHECK(
          tv->getMaxProducerPosition() <= id_idx,
          "Invalid use of ParallelType::Group.",
          " ProduceAt position must be left of grouped IDs: ",
          tv->toString());

      // Halo is not allowed
      TORCH_CHECK(
          GpuLower::current()->haloInfo()->getExtent(id) == nullptr,
          "Invalid use of ParallelType::Group.",
          " Grouping of halo-extended IterDomain, ",
          id->toString(),
          ", is not supported. ",
          tv->toString());
    }

    if (!is_grouped) {
      continue;
    }

    // Must be defined by ReductionOp
    auto def = tv->definition();
    TORCH_CHECK(
        def != nullptr,
        "Invalid use of ParallelType::Group.",
        " Definition of tv with ParallelType::Group not found. ",
        tv->toString());

    TORCH_CHECK(
        tv->definition()->isA<ReductionOp>() ||
            tv->definition()->isA<GroupedReductionOp>() ||
            tv->definition()->isA<WelfordOp>() ||
            tv->definition()->isA<GroupedWelfordOp>(),
        "Invalid use of ParallelType::Group. Only ReductionOp, GroupedReductionOp, WelfordOp and GroupedWelfordOp are allowed. ",
        tv->definition()->toString());

    // Convert ReductionOp to GroupedReductionOp
    if (tv->definition()->isA<ReductionOp>()) {
      auto rop = def->as<ReductionOp>();
      auto is_allreduce = rop->isAllreduce();

      TORCH_CHECK(
          is_allreduce,
          "Invalid use of ParallelType::Group.",
          " Only enabled for allreduce reductions: ",
          rop->toString());

      TORCH_CHECK(
          tv->domain()->hasGridReduction(),
          "Invalid use of ParallelType::Group.",
          " Only enabled for grid reductions: ",
          rop->toString());

      std::vector<BinaryOpType> op_types({rop->getReductionOpType()});
      std::vector<Val*> init_vals({rop->init()});
      std::vector<Val*> outputs({rop->out()});
      std::vector<Val*> inputs({rop->in()});

      fusion->removeExpr(rop);
      IrBuilder::create<GroupedReductionOp>(
          static_cast<IrContainer*>(fusion),
          op_types,
          init_vals,
          outputs,
          inputs,
          is_allreduce);
    } else if (tv->definition()->isA<WelfordOp>()) {
      // Convert WelfordOp to GroupedWelfordOp
      auto wop = def->as<WelfordOp>();
      auto is_allreduce = wop->isAllreduce();

      TORCH_CHECK(
          is_allreduce,
          "Invalid use of ParallelType::Group.",
          " Only enabled for allreduce reductions: ",
          wop->toString());

      TORCH_CHECK(
          tv->domain()->hasGridReduction(),
          "Invalid use of ParallelType::Group.",
          " Only enabled for grid reductions: ",
          wop->toString());

      std::vector<WelfordTriplet> output_vals(
          {{wop->outAvg(), wop->outVar(), wop->outN()}});
      std::vector<WelfordTriplet> input_vals(
          {{wop->inAvg(), wop->inVar(), wop->inN()}});
      std::vector<WelfordTriplet> init_vals(
          {{wop->initAvg(), wop->initVar(), wop->initN()}});
      fusion->removeExpr(wop);
      IrBuilder::create<GroupedWelfordOp>(
          static_cast<IrContainer*>(fusion),
          output_vals,
          input_vals,
          init_vals,
          is_allreduce);
    }
  }
}

void validateGroupedReductions(Fusion* fusion) {
  for (auto expr : StmtSort::getExprs(fusion)) {
    if (auto grouped_reduction_op = dynamic_cast<GroupedReductionOp*>(expr)) {
      const auto num_exprs = grouped_reduction_op->numExprs();
      int num_grouped_iterations = 1;
      auto out_tv = ir_utils::getTvOutput(grouped_reduction_op);
      for (auto axis : out_tv->domain()->domain()) {
        if (axis->getParallelType() == ParallelType::Group) {
          num_grouped_iterations *= axis->extent()->getInt().value();
        }
      }
      TORCH_CHECK(
          num_exprs * num_grouped_iterations <= kMaxNumGroupedReductions,
          "Too many grouped reductions: ",
          grouped_reduction_op->toString(),
          ". Up to ",
          kMaxNumGroupedReductions,
          " reductions are allowed.");
    }
  }
}

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
