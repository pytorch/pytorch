
#include <torch/csrc/jit/codegen/cuda/instrumentation.h>
#include <torch/csrc/jit/codegen/cuda/ir_utils.h>
#include <torch/csrc/jit/codegen/cuda/lower2device.h>
#include <torch/csrc/jit/codegen/cuda/lower_index_compute.h>

#include <torch/csrc/jit/codegen/cuda/lower_sync_information.h>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

namespace {

// Validate parallelization of a single tensor
void validateParallelizationOfTensor(TensorView* tv) {
  // Each ParallelType can be used only once.
  ParallelTypeBitmap pt_map;
  for (size_t i = 0; i < tv->nDims(); ++i) {
    auto axis = tv->axis(i);
    auto ptype = axis->getParallelType();
    if (!isParallelTypeThread(ptype)) {
      continue;
    }

    // It doesn't matter if this axis is a non-concretized broadcast
    // TODO: merging broadcast and non-broadcast
    if (axis->isBroadcast() &&
        !GpuLower::current()->concretizedBroadcastDomains()->isConcretized(
            axis)) {
      continue;
    }

    TORCH_INTERNAL_ASSERT(
        !pt_map.get(ptype),
        "Multiple use of ",
        ptype,
        " in tensor t",
        tv->name(),
        ": ",
        tv);
    pt_map.set(ptype);
  }

  // If this tensor is predicated by a paralel type, it should not be
  // used to parallelize any domain of this tensor

  const auto thread_pred =
      GpuLower::current()->threadPredMap().getPredicateInfo(tv);

  auto predicated_parallel_types = pt_map & thread_pred.limited_types;

  TORCH_INTERNAL_ASSERT(
      predicated_parallel_types.none(),
      "Invalid parallelization of tensor t",
      tv->name(),
      ". The tensor is parallelized with ",
      predicated_parallel_types.toString(),
      ", but it's invalid to use the types as the tensor is also predicated with them.",
      ", thread pred: ",
      thread_pred.limited_types.toString());
}

//! Properties used in useSameIndex that only depends on the producer and
//! consumer tensors and can be reused for validating different pairs
//! of their leaf IDs. Works as caching as some properties can be
//! expensive to compute.
struct ProducerConsumerIndexingInfoCache {
 public:
  ProducerConsumerIndexingInfoCache(
      TensorView* producer_tv,
      TensorView* consumer_tv)
      : producer_tv_(producer_tv), consumer_tv_(consumer_tv) {}

  const std::vector<IterDomain*>& getConsumerLeafIDsSharedWithProducer() {
    if (!consumer_leaf_ids_shared_with_producer_.has_value()) {
      const auto& ca_map = *(GpuLower::current()->caMap());
      std::vector<IterDomain*> consumer_leaf_ids_shared_with_producer;
      std::copy_if(
          consumer_tv_->domain()->domain().begin(),
          consumer_tv_->domain()->domain().end(),
          std::back_inserter(consumer_leaf_ids_shared_with_producer),
          [&](auto consumer_leaf_id) {
            return std::find_if(
                       producer_tv_->domain()->domain().begin(),
                       producer_tv_->domain()->domain().end(),
                       [&](auto producer_leaf_id) {
                         return ca_map.areMapped(
                             producer_leaf_id,
                             consumer_leaf_id,
                             IdMappingMode::LOOP);
                       }) != producer_tv_->domain()->domain().end();
          });
      consumer_leaf_ids_shared_with_producer_ =
          std::move(consumer_leaf_ids_shared_with_producer);
    }
    return *consumer_leaf_ids_shared_with_producer_;
  }

  const std::vector<Val*>& getConsumerRootIDsSharedWithProducer() {
    if (!consumer_root_ids_shared_with_producer_.has_value()) {
      const auto& consumer_leaf_ids_shared_with_producer =
          getConsumerLeafIDsSharedWithProducer();
      consumer_root_ids_shared_with_producer_ = InputsOf::outputs(
          producer_tv_->fusion(),
          {consumer_leaf_ids_shared_with_producer.begin(),
           consumer_leaf_ids_shared_with_producer.end()});
    }
    return *consumer_root_ids_shared_with_producer_;
  }

  const std::vector<IterDomain*>& getConsumerOnlyPermissiveLeafIds() {
    // When a given ID is the factor of 1 of a split, return the other
    // output. Return nullptr otherwise.
    auto get_split1_other_out = [](IterDomain* id) -> IterDomain* {
      if (id->extent()->isOneInt() && id->definition() != nullptr &&
          id->definition()->isA<Split>()) {
        auto split = id->definition()->as<Split>();
        if (split->innerSplit() && split->inner() == id) {
          return split->outer();
        } else if (!split->innerSplit() && split->outer() == id) {
          return split->inner();
        }
      }
      return nullptr;
    };

    if (!consumer_only_permissive_leaf_ids_.has_value()) {
      // consumer_only_permissive_leaf_ids_ = {};
      std::vector<IterDomain*> consumer_only_permissive_leaf_ids;
      const auto& ca_map = *(GpuLower::current()->caMap());
      std::copy_if(
          consumer_tv_->domain()->domain().begin(),
          consumer_tv_->domain()->domain().end(),
          std::back_inserter(consumer_only_permissive_leaf_ids),
          [&](IterDomain* consumer_leaf_id) {
            const auto& consumer_leaf_ids_shared_with_producer =
                getConsumerLeafIDsSharedWithProducer();
            if (std::find(
                    consumer_leaf_ids_shared_with_producer.begin(),
                    consumer_leaf_ids_shared_with_producer.end(),
                    consumer_leaf_id) !=
                consumer_leaf_ids_shared_with_producer.end()) {
              return false;
            }

            auto loop_concrete_id = ca_map.getConcreteMappedID(
                consumer_leaf_id, IdMappingMode::LOOP);

            // If the loop concrete ID has the same info as the
            // consumer leaf ID, indexing shouldn't be affected by the
            // loop concrete ID
            if (ca_map.areMapped(
                    consumer_leaf_id,
                    loop_concrete_id,
                    IdMappingMode::ALMOSTEXACT)) {
              return false;
            }

            // Note that the factor output domain of split-by-one is
            // not mapped in the almost exact map. As long as the
            // other domains are almost-exactly mapped, this shouldn't
            // affect the indexing neither.
            auto consumer_split1_other = get_split1_other_out(consumer_leaf_id);
            auto loop_concrete_split1_other =
                get_split1_other_out(loop_concrete_id);

            if (consumer_split1_other != nullptr &&
                loop_concrete_split1_other != nullptr &&
                ca_map.areMapped(
                    consumer_split1_other,
                    loop_concrete_split1_other,
                    IdMappingMode::ALMOSTEXACT)) {
              return false;
            }

            return true;
          });
      consumer_only_permissive_leaf_ids_ =
          std::move(consumer_only_permissive_leaf_ids);
    }
    return *consumer_only_permissive_leaf_ids_;
  }

  const VectorOfUniqueEntries<IterDomain*>& getConsumerLoopIndexingIDs() {
    if (!consumer_loop_indexing_ids_.has_value()) {
      consumer_loop_indexing_ids_ =
          LoopIndexingAnalysis::getReplayableConcreteIDs(
              getConsumerOnlyPermissiveLeafIds(), consumer_tv_);
    }
    return *consumer_loop_indexing_ids_;
  }

 private:
  TensorView* producer_tv_ = nullptr;
  TensorView* consumer_tv_ = nullptr;
  // Consumer leaf IDs that are also used to index the producer, i.e.,
  // those that are loop-mapped with the producer leaf IDs
  c10::optional<std::vector<IterDomain*>>
      consumer_leaf_ids_shared_with_producer_;
  // Root IDs of the shared leaf IDs
  c10::optional<std::vector<Val*>> consumer_root_ids_shared_with_producer_;
  // Consumer CA leaf IDs that are not shared with producer and
  // permissively mapped with consumers of the consumer
  c10::optional<std::vector<IterDomain*>> consumer_only_permissive_leaf_ids_;
  // IDs whose index depends on consumer_only_permissive_leaf_ids_
  c10::optional<VectorOfUniqueEntries<IterDomain*>> consumer_loop_indexing_ids_;
};

// For a given pair of a producer and consumer leaf ID, check if the
// root domains that have dependencies with them are guaranteed to
// have the same index.
//
// The algorithm first sees if the root domains reachable from the
// consumer domain are all exactly mapped with the root domains
// reachable from the producer domain. This is to detect merged
// broadcast domains that only show up in the consumer. If such a
// consumer-only root domain is found, it can mean the producer and
// consumer are indexed differently, but not always. If there's a
// consumer leaf ID that is shared with the producer through
// computeAt, and if there's a dependency from the leaf ID to the
// consumer-only root ID, the producer indexing also uses the shared
// consumer ID and the indexing traversal reach at the consumer-only
// broadcast root domain, generating the same index as that of the
// consumer.
//
// It is also necessary to check non-CA-shared consumer leaf IDs that
// are permissively mapped with its consumers. See inline comments
// below.
bool useSameIndex(
    TensorView* producer_tv,
    IterDomain* producer_id,
    TensorView* consumer_tv,
    IterDomain* consumer_id,
    ProducerConsumerIndexingInfoCache& indexing_info) {
  const auto& ca_map = *(GpuLower::current()->caMap());

  // At least, they must be mapped exactly or permissively
  if (!ca_map.areMapped(producer_id, consumer_id, IdMappingMode::EXACT) &&
      !ca_map.areMapped(producer_id, consumer_id, IdMappingMode::PERMISSIVE)) {
    return false;
  }

  // If the producer ID is left of the CA position, the indexing is
  // done with the corresponding consumer ID
  auto producer_id_pos = std::distance(
      producer_tv->domain()->domain().begin(),
      std::find(
          producer_tv->domain()->domain().begin(),
          producer_tv->domain()->domain().end(),
          producer_id));
  if (producer_id_pos < producer_tv->getComputeAtPosition()) {
    return true;
  }

  // Grab all consumer root IDs that have the threading index of
  // consumer_id. The goal of the analysis below is to find out if all
  // of the root IDs are indexed in the same way between the producer
  // and consumer tensors.
  auto consumer_root_ids = InputsOf::output(consumer_id->fusion(), consumer_id);

  auto producer_root_vals = StmtSort::getStmtsBetween(
      producer_id->fusion(),
      {producer_tv->getMaybeRFactorDomain().begin(),
       producer_tv->getMaybeRFactorDomain().end()},
      {producer_id});
  auto producer_root_ids =
      ir_utils::filterByType<IterDomain>(producer_root_vals);

  // For each of the root IDs that consumer_id is dependent on, check
  // if the producer uses the same indexing as the consumer. This
  // requires that the producer has a root ID that is exactly mapped with
  // the consumer root ID. Another case is when the consumer root ID
  // has a dependency with any of the leaf consumer IDs that are
  // shared with the producer. In that case, the producer uses those
  // shared consumer leaf IDs to index the root ID and thus uses the same index
  if (!std::all_of(
          ir_utils::filterByType<IterDomain>(consumer_root_ids).begin(),
          ir_utils::filterByType<IterDomain>(consumer_root_ids).end(),
          [&](IterDomain* consumer_root_id) {
            return std::find_if(
                       producer_root_ids.begin(),
                       producer_root_ids.end(),
                       [&](IterDomain* producer_root_id) {
                         return ca_map.areMapped(
                             producer_root_id,
                             consumer_root_id,
                             IdMappingMode::EXACT);
                       }) != producer_root_ids.end() ||
                std::find(
                    indexing_info.getConsumerRootIDsSharedWithProducer()
                        .begin(),
                    indexing_info.getConsumerRootIDsSharedWithProducer().end(),
                    consumer_root_id) !=
                indexing_info.getConsumerRootIDsSharedWithProducer().end();
          })) {
    return false;
  }

  // At this point, consumer_root_ids is the set of root IDs that
  // commonly have dependencies with producer_id and consumer_id.
  //
  // It is also necessary to look at consumer leaf IDs that are
  // computed-at its consumers, which means the consumer is indexed
  // using its consumer domains. Unless such IDs are also shared with the
  // producer, the consumer may have a different index as that of the
  // producer.

  // Example:
  // t0: [I0], t1: [I0, I1]
  // t2 = t0
  // t3 = broadcast(t2, {true, false})
  // t4 = t3 + t1
  //
  // t0: [I0]
  // t1: [I0, I1]
  // t2: [I0]
  // t3: [I0, B0]
  // t4: [I0, I1]
  //
  // t4->merge(0)->split(0, 4)
  // propagate t4 transformations
  // parallelize axis(-1) with tidx
  //
  // t0: [I0/4, tidx(4)]
  // t1: [I0*I1/4, tidx(4)]
  // t2: [I0/4, tidx(4)]
  // t3: [I0*B0/4, tidx(4)]
  // t4: [I0*I1/4, tidx(4)]
  //
  // t2->computeAt(t4, 1)
  //
  // t0: [I0/4, tidx(4)]
  // t1: [I0*I1/4, tidx(4)]
  // t2: [I0/4, tidx(4)] ca(1)
  // t3: [I0*B0/4, tidx(4)] ca(1)
  // t4: [I0*I1/4, tidx(4)] produce(1)
  //
  // The interesting part here is t0 and t2. They are completely
  // exactly mapped, but the CA of t2 makes it indexed based on its
  // consumer, t4. Specifically, the code would look like:
  //
  // for (i: I0/4)
  //   t0[i * bdimx + tidx] = ...
  // for (i: I0*I1/4)
  //   t2[(i * bdimx + tidx) % bdimx] = t0[...]
  //   t3[(i * bdimx + tidx) % bdimx] = t2[...]
  //   t4[i * bdimx + tidx] = t3[...] + t1[...]
  //
  // t2->axis(0) is an example of consumer-only leaf IDs that are
  // permissively mapped with consumers of consumers. Since it's
  // effectively replaced with t4->axis(0) when indexing t2, whereas
  // t0 is independently indexed, t0 must be placed on shared memory
  // (or global memory) with a RAW sync. See See FusionValidateParallelize10.
  //
  // For the same original fusion, consider this transformation:
  //
  // t4->merge(0)->split(0, 4)->split->(0, 2)
  // propagate t4 transformations
  // parallelize axis(-1) with tidx
  //
  // t0: [I0/4/2, 2, tidx(4)]
  // t1: [I0*I1/4/2, 2, tidx(4)]
  // t2: [I0/4/2, 2, tidx(4)]
  // t3: [I0*B0/4/2, 2, tidx(4)]
  // t4: [I0*I1/4/2, 2, tidx(4)]
  //
  // t0->computeAt(t4, 1)
  //
  // t0: [I0/4/2, 2, tidx(4)] ca(1)
  // t1: [I0*I1/4/2, 2, tidx(4)]
  // t2: [I0/4/2, 2, tidx(4)] ca(1)
  // t3: [I0*B0/4/2, 2, tidx(4)]
  // t4: [I0*I1/4/2, 2, tidx(4)] produce(1)
  //
  // For t1 and t2, t2->axis(1) is again a consumer-only leaf ID
  // permissively mapped with its consumer. However, in this case, t0
  // also shares the first leaf ID with t2 and t4, making it indexed
  // using t4.
  //
  // for (i: I0*I1/4/2)
  //   for (j: 2)
  //     t0[((i * 2 + j) * bdimx + tidx) % bdimx] = ...
  //   for (j: 2)
  //     t2[((i * 2 + j) * bdimx + tidx) % bdimx] = t0[...]
  //   for (j: 2)
  //     t3[((i * 2 + j) * bdimx + tidx) % bdimx] = t2[...]
  //   for (j: 2)
  //     t4[(i * 2 + j) * bdimx + tidx] = t3[...] + t1[...]
  //
  // All of the tensors are indexed consistently, so no RAW sync is
  // required in this case. See FusionValidateParallelize11.

  // If there's no consumer-only leaf ID that is permissively mapped
  // with its consumers, this pair of producer and consumer indices
  // should be used in the same way
  if (indexing_info.getConsumerOnlyPermissiveLeafIds().empty()) {
    return true;
  }

  return std::all_of(
      ir_utils::filterByType<IterDomain>(consumer_root_ids).begin(),
      ir_utils::filterByType<IterDomain>(consumer_root_ids).end(),
      [&](IterDomain* consumer_root_id) {
        // If the consumer root ID is part of the shared root IDs
        // with the producer, it is guaranteed to be indexed in
        // the same way. See the second example above.
        if (std::find(
                indexing_info.getConsumerRootIDsSharedWithProducer().begin(),
                indexing_info.getConsumerRootIDsSharedWithProducer().end(),
                consumer_root_id) !=
            indexing_info.getConsumerRootIDsSharedWithProducer().end()) {
          return true;
        }

        // Check if the consumer root ID has a dependency with any
        // of the consumer-only leaf IDs. If so, its index may be
        // different from the producer. The dependency here means
        // the indexing traversal from the LOOP concrete domains of the
        // leaf IDs. It's not just enough to do normal backward
        // travesal from the concrete domains as they may come from
        // post-view tensors.
        return !indexing_info.getConsumerLoopIndexingIDs().has(
            ca_map.getConcreteMappedID(consumer_root_id, IdMappingMode::EXACT));
      });
}

} // namespace

SyncMap::SyncMap(Fusion* fusion) {
  FUSER_PERF_SCOPE("SyncMap::SyncMap");
  FusionGuard fg(fusion);

  const auto& ca_map = GpuLower::current()->caMap();
  const auto& pred_map = GpuLower::current()->threadPredMap();

  auto exprs = StmtSort::getExprs(fusion);

  // Run through expressions and check for communication across threads/blocks
  // occuring from producer to consumer of the expression
  for (auto expr : exprs) {
    if (!ir_utils::isTvOp(expr)) {
      continue;
    }

    // Validate parallelization of each consumer by itself
    for (auto consumer : ir_utils::filterByType<TensorView>(expr->outputs())) {
      validateParallelizationOfTensor(consumer);
    }

    // It's probably enough to just check all producers to one consumer as
    // multi-consumers are guaranteed to be transformed/parallelized the same,
    // but to be conservative for now checking every producer <-> consumer
    // relationship.
    for (auto producer : ir_utils::filterByType<TensorView>(expr->inputs())) {
      // Parallelization on input tensors have no effect.
      if (producer->isFusionInput()) {
        continue;
      }

      ParallelTypeBitmap raw_dims;

      const auto parallel_bcast_doms =
          pred_map.getParallelBroadcastDomains(producer);

      // Stash information about parallelized producer iteration domains
      std::vector<IterDomain*> producer_parallel_ids(
          ParallelTypeBitmap::kNumParallelTypes, nullptr);
      ParallelTypeBitmap producer_parallel_bitmap;

      // Get the parallel types that producer will be predicated off in producer
      // writes.
      //  In this case we need a sync whether the producer-consumer axes are
      //  mapped or not since the predicate pass will generate pattern like
      //  below to eliminate redundant writes: if(threadIdx.x == 0)
      //    shared[threadIdx.x + i] = ...
      // We will need a raw sync after this pattern for correctness.
      auto producer_redundant_types = GpuLower::current()
                                          ->threadPredMap()
                                          .getPredicateInfo(producer)
                                          .redundant_types;
      // Get the parallel types that are inactive in consumer's use chains.
      auto producer_redundant_use_types = GpuLower::current()
                                              ->threadPredMap()
                                              .getPredicateInfo(producer)
                                              .redundant_use_types;

      // In sync info pass we only consider the parallel types in
      //  producer that are redundantly produced but not redundantly consumed.
      producer_redundant_types =
          producer_redundant_types & (~producer_redundant_use_types);

      for (const auto producer_i : c10::irange(producer->nDims())) {
        auto producer_axis = producer->axis(producer_i);
        auto producer_ptype =
            ca_map->getConcreteMappedID(producer_axis, IdMappingMode::LOOP)
                ->getParallelType();

        if (!isParallelTypeThread(producer_ptype)) {
          continue;
        }

        // Producer reductions shouldn't map to consumers
        if (producer_axis->isReduction()) {
          continue;
        }

        producer_parallel_bitmap.set(producer_ptype);
        producer_parallel_ids[getParallelTypeBitMapOffset(producer_ptype)] =
            producer_axis;
      }

      for (auto consumer :
           ir_utils::filterByType<TensorView>(expr->outputs())) {
        // Stash information about parallelized consumer iteration domains
        std::vector<IterDomain*> consumer_parallel_ids(
            ParallelTypeBitmap::kNumParallelTypes, nullptr);
        ParallelTypeBitmap consumer_parallel_bitmap;
        for (const auto consumer_i : c10::irange(consumer->nDims())) {
          auto consumer_axis = consumer->axis(consumer_i);
          auto consumer_ptype =
              ca_map->getConcreteMappedID(consumer_axis, IdMappingMode::LOOP)
                  ->getParallelType();

          if (!isParallelTypeThread(consumer_ptype)) {
            continue;
          }

          // When the consumer axis is a broadcast, it is not really
          // parallelized unless thread-predicated and eventually concretized
          if (consumer_axis->isBroadcast() &&
              (!parallel_bcast_doms.get(consumer_ptype) ||
               !GpuLower::current()
                    ->concretizedBroadcastDomains()
                    ->isConcretized(consumer_axis))) {
            continue;
          }

          consumer_parallel_bitmap.set(consumer_ptype);
          consumer_parallel_ids[getParallelTypeBitMapOffset(consumer_ptype)] =
              consumer_axis;
        }

        ProducerConsumerIndexingInfoCache indexing_info(producer, consumer);

        // At this point each parallel type that's present in the consumer or
        // the producer will be present in their corresponding `_parallel_ids`
        // map going from parallel index type (only size 6 for grid/block dims)
        // to the iteration domain of that parallel type.
        for (auto parallel_type : kParallelTypeThreads) {
          // TIDx is reserved for lane_id in the case of mma ops.
          //  It is swizzled and handled separately in validateMma.
          if (parallel_type == ParallelType::TIDx && expr->isA<MmaOp>()) {
            continue;
          }

          // In the case when the parallel id's are mapped by ca map,
          //   will additionally need to consider if the producer is
          //   a redundant write. The raw dim can be skipped only if
          //   consumer use chains only contain redundant uses.
          //  TODO:
          //    still losing a bit precision here for expr ordering
          //  sensitive cases, but we could wait until that becomes
          //  a perf limiter to fix.
          if (producer_redundant_types.get(parallel_type)) {
            raw_dims.set(parallel_type);
            continue;
          }

          auto parallel_type_i = getParallelTypeBitMapOffset(parallel_type);

          auto p_id = producer_parallel_ids[parallel_type_i];
          auto c_id = consumer_parallel_ids[parallel_type_i];

          if (p_id == nullptr && c_id == nullptr) {
            continue;
          } else if (p_id != nullptr && c_id != nullptr) {
            if (GpuLower::current()->caMap()->areMapped(
                    p_id, c_id, IdMappingMode::PERMISSIVE)) {
              const auto halo_info = GpuLower::current()->haloInfo();

              if (halo_info->hasHaloWidth(p_id) !=
                      halo_info->hasHaloWidth(c_id) ||
                  (halo_info->hasHaloWidth(p_id) &&
                   halo_info->hasHaloWidth(c_id) &&
                   halo_info->getHaloWidth(p_id) !=
                       halo_info->getHaloWidth(c_id))) {
                raw_dims.set(parallel_type);
                continue;
              }
            }
          } else {
            if (p_id != nullptr) {
              auto it = std::find_if(
                  consumer->domain()->domain().begin(),
                  consumer->domain()->domain().end(),
                  [&](IterDomain* c_id) {
                    return GpuLower::current()->caMap()->areMapped(
                        p_id, c_id, IdMappingMode::PERMISSIVE);
                  });

              // If there isn't a mapping from producer to a consumer domain,
              // need to assume there's communication across this parallel
              // dimension.
              c_id = it == consumer->domain()->domain().end() ? nullptr : *it;
              // i.e. if producer is parallelized across threadIdx.x in a
              // certain split, if the consumer doesn't map to this split,
              // then we need to assume it has to be in smem with proper
              // syncs.
            } else {
              auto it = std::find_if(
                  producer->domain()->domain().begin(),
                  producer->domain()->domain().end(),
                  [&](IterDomain* p_id) {
                    return GpuLower::current()->caMap()->areMapped(
                        p_id, c_id, IdMappingMode::PERMISSIVE);
                  });
              if (it == producer->domain()->domain().end()) {
                // Can't infer anything if producer doesn't have a matching axis
                // to parallel consumer dim.
                continue;
              }
              p_id = *it;
            }
          }

          // Comm pattern options (when parallel types don't have matching
          // axes) and required memory, Chart is producer parallel type,
          // consumer parallel type Parallel types are Serial(S),
          // threadIdx(T), blockIdx(B), Memory required for the producer is
          // Local(L), Shared(S), Global(G), Sync is None (N/A), blockSync(B),
          // grid_sync(G)
          //
          // P    C   Mem Req   Sync Type
          // S    S      L          N/A
          // S    T      L          N/A
          // S    B      L          N/A
          // T    S      S           B
          // T    T      S           B
          // T    B      S           B
          // B    S      G           G
          // B    T      G           G
          // B    B      G           G

          auto producer_ptype =
              ca_map->getConcreteMappedID(p_id, IdMappingMode::LOOP)
                  ->getParallelType();
          auto consumer_ptype = c_id == nullptr
              ? ParallelType::Serial
              : ca_map->getConcreteMappedID(c_id, IdMappingMode::LOOP)
                    ->getParallelType();

          auto producer_parallel_bcast = p_id->isBroadcast() &&
              isParallelTypeThread(producer_ptype) &&
              parallel_bcast_doms.get(producer_ptype) &&
              GpuLower::current()->concretizedBroadcastDomains()->isConcretized(
                  p_id);

          auto producer_parallelized = isParallelTypeThread(producer_ptype) &&
              (!p_id->isBroadcast() || producer_parallel_bcast);

          // Handle special cases first

          // If any leaf id of producer is block or grid parallel and is
          // involved
          //  in any swizzle pattern, track this parallel dim as a communication
          //  dimension that requires the corresponding synchronization and
          //  memory type.
          if (isParallelTypeThread(producer_ptype) &&
              producer->hasSwizzleOp()) {
            if (!ir_utils::getAllSwizzlesBetween(
                     producer->getMaybeRFactorDomain(), {p_id})
                     .empty()) {
              raw_dims.set(producer_ptype);
            }
          }

          // In shift or gather operations, if a thread or block
          // domain's root ID is shifted or gathered, it can overlap
          // in shared or global memory. This doesn't
          // require a RAW sync since each thread would still write every value
          // it would read, but it can require a WAR sync for Shared Memory.
          // Since there isn't a separate structure for WAR than RAW for now
          // we'll flag it on RAW which will trigger the WAR.
          // See test FusionValidateParallelizeShift_CUDA for a
          // concrete example where this sync is required.
          if ((expr->getExprType() == ExprType::GatherOp ||
               expr->getExprType() == ExprType::ShiftOp) &&
              producer->getMemoryType() == MemoryType::Shared &&
              isParallelTypeThreadDim(producer_ptype)) {
            std::unordered_set<Val*> shifted_rfactor_ids;
            if (expr->getExprType() == ExprType::GatherOp) {
              auto gather_op = expr->as<GatherOp>();
              for (auto root_i :
                   c10::irange(producer->getMaybeRFactorDomain().size())) {
                auto rfactor_id = producer->getMaybeRFactorDomain()[root_i];
                // If the window shape is 1, it just copies the
                // producer to the consumer
                if (gather_op->windowShape()[root_i] != 1) {
                  shifted_rfactor_ids.insert(rfactor_id);
                }
              }
            } else if (expr->getExprType() == ExprType::ShiftOp) {
              auto shift_op = expr->as<ShiftOp>();
              for (auto root_i :
                   c10::irange(producer->getMaybeRFactorDomain().size())) {
                auto rfactor_id = producer->getMaybeRFactorDomain()[root_i];
                // If the shift offset is 0, it doesn't actually shift
                if (shift_op->offsets()[root_i] != 0) {
                  shifted_rfactor_ids.insert(rfactor_id);
                }
              }
            }

            // Grab all values between shifted rfactor domains and p_id so we
            // can identify which rfactor domains are inputs to the p_id
            auto p_id_dep_vals =
                DependencyCheck::getAllValsBetween(shifted_rfactor_ids, {p_id});
            // If this shifted rfactor domain is an input to p_id, we
            // must have a WAR sync. Mark raw sync so it will be generated.
            if (!p_id_dep_vals.empty()) {
              raw_dims.set(producer_ptype);
            }
          }

          // When the producer axis is not parallelized, no sync is
          // necessary
          if (!producer_parallelized) {
            continue;
          }

          // When the producer is parallelized, the producer and the
          // consumer must use the same index with the same parallel
          // type. Otherwise, a sync is required. This is not the case
          // when this op is a parallel broadcast.

          if (producer_parallel_bcast) {
            // As long as they are permissively mapped using the same
            // parallel type, no communication is required
            if (producer_ptype == consumer_ptype &&
                ca_map->areMapped(p_id, c_id, IdMappingMode::PERMISSIVE)) {
              continue;
            }
            // Can this happen?
            TORCH_INTERNAL_ASSERT(
                false,
                "Unexpected case. Producer: ",
                producer->toString(),
                ", consumer: ",
                consumer->toString());
          }

          if (producer_ptype == consumer_ptype &&
              useSameIndex(producer, p_id, consumer, c_id, indexing_info)) {
            continue;
          }

          raw_dims.set(producer_ptype);
        } // end for ptypes

        if (raw_dims.hasBID()) {
          TORCH_INTERNAL_ASSERT(
              producer->getMemoryType() == MemoryType::Global,
              "Inconsistent parallelization found between TV",
              producer->name(),
              " (",
              producer->toString(),
              ") and TV",
              consumer->name(),
              "(",
              consumer->toString(),
              "). Producer is required to be in Global Memory based on parallelization strategy.",
              " RAW flags: ",
              raw_dims.toString());
        } else if (raw_dims.hasTID()) {
          TORCH_INTERNAL_ASSERT(
              producer->getMemoryType() == MemoryType::Global ||
                  producer->getMemoryType() == MemoryType::Shared,
              "Inconsistent parallelization found between TV",
              producer->name(),
              " (",
              producer->toString(),
              ") and TV",
              consumer->name(),
              "(",
              consumer->toString(),
              "). Producer is required to be in Global or Shared Memory based on parallelization strategy.",
              " RAW flags: ",
              raw_dims.toString());
        }

      } // end for consumers

      if (raw_dims.any()) {
        needs_raw_sync_[producer] |= raw_dims;
      }

    } // end producer
  }
}

std::string SyncMap::toString() const {
  std::stringstream ss;
  ss << "SyncMap:";
  std::vector<TensorView*> sorted_tvs;
  std::transform(
      needs_raw_sync_.begin(),
      needs_raw_sync_.end(),
      std::back_inserter(sorted_tvs),
      [](auto kv) { return kv.first; });
  std::sort(
      sorted_tvs.begin(),
      sorted_tvs.end(),
      [](TensorView* tv1, TensorView* tv2) {
        return tv1->name() < tv2->name();
      });
  bool is_first = true;
  for (auto tv : sorted_tvs) {
    if (!is_first) {
      ss << ",";
    }
    ss << " " << tv->toString() << " -> " << needs_raw_sync_.at(tv).toString();
    is_first = false;
  }
  return ss.str();
}

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
