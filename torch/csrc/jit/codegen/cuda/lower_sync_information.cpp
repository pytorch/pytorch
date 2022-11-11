
#include <torch/csrc/jit/codegen/cuda/instrumentation.h>
#include <torch/csrc/jit/codegen/cuda/ir_utils.h>
#include <torch/csrc/jit/codegen/cuda/lower2device.h>

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

//! Return true if axis is derived from a root axis that is an input
//! to a CA leaf axis.
bool derivedFromRootCAAxes(TensorView* tv, IterDomain* axis) {
  std::vector<IterDomain*> ca_axes(
      tv->domain()->domain().begin(),
      tv->domain()->domain().begin() + tv->getComputeAtPosition());

  auto ca_root_vals = IterVisitor::getInputsTo(
      std::vector<Val*>(ca_axes.begin(), ca_axes.end()));

  auto root_vals = IterVisitor::getInputsTo({axis});

  return std::any_of(
      root_vals.begin(), root_vals.end(), [&ca_root_vals](auto root) {
        return std::find(ca_root_vals.begin(), ca_root_vals.end(), root) !=
            ca_root_vals.end();
      });
}

} // namespace

void SyncMap::build(Fusion* fusion) {
  FUSER_PERF_SCOPE("GpuLower::Lower::validateParallelize");
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

      // Tracking for quick check later
      std::unordered_set<IterDomain*> producer_within_compute_at;

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

        if (producer_i < producer->getComputeAtPosition()) {
          producer_within_compute_at.emplace(producer_axis);
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

          if (!p_id->isBroadcast() && isParallelTypeThread(producer_ptype) &&
              !(isParallelTypeThread(consumer_ptype) &&
                parallel_bcast_doms.get(consumer_ptype)) &&
              // Being in compute at means consumer and producer rely on the
              // same loop size
              !producer_within_compute_at.count(p_id) &&
              // For usage of derivedFromRootCAAxes check
              // NVFuserTest.FusionAdvancedIndexing1_CUDA
              (c_id == nullptr || !derivedFromRootCAAxes(producer, p_id))) {
            // There must be a consumer axis that uses the same indexing
            // with the same parallel type as the producer axis. The index
            // map is used to to find such an axis. In addition, even when
            // no mapped axis is found in the index map, but when an mapped
            // axis exists in the loop map, the producer and consumer axes
            // may still use the same indexing. That only happens when the
            // producer is derived from a root axis that is an input to any
            // leaf CA axes. In such a case, the axis in the reference
            // tensor that maps to the producer axis is created based on the
            // consumer, so both the producer and consumer axes should have
            // the same indexing. See issue #995 as well as the
            // FusionValidateParallelize6 test for a concrete example.
            auto it = std::find_if(
                consumer->domain()->domain().begin(),
                consumer->domain()->domain().end(),
                [&](IterDomain* c_id_) {
                  return ca_map->areMapped(p_id, c_id_, IdMappingMode::EXACT);
                });
            if (it == consumer->domain()->domain().end()) {
              if (isParallelTypeThread(producer_ptype)) {
                raw_dims.set(producer_ptype);
              }
              if (isParallelTypeThread(consumer_ptype)) {
                raw_dims.set(consumer_ptype);
              }
            }
          }

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

          // When the producer axis is a broadcast, it is not really
          // parallelized unless thread-predicated and concretized
          if (isParallelTypeThread(producer_ptype) && p_id->isBroadcast() &&
              (!parallel_bcast_doms.get(producer_ptype) ||
               !GpuLower::current()
                    ->concretizedBroadcastDomains()
                    ->isConcretized(p_id))) {
            continue;
          }

          // If matching dims and matching parallel types, no comm is necessary.
          if (producer_ptype == consumer_ptype &&
              GpuLower::current()->caMap()->areMapped(
                  p_id, c_id, IdMappingMode::PERMISSIVE)) {
            continue;
          }

          // Set parallel dimensions that communication is occuring over.
          if (isParallelTypeThread(producer_ptype)) {
            raw_dims.set(producer_ptype);
          }
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
              "). Producer is required to be in Global Memory based on parallelization strategy.");
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
              "). Producer is required to be in Global or Shared Memory based on parallelization strategy.");
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
  bool is_first = true;
  for (auto entry : needs_raw_sync_) {
    if (!is_first) {
      ss << ",";
    }
    ss << " " << entry.first->toString() << " -> " << entry.second.toString();
    is_first = false;
  }
  return ss.str();
}

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
