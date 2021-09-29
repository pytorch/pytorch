#include <torch/csrc/jit/codegen/cuda/scheduler/registry.h>
#include <torch/csrc/jit/codegen/cuda/scheduler/utils.h>

#include <torch/csrc/jit/codegen/cuda/arith.h>
#include <torch/csrc/jit/codegen/cuda/compute_at_map.h>
#include <torch/csrc/jit/codegen/cuda/expr_evaluator.h>
#include <torch/csrc/jit/codegen/cuda/instrumentation.h>
#include <torch/csrc/jit/codegen/cuda/ir_iostream.h>
#include <torch/csrc/jit/codegen/cuda/ir_utils.h>
#include <torch/csrc/jit/codegen/cuda/iter_visitor.h>
#include <torch/csrc/jit/codegen/cuda/root_domain_map.h>
#include <torch/csrc/jit/codegen/cuda/transform_replay.h>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {
namespace scheduler_utils {
size_t mergeReduction(
    TensorView* tv,
    const std::unordered_set<IterDomain*>& dont_merge) {
  int prev_i = -1;
  size_t num_merged = 0;
  for (int i = static_cast<int>(tv->nDims()) - 1; i >= 0; i--) {
    if (!tv->axis(i)->isReduction() || dont_merge.count(tv->axis(i))) {
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

size_t mergeNonReduction(
    TensorView* tv,
    const std::unordered_set<IterDomain*>& dont_merge) {
  int prev_i = -1;
  size_t num_merged = 0;
  if (tv->nDims() == 0) {
    return 0;
  }
  for (int i = static_cast<int>(tv->nDims()) - 1; i >= 0; i--) {
    if (tv->axis(i)->isReduction() || dont_merge.count(tv->axis(i))) {
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
  for (auto inp_tv : ir_utils::inputTvsOf(consumer)) {
    inp_tv->computeAt(consumer, pos, mode);
  }
}

void computeWithOutputs(TensorView* producer, int pos, ComputeAtMode mode) {
  for (auto out_tv : ir_utils::outputTvsOf(producer)) {
    producer->computeWith(out_tv, pos, mode);
  }
}

PersistentBufferInfo persistentBuffers(Fusion* fusion) {
  FusionGuard fg(fusion);

  PersistentBufferInfo info;

  ComputeAtRootDomainMap root_map;
  root_map.build();

  auto all_tvs = ir_utils::allTvs(fusion);

  for (auto producer : all_tvs) {
    bool mappable = true;
    auto consumers = ir_utils::consumerTvsOf(producer);
    if (consumers.empty()) {
      continue;
    }

    auto mappable_roots =
        root_map.getMappableDims(producer->domain(), consumers[0]->domain());

    auto p_root = producer->getMaybeRFactorDomain();

    for (auto p_root_id : p_root) {
      if (p_root_id->isReduction()) {
        continue;
      }
      if (!mappable_roots.count(p_root_id)) {
        mappable = false;
        info.unmappable_dims.emplace(p_root_id);
      }
    }

    if (!mappable) {
      info.buffers.push_back(producer);
    }
  }
  return info;
}

TvProperties getProperties(
    Fusion* fusion,
    SchedulerRuntimeInfo& runtime_info,
    TensorView* tv) {
  TvProperties properties;
  FusionGuard fg(fusion);

  auto red_root_dom = tv->getRootDomain();
  for (size_t i = red_root_dom.size(); i > 0; i--) {
    if (red_root_dom[i - 1]->isBroadcast()) {
      continue;
    } else if (red_root_dom[i - 1]->isReduction()) {
      break;
    } else {
      properties.fastest_dim_reduction = false;
      break;
    }
  }

  bool hit_reduction = false;
  auto root_dom = tv->getMaybeRFactorDomain();
  for (auto it = root_dom.rbegin(); it != root_dom.rend(); ++it) {
    auto id = *it;

    auto inferred_val =
        runtime_info.expressionEvaluator().evaluate(id->extent());
    TORCH_INTERNAL_ASSERT(
        inferred_val.has_value(), "Error inferring reduction size.");
    if (id->isReduction()) {
      hit_reduction = true;
      properties.reduction_numel *= inferred_val.value();
    } else {
      auto dim_size = inferred_val.value();
      properties.iteration_numel *= dim_size;
      if (hit_reduction) {
        properties.iter_outside_red *= dim_size;
      } else {
        properties.iter_inside_red *= dim_size;
      }
    }
  }

  if (properties.reduction_numel == 1) {
    properties.iter_outside_red =
        properties.iter_outside_red * properties.iter_inside_red;
    properties.iter_inside_red = 1;
    properties.fastest_dim_reduction = true;
  }

  return properties;
}

void computeAtBetween(
    const std::vector<TensorView*>& producers,
    const std::vector<TensorView*>& overall_consumers,
    int pos,
    ComputeAtMode mode,
    std::unordered_set<IterDomain*> mapped_to_trivial_reduction) {
  for (auto producer : producers) {
    // Figure out what's between producer and overall_consumers, will not give
    // back any consumers that are not downstream from producer
    auto all_vals_between = DependencyCheck::getAllValsBetween(
        {producer}, {overall_consumers.begin(), overall_consumers.end()});

    std::unordered_set<Val*> all_vals_between_set(
        all_vals_between.begin(), all_vals_between.end());

    for (auto consumer : overall_consumers) {
      if (all_vals_between_set.count(consumer)) {
        // The way we generate producers and consumers is that we inch away from
        // inputs/outputs. There's a chance we could meet in the middle.
        if (producer == consumer) {
          continue;
        }

        auto pos_it = std::find_if(
            consumer->domain()->domain().begin(),
            consumer->domain()->domain().end(),
            [&mapped_to_trivial_reduction](IterDomain* id) {
              return mapped_to_trivial_reduction.count(id);
            });

        pos = pos_it == consumer->domain()->domain().end()
            ? pos
            : std::min(
                  (int)std::distance(
                      consumer->domain()->domain().begin(), pos_it) +
                      1,
                  (pos < 0 ? pos + (int)consumer->nDims() : pos));
        // Assume we don't want to reset computeAt on tensors that have already
        // performed it.
        producer->computeAt(consumer, pos, mode);
      }
    }
  }
}

int64_t persistentBufferSize(
    Fusion* fusion,
    SchedulerRuntimeInfo& runtime_info,
    PersistentBufferInfo& persistent_buffers,
    HeuristicSummary* data_cache) {
  FUSER_PERF_SCOPE("scheduler_utils::persistentBufferSize");

  if (persistent_buffers.buffers.empty()) {
    return 0;
  }

  int64_t persistent_buffer_size = 0;

  using ValToFactorMap = std::unordered_map<Val*, int>;
  using ValToFactorMapPtr = std::unique_ptr<ValToFactorMap>;
  using ScopedPersistenceFactorMap =
      std::unordered_map<Val*, ValToFactorMapPtr>;

  HeuristicCacheAccessor<ScopedPersistenceFactorMap>
      scoped_persistent_factor_data;
  // TODO: move all these boilerplate code into the accessor class
  // (follow up)

  // Caching traversal result in this case.
  //  This one is slightly more involving. The end result we want is all the
  //  concrete
  //   int values in scoped_persistence. Essentially:
  //     scoped_persistence [val] = sum_over_all_persistent_tv (
  //     contrubution_from_tv_to_val * persistent_size_of_tv  )
  //  Here contrubution_from_tv_to_val can be determined at compile time.
  //  persistent_size_of_tv is a runtime value but
  //   doesn't require heavy graph traversal.
  //  So in this cache entry we try to save a matrix of contribution factors,
  //  i.e.
  //
  //   new_persistent_factor_map[tv][val] = contribution_from_tv_to_val, from
  //   compile time and we combine the factor
  //
  //   with runtime persistent buffer sizes at runtime.
  if (data_cache && !data_cache->isRecording()) {
    scoped_persistent_factor_data.writeTemporary(
        data_cache->getScopedPersistenceFactorMap());
  } else {
    // Compute new scoped persisitence factor:
    auto new_persistent_factor_map_ptr =
        std::make_unique<ScopedPersistenceFactorMap>();
    auto& new_persistent_factor_map = *new_persistent_factor_map_ptr;

    for (auto tv : persistent_buffers.buffers) {
      auto& consumer_tv_to_factor_map_ptr = new_persistent_factor_map[tv];
      consumer_tv_to_factor_map_ptr = std::make_unique<ValToFactorMap>();
      auto& consumer_tv_to_factor_map = *consumer_tv_to_factor_map_ptr;

      // All expressions between tv and its consumers must have tv's persistent
      // buffer allocated. This is an optimistic view on how many registers we
      // need allocated in the kernel, since if we ordered two persistent
      // buffers that are completely independent to somehow overlap with
      // eachother we would assume we wouldn't need those two buffers active at
      // the same time, even though they would be.
      //
      // Unfortunately this limitation is hard to work around as we would have
      // to actually generate the kernel before we know if it would fit
      // persistently in registers. In practice, though, this should not happen
      // as inlining loop structures where the persistent buffer is used should
      // prevent muiltiple persistent buffers from being merged togther if not
      // necessary.
      auto consumers_of_tv = ir_utils::consumerTvsOf(tv);
      for (auto val : DependencyCheck::getAllValsBetween(
               {tv}, {consumers_of_tv.begin(), consumers_of_tv.end()})) {
        // Persistent normalization kernels imply that all persistent buffers
        // have the same dimensionality. Assume if a persistent buffer is
        // consumed by another we can alias and reuse the memory.
        if (val == tv) {
          continue;
        }

        if (consumer_tv_to_factor_map.count(val)) {
          consumer_tv_to_factor_map.at(val) += 1;
        } else {
          consumer_tv_to_factor_map[val] = 1;
        }
      }
    }

    // Caching boilerplate (TO be cleaned up in a follow up)
    scoped_persistent_factor_data.takeNew(new_persistent_factor_map_ptr);
    if (data_cache && data_cache->isRecording()) {
      data_cache->setScopedPersistenceFactorMap(
          scoped_persistent_factor_data.read());
    }
  }

  auto& scoped_persistence_factor = scoped_persistent_factor_data.read();

  // Runtime: convert the persistent factor to actual values
  std::unordered_map<Val*, int64_t> scoped_persistence;

  for (auto tv : persistent_buffers.buffers) {
    int64_t tv_persistent_numel = -1;
    for (auto id : tv->getMaybeRFactorDomain()) {
      if (id->isReduction() || id->isBroadcast()) {
        continue;
      }
      // Unmappable dimensions are those that we cannot inline into other
      // tensor views. So they're the ones that need to be persistent.
      if (!persistent_buffers.unmappable_dims.count(id)) {
        continue;
      }

      auto id_size = runtime_info.expressionEvaluator().evaluate(id->extent());
      TORCH_INTERNAL_ASSERT(
          id_size.has_value(),
          "Cannot generate heuristics if we don't have input information.");
      if (tv_persistent_numel == -1) {
        tv_persistent_numel = id_size.value();
      } else {
        tv_persistent_numel *= id_size.value();
      }
    }

    persistent_buffer_size =
        tv_persistent_numel * dataTypeSize(tv->getDataType().value());

    // Look up the contribution part from the cached matrix:
    auto scoped_factor_it = scoped_persistence_factor.find(tv);
    if (scoped_factor_it != scoped_persistence_factor.end()) {
      // now looking at scoped_persistence_factor[tv]
      for (auto val_to_factor_it : *(scoped_factor_it->second)) {
        // (val_to_factor_it) is (val, factor)
        int64_t persistent_buffer_size_contribution =
            persistent_buffer_size * val_to_factor_it.second;

        //  try to write factor * persistent_buffer_size into
        //  scoped_persistence[val]
        auto val_it = scoped_persistence.find(val_to_factor_it.first);
        if (val_it == scoped_persistence.end()) {
          scoped_persistence[val_to_factor_it.first] =
              persistent_buffer_size_contribution;
        } else {
          val_it->second += persistent_buffer_size_contribution;
        }
      }
    }
  }

  // Find the maximum persistent buffer use
  int64_t max_persistence_size = 0;
  for (auto persistent_entry : scoped_persistence) {
    max_persistence_size =
        std::max(max_persistence_size, persistent_entry.second);
  }

  return max_persistence_size;
}

std::unordered_set<IterDomain*> getTrivialReductionMap(Fusion* fusion) {
  auto all_tvs = ir_utils::allTvs(fusion);
  std::unordered_set<IterDomain*> mapped_to_trivial_reduction;
  for (auto tv : all_tvs) {
    // root domain vs domain shouldn't matter as at this point we shouldn't have
    // any transformations.
    for (auto id : tv->getRootDomain()) {
      if (id->isTrivialReduction()) {
        mapped_to_trivial_reduction.emplace(id);
      }
    }
  }

  if (!mapped_to_trivial_reduction.empty()) {
    // Shouldn't matter which compute at map we use
    auto ca_index_map = ComputeAtMap(ComputeAtMap::MappingMode::INDEX);
    ca_index_map.build(fusion);
    // Make a copy we need to check mappings of all
    auto trivial_ids = mapped_to_trivial_reduction;
    for (auto tv : all_tvs) {
      for (auto id : tv->getRootDomain()) {
        if (!id->extent()->isOneInt()) {
          continue;
        }
        if (std::any_of(
                trivial_ids.begin(),
                trivial_ids.end(),
                [&ca_index_map, &id](IterDomain* trivial_id) {
                  return ca_index_map.areMapped(id, trivial_id);
                })) {
          mapped_to_trivial_reduction.emplace(id);
        }
      }
    }
  }
  return mapped_to_trivial_reduction;
}

std::pair<bool, bool> canonicalDimReduction(Fusion* fusion, TensorView* tv) {
  std::unordered_set<IterDomain*> mapped_to_trivial_reduction =
      getTrivialReductionMap(fusion);

  TORCH_INTERNAL_ASSERT(tv != nullptr);

  // We coalesce all reduction axes to the right;
  bool has_red_axis = mergeReduction(tv, mapped_to_trivial_reduction) > 0;

  bool has_iter_axis = mergeNonReduction(tv, mapped_to_trivial_reduction) > 0;
  return {has_iter_axis, has_red_axis};
}

std::vector<TensorView*> getReductionTvs(Fusion* fusion) {
  auto all_tvs = ir_utils::allTvs(fusion);
  std::vector<TensorView*> reduction_tvs;
  for (auto tv : all_tvs) {
    if (!tv->isFusionInput() &&
        std::any_of(
            tv->domain()->domain().begin(),
            tv->domain()->domain().end(),
            [](IterDomain* id) {
              return id->isReduction() && !id->isTrivialReduction();
            })) {
      reduction_tvs.emplace_back(tv);
    }
  }

  // Remove multi outputs from reduction tensor views
  std::unordered_set<Expr*> seen_reduction_exprs;
  reduction_tvs.erase(
      std::remove_if(
          reduction_tvs.begin(),
          reduction_tvs.end(),
          [&seen_reduction_exprs](TensorView* tv) {
            TORCH_INTERNAL_ASSERT(
                tv->definition() != nullptr,
                "Somehow a tensor view without a definition but a reduction snuck into the scheduler reduction list.");
            if (!seen_reduction_exprs.emplace(tv->definition()).second) {
              return true;
            }
            return false;
          }),
      reduction_tvs.end());
  return reduction_tvs;
}

TensorView* scheduleReductionTV(
    const ReductionParams& rparams,
    TensorView* reduction_tv,
    bool has_iter_axis) {
  TensorView* reference_tv = nullptr;
  if (rparams.fastest_dim) {
    const int iter_axis = 0;
    const int reduce_axis = has_iter_axis ? 1 : 0;

    // Do multiple reductions per block
    if (rparams.multiple_reds_per_blk) {
      if (rparams.reduction_unroll) {
        // Fastest dim, multiple reductions per block
        // Output Dimensions
        // [x-BIDx, x-TIDy
        //  0       1
        //
        //  Reduction Dimensions
        //  rF-Remain, rf-Unswitch, rf-Unroll, X-TIDx]
        //  2(r)          3(r+1)     4(r+2)    5(r+3)
        //  Reduction Dimensions
        //  rF-Remain, rf-Unswitch, X-TIDx, rf-Vectorize]
        //  2(r)          3(r+1)     4(r+2)    5(r+3)

        //  X-TIDx, rF-Remain, rf-Unswitch, rf-Unroll/Vect]
        //   2(r)     3(r+1)       4(r+2)      5(r+3)

        if (!rparams.persistent_kernel) {
          if (rparams.vectorize) {
            reduction_tv->split(reduce_axis, rparams.loop_unroll);
            reduction_tv->split(
                reduce_axis, NamedScalar::getParallelDim(ParallelType::TIDx));
          } else {
            reduction_tv->split(
                reduce_axis, NamedScalar::getParallelDim(ParallelType::TIDx));
            reduction_tv->split(reduce_axis, rparams.loop_unroll);
          }
          // Unswitch axis which gives us finer control on allocations with
          // unrolling
          reduction_tv->split(reduce_axis, 1);
        } else {
          if (rparams.vectorize) {
            reduction_tv->split(reduce_axis, rparams.batches_per_block, false);
            reduction_tv->split(reduce_axis + 1, rparams.loop_unroll);
          } else {
            reduction_tv->split(
                reduce_axis,
                rparams.batches_per_block * rparams.loop_unroll,
                false);
            reduction_tv->split(reduce_axis, rparams.loop_unroll);
          }
          // Unswitch axis which gives us finer control on allocations with
          // unrolling
          reduction_tv->split(reduce_axis, 1);
        }

        if (rparams.vectorize) {
          reduction_tv->reorder(
              {{reduce_axis, reduce_axis + 1},
               {reduce_axis + 1, reduce_axis + 2},
               {reduce_axis + 2, reduce_axis}});
        } else {
          reduction_tv->reorder(
              {{reduce_axis + 3, reduce_axis},
               {reduce_axis, reduce_axis + 1},
               {reduce_axis + 1, reduce_axis + 2},
               {reduce_axis + 2, reduce_axis + 3}});
        }

        reference_tv = ir_utils::rfactorHelper(
            reduction_tv, {reduce_axis + 1, reduce_axis + 2, reduce_axis + 3});

        reference_tv->axis(reduce_axis)->parallelize(ParallelType::TIDx);

        if (rparams.vectorize) {
          reference_tv->axis(reduce_axis + 3)
              ->parallelize(ParallelType::Vectorize);
        } else {
          reference_tv->axis(reduce_axis + 3)
              ->parallelize(ParallelType::Unroll);
        }
        reference_tv->axis(reduce_axis + 2)
            ->parallelize(ParallelType::Unswitch);

        if (has_iter_axis) {
          reference_tv->split(
              iter_axis, NamedScalar::getParallelDim(ParallelType::TIDy));
          reference_tv->axis(iter_axis + 1)->parallelize(ParallelType::TIDy);
          if (rparams.split_grid_dim) {
            reference_tv->split(iter_axis, x_grid_limit);
            reference_tv->axis(iter_axis + 1)->parallelize(ParallelType::BIDx);
          } else {
            reference_tv->axis(iter_axis)->parallelize(ParallelType::BIDx);
          }
        }
      } else {
        TORCH_INTERNAL_ASSERT(
            has_iter_axis,
            "This scheduler requires an outer dim to the reduction.");
        // Fastest dim, Multiple reductions per block iter unroll
        // Output Dimensions
        // [x-BIDx, x-Unswitch, x-Unroll, x-TIDy
        //  0       1           2         3
        //
        //  Reduction Dimensions
        //  rF-Remain, r-TIDx]
        //  4(r)     5(r+1)
        if (!rparams.persistent_kernel) {
          reduction_tv->split(
              1, NamedScalar::getParallelDim(ParallelType::TIDx));
        } else {
          reduction_tv->split(1, rparams.batches_per_block, false);
        }

        reference_tv = ir_utils::rfactorHelper(reduction_tv, {1});

        reference_tv->split(0, NamedScalar::getParallelDim(ParallelType::TIDy));
        reference_tv->split(0, rparams.loop_unroll);
        // Unswitch axis which gives us finer control on allocations with
        // unrolling
        reference_tv->split(0, 1);

        // [x-BIDx, x-Unswitch, x-Unroll, x-TIDy, rF-Remain, r-TIDx]
        //     0         1          2        3        4         5
        // -> [x-BIDx, x-TIDy, rF-Remain, x-Unswitch, x-Unroll, r-TIDx]
        //       0        1         2           3          4       5

        reference_tv->reorder({{1, 3}, {2, 4}, {3, 1}, {4, 2}});

        reference_tv->axis(1)->parallelize(ParallelType::TIDy);
        reference_tv->axis(3)->parallelize(ParallelType::Unswitch);
        reference_tv->axis(4)->parallelize(ParallelType::Unroll);
        reference_tv->axis(5)->parallelize(ParallelType::TIDx);

        if (rparams.split_grid_dim) {
          reference_tv->split(0, x_grid_limit);
          reference_tv->axis(1)->parallelize(ParallelType::BIDx);
        } else {
          reference_tv->axis(0)->parallelize(ParallelType::BIDx);
        }
      }
    } else {
      // Not multiple reductions per block
      if (rparams.cross_grid) {
        TORCH_INTERNAL_ASSERT(
            rparams.reduction_unroll,
            "Unrolling on iter domain not supported in this scheduler.");

        TORCH_INTERNAL_ASSERT(
            !rparams.persistent_kernel,
            "Grid reductions not implemented yet for persistent kernels.");

        // Fastest dim, cross grid, cross block
        //      [outputs,
        // Idx:     0
        //   | rf-Remain, r-BIDx, r-TIDy, rf-Unswitch, rf-Unroll, r-TIDx]
        //       1(r)     2(r+1)  3(r+2)     4(r+3)      5(r+4)   6(r+5)|
        //   | rf-Remain, r-BIDx, r-TIDy, rf-Unswitch, r-TIDx, r-Vectorize]
        //       1(r)     2(r+1)  3(r+2)     4(r+3)    5(r+4)    6(r+5)|
        //                Reduction Dimensions

        //   | r-BIDx, r-TIDy, r-TIDx, rf-Remain, rf-Unswitch, rf-Unroll/Vect]
        //       1(r)  2(r+1)  3(r+2)   4(r+3)       5(r+4)     6(r+5)  |
        //                Reduction Dimensions

        if (rparams.vectorize) {
          reduction_tv->split(reduce_axis, rparams.loop_unroll);
          reduction_tv->split(
              reduce_axis, NamedScalar::getParallelDim(ParallelType::TIDx));
        } else {
          reduction_tv->split(
              reduce_axis, NamedScalar::getParallelDim(ParallelType::TIDx));
          reduction_tv->split(reduce_axis, rparams.loop_unroll);
        }
        reduction_tv->split(reduce_axis, 1);
        // Unswitch axis which gives us finer control on allocations with
        // unrolling
        reduction_tv->split(
            reduce_axis, NamedScalar::getParallelDim(ParallelType::TIDy));
        reduction_tv->split(
            reduce_axis, NamedScalar::getParallelDim(ParallelType::BIDx));

        if (rparams.vectorize) {
          reduction_tv->reorder(
              {{reduce_axis, reduce_axis + 3},
               {reduce_axis + 1, reduce_axis},
               {reduce_axis + 2, reduce_axis + 1},
               {reduce_axis + 3, reduce_axis + 4},
               {reduce_axis + 4, reduce_axis + 2}});
        } else {
          reduction_tv->reorder(
              {{reduce_axis, reduce_axis + 3},
               {reduce_axis + 1, reduce_axis},
               {reduce_axis + 2, reduce_axis + 1},
               {reduce_axis + 3, reduce_axis + 4},
               {reduce_axis + 4, reduce_axis + 5},
               {reduce_axis + 5, reduce_axis + 2}});
        }

        reference_tv = ir_utils::rfactorHelper(
            reduction_tv, {reduce_axis + 3, reduce_axis + 4, reduce_axis + 5});

        if (rparams.vectorize) {
          reference_tv->axis(reduce_axis + 5)
              ->parallelize(ParallelType::Vectorize);
        } else {
          reference_tv->axis(reduce_axis + 5)
              ->parallelize(ParallelType::Unroll);
        }
        reference_tv->axis(reduce_axis + 4)
            ->parallelize(ParallelType::Unswitch);

        reference_tv->axis(reduce_axis + 2)->parallelize(ParallelType::TIDx);
        reference_tv->axis(reduce_axis + 1)->parallelize(ParallelType::TIDy);
        reference_tv->axis(reduce_axis)->parallelize(ParallelType::BIDx);

        if (has_iter_axis) {
          if (rparams.split_grid_dim) {
            reference_tv->split(iter_axis, y_grid_limit);
            reference_tv->axis(iter_axis + 1)->parallelize(ParallelType::BIDy);
          } else {
            reference_tv->axis(iter_axis)->parallelize(ParallelType::BIDy);
          }
        }

      } else {
        // Not cross grid
        if (rparams.reduction_unroll) {
          // Fastest dim, Reduction unroll
          // Output Dimensions
          // [BIDx
          //  0
          //
          // Reduction Dimensions
          // rF-Remain, rf-Unswitch, rf-Unroll, r-TIDx]
          // 1(r)      2(r+1)        3(r+2)      4(r+3)
          // rF-Remain, rf-Unswitch, r-TIDx, rf-Vectorize]
          // 1(r)      2(r+1)        3(r+2)      4(r+3)

          //  r-TIDx, rF-Leftover, rf-Unswitch, rf-Unroll]
          //  1(r)       2(r+1)      3(r+2)       4(r+3)

          if (!rparams.persistent_kernel) {
            if (rparams.vectorize) {
              reduction_tv->split(reduce_axis, rparams.loop_unroll);
              reduction_tv->split(
                  reduce_axis, NamedScalar::getParallelDim(ParallelType::TIDx));
            } else {
              reduction_tv->split(
                  reduce_axis, NamedScalar::getParallelDim(ParallelType::TIDx));
              reduction_tv->split(reduce_axis, rparams.loop_unroll);
            }
            // Unswitch axis which gives us finer control on allocations with
            // unrolling
            reduction_tv->split(reduce_axis, 1);
          } else {
            if (rparams.vectorize) {
              reduction_tv->split(
                  reduce_axis, rparams.batches_per_block, false);
              reduction_tv->split(reduce_axis + 1, rparams.loop_unroll);
            } else {
              reduction_tv->split(
                  reduce_axis,
                  rparams.batches_per_block * rparams.loop_unroll,
                  false);
              reduction_tv->split(reduce_axis, rparams.loop_unroll);
            }
            // Unswitch axis which gives us finer control on allocations with
            // unrolling
            reduction_tv->split(reduce_axis, 1);
          }

          if (rparams.vectorize) {
            reduction_tv->reorder(
                {{reduce_axis + 2, reduce_axis},
                 {reduce_axis, reduce_axis + 1},
                 {reduce_axis + 1, reduce_axis + 2}});
          } else {
            reduction_tv->reorder(
                {{reduce_axis + 3, reduce_axis},
                 {reduce_axis, reduce_axis + 1},
                 {reduce_axis + 1, reduce_axis + 2},
                 {reduce_axis + 2, reduce_axis + 3}});
          }

          reference_tv = ir_utils::rfactorHelper(
              reduction_tv,
              {reduce_axis + 1, reduce_axis + 2, reduce_axis + 3});

          reference_tv->axis(reduce_axis)->parallelize(ParallelType::TIDx);
          if (rparams.vectorize) {
            reference_tv->axis(reduce_axis + 3)
                ->parallelize(ParallelType::Vectorize);
          } else {
            reference_tv->axis(reduce_axis + 3)
                ->parallelize(ParallelType::Unroll);
          }
          reference_tv->axis(reduce_axis + 2)
              ->parallelize(ParallelType::Unswitch);

          if (has_iter_axis) {
            if (rparams.split_grid_dim) {
              reference_tv->split(iter_axis, x_grid_limit);
              reference_tv->axis(iter_axis + 1)
                  ->parallelize(ParallelType::BIDx);
            } else {
              reference_tv->axis(iter_axis)->parallelize(ParallelType::BIDx);
            }
          }
        } else {
          TORCH_INTERNAL_ASSERT(
              has_iter_axis, "Need iteration axis for iteration unroll.");
          // Fastest dim, Reduction Splits
          // Output Dimensions
          // [BIDx, x-Unswitch, x-Unroll
          //  0
          //
          // Reduction Dimensions
          // rF-Remain, r-TIDx]
          // 1(r)       2(r+1)

          if (!rparams.persistent_kernel) {
            reduction_tv->split(
                reduce_axis, NamedScalar::getParallelDim(ParallelType::TIDx));
          } else {
            reduction_tv->split(reduce_axis, rparams.batches_per_block, false);
          }

          reduction_tv->split(iter_axis, rparams.loop_unroll);
          // Unswitch axis which gives us finer control on allocations with
          // unrolling
          reduction_tv->split(iter_axis, 1);

          // [x-BIDx, x-Unswitch, x-Unroll, rF-Remain, r-TIDx]
          //     0         1          2        3        4
          // -> [x-BIDx, rF-Remain, x-Unswitch, x-Unroll, r-TIDx]
          //       0        1          2           3          4

          reduction_tv->reorder({{1, 2}, {2, 3}, {3, 1}});

          reference_tv = ir_utils::rfactorHelper(reduction_tv, {1});

          reference_tv->axis(4)->parallelize(ParallelType::TIDx);
          reference_tv->axis(3)->parallelize(ParallelType::Unroll);
          reference_tv->axis(2)->parallelize(ParallelType::Unswitch);

          if (rparams.split_grid_dim) {
            reference_tv->split(0, x_grid_limit);
            reference_tv->axis(1)->parallelize(ParallelType::BIDx);
          } else {
            reference_tv->axis(0)->parallelize(ParallelType::BIDx);
          }
        }
      }
    }
  } else {
    if (rparams.cross_block) {
      if (rparams.cross_grid) {
        TORCH_INTERNAL_ASSERT(
            rparams.reduction_unroll,
            "Unrolling on iter domain not supported in this scheduler.");

        TORCH_INTERNAL_ASSERT(
            !rparams.persistent_kernel,
            "Grid reductions not implemented yet for persistent kernels.");

        // Outer Dim, cross grid, cross block

        // Unrolling in this case can only be applied to the reduction
        // dimension since currently, grid reductions cannot be called
        // multiple times
        //
        // Output Dimensions
        // [x-BIDx, x-TIDx,
        //  0         1
        //
        // Reduction Dimensions
        // rF-Leftover, r-BIDy, r-TIDy, rf-Unswitch, rf-Unroll]
        // 2(-5)        3(-4)   4(-3)   5(-2)        6(-1)

        // r-BIDy, r-TIDy, rF-Leftover, rf-Unswitch, rf-Unroll]
        // 2(-5)    3(-4)      4(-3)       5(-2)        6(-1)

        reduction_tv->split(1, rparams.loop_unroll);
        // Unswitch axis which gives us finer control on allocations with
        // unrolling
        reduction_tv->split(1, 1);
        reduction_tv->split(1, NamedScalar::getParallelDim(ParallelType::TIDy));
        reduction_tv->split(1, NamedScalar::getParallelDim(ParallelType::BIDy));

        reduction_tv->split(0, NamedScalar::getParallelDim(ParallelType::TIDx));

        reduction_tv->reorder({{2, 4}, {3, 2}, {4, 3}});

        reference_tv = ir_utils::rfactorHelper(
            reduction_tv,
            {4, 5, 6}); // NOLINT(cppcoreguidelines-avoid-magic-numbers)

        reference_tv->axis(6)->parallelize(ParallelType::Unroll);
        reference_tv->axis(5)->parallelize(ParallelType::Unswitch);
        reference_tv->axis(3)->parallelize(ParallelType::TIDy);
        reference_tv->axis(2)->parallelize(ParallelType::BIDy);
        reference_tv->axis(1)->parallelize(ParallelType::TIDx);
        reference_tv->axis(0)->parallelize(ParallelType::BIDx);
      } else {
        if (rparams.reduction_unroll || rparams.loop_unroll == 1) {
          // Outer Dim, cross block, unroll reduction dimension

          // Reduction Splits
          // Output Dimensions
          // [x-BIDx, x-TIDx
          //  0       1
          //
          // Reduction Dimensions
          // rF-Leftover, r-TIDy, rf-Unswitch, rf-Unroll]
          // 2(-4)        3(-3)   4(-2)       5(-1)

          // r-TIDy, rF-Leftover, rf-Unswitch, rf-Unroll]
          // 2(-4)      3(-3)       4(-2)       5(-1)
          if (!rparams.persistent_kernel) {
            reduction_tv->split(1, rparams.loop_unroll);
            // Unswitch axis which gives us finer control on allocations with
            // unrolling
            reduction_tv->split(1, 1);
            reduction_tv->split(
                1, NamedScalar::getParallelDim(ParallelType::TIDy));
          } else {
            reduction_tv->split(1, rparams.batches_per_block, false);
            reduction_tv->split(2, rparams.loop_unroll);
            reduction_tv->split(2, 1);
          }

          reduction_tv->split(
              0, NamedScalar::getParallelDim(ParallelType::TIDx));

          reduction_tv->reorder({{2, 3}, {3, 2}});

          reference_tv = ir_utils::rfactorHelper(
              reduction_tv,
              {3, 4, 5}); // NOLINT(cppcoreguidelines-avoid-magic-numbers)

          reference_tv->axis(5)->parallelize(ParallelType::Unroll);
          reference_tv->axis(4)->parallelize(ParallelType::Unswitch);
          reference_tv->axis(2)->parallelize(ParallelType::TIDy);
          reference_tv->axis(1)->parallelize(ParallelType::TIDx);
          reference_tv->axis(0)->parallelize(ParallelType::BIDx);
        } else {
          // Outer Dim, cross block, unroll iter dimension

          // Output Dimensions
          // [x-BIDx, x-Unswitch, x-Unroll, x-TIDx
          //  0       1           2         3
          // [x-BIDx, x-Unswitch, x-TIDx, x-Vectorize
          //  0       1           2         3
          //
          // Reduction Dimensions
          // rF-Leftover, r-TIDy]
          // 4(-2)        5(-1)

          // The unroll/unswitch dimension needs to be within the rF-Leftover
          // dimension
          //    [x-BIDx, x-Unswitch, x-Unroll, x-TIDx, rF-Leftover, r-TIDy]
          //      0(-6)     1(-5)      2(-4)    3(-3)     4(-2)      5(-1)
          //    [x-BIDx, x-Unswitch, x-TIDx, x-Vectorize, rF-Leftover, r-TIDy]
          //      0(-6)     1(-5)      2(-4)    3(-3)     4(-2)      5(-1)
          // -> [x-BIDx, x-TIDx, rF-Leftover, x-Unswitch, x-Unroll/Vect,
          // r-TIDy]
          //      0(-6)   1(-5)     2(-4)        3(-3)      4(-2)        5(-1)

          if (!rparams.persistent_kernel) {
            reduction_tv->split(
                1, NamedScalar::getParallelDim(ParallelType::TIDy));
          } else {
            reduction_tv->split(1, rparams.batches_per_block, false);
          }
          if (rparams.vectorize) {
            reduction_tv->split(0, rparams.loop_unroll);
            reduction_tv->split(
                0, NamedScalar::getParallelDim(ParallelType::TIDx));

          } else {
            reduction_tv->split(
                0, NamedScalar::getParallelDim(ParallelType::TIDx));
            reduction_tv->split(0, rparams.loop_unroll);
          }
          // Unswitch axis which gives us finer control on allocations with
          // unrolling
          reduction_tv->split(0, 1);

          if (rparams.vectorize) {
            reduction_tv->reorder({{1, 3}, {2, 1}, {3, 4}, {4, 2}});
          } else {
            reduction_tv->reorder({{1, 3}, {2, 4}, {3, 1}, {4, 2}});
          }

          reference_tv = ir_utils::rfactorHelper(
              reduction_tv,
              {2}); // NOLINT(cppcoreguidelines-avoid-magic-numbers)

          reference_tv->axis(5)->parallelize(ParallelType::TIDy);
          reference_tv->axis(1)->parallelize(ParallelType::TIDx);
          if (rparams.vectorize) {
            reference_tv->axis(4)->parallelize(ParallelType::Vectorize);
          } else {
            reference_tv->axis(4)->parallelize(ParallelType::Unroll);
          }
          reference_tv->axis(3)->parallelize(ParallelType::Unswitch);
          reference_tv->axis(0)->parallelize(ParallelType::BIDx);
        }
      }
    } else {
      if (rparams.reduction_unroll) {
        // Outer Dim, no parallelization on reduction, unroll reduction axis
        // Output Dimensions
        // [x-BIDx, x-TIDx
        //  0       1
        //
        // Reduction Dimensions
        // rf-Leftover, rf-Unswitch, r-Unroll]
        //       2            3         4
        if (rparams.persistent_kernel) {
          reduction_tv->split(1, rparams.batches_per_block, false);
          reduction_tv->split(2, rparams.loop_unroll);
          // Reduction Dimensions
          // rf-Leftover, r-TIDy, rf-Unroll]
          //       2         3         4
        } else {
          reduction_tv->split(1, rparams.loop_unroll);
          // Unswitch axis which gives us finer control on allocations with
          // unrolling
          reduction_tv->split(1, 1);
        }

        reduction_tv->split(0, NamedScalar::getParallelDim(ParallelType::TIDx));

        if (rparams.persistent_kernel) {
          // [x-BIDx, x-TIDx, rf-Leftover, r-TIDy, rf-Unroll]
          //     0       1         2         3         4
          reduction_tv->reorder({{3, 2}, {2, 3}});
          // [x-BIDx, x-TIDx, r-TIDy, rf-Leftover, rf-Unroll]
          //     0       1       2           3         4
          reference_tv = ir_utils::rfactorHelper(
              reduction_tv,
              {3, 4}); // NOLINT(cppcoreguidelines-avoid-magic-numbers)
          reference_tv->axis(0)->parallelize(ParallelType::BIDx);
          reference_tv->axis(1)->parallelize(ParallelType::TIDx);
          reference_tv->axis(2)->parallelize(ParallelType::TIDy);
          reference_tv->axis(3)->parallelize(ParallelType::Unswitch);
          reference_tv->axis(4)->parallelize(ParallelType::Unroll);
        } else {
          reference_tv = ir_utils::rfactorHelper(
              reduction_tv,
              {2, 3}); // NOLINT(cppcoreguidelines-avoid-magic-numbers)
          reference_tv->axis(0)->parallelize(ParallelType::BIDx);
          reference_tv->axis(1)->parallelize(ParallelType::TIDx);
          reference_tv->axis(3)->parallelize(ParallelType::Unswitch);
          reference_tv->axis(4)->parallelize(ParallelType::Unroll);
        }
      } else {
        // No parallelization on reduction, unroll iter axis
        // Output Dimensions
        // [x-BIDx, x-Unswitch, x-Unroll, x-TIDx
        //  0       1           2         3
        // [x-BIDx, x-Unswitch, x-TIDx, x-Vectorize
        //  0       1           2         3
        //
        // Reduction Dimensions
        // rf-Leftover, r-{1}]
        // 4(-1)
        //
        // Fake an rfactor to make scheduling more consistent.
        //
        // The unroll/unswitch dimension needs to be within the rF-Leftover
        // dimension
        if (rparams.persistent_kernel) {
          reduction_tv->split(1, rparams.batches_per_block, false);
        } else {
          reduction_tv->split(1, 1);
        }

        if (rparams.vectorize) {
          reduction_tv->split(0, rparams.loop_unroll);
          reduction_tv->split(
              0, NamedScalar::getParallelDim(ParallelType::TIDx));
        } else {
          reduction_tv->split(
              0, NamedScalar::getParallelDim(ParallelType::TIDx));
          reduction_tv->split(0, rparams.loop_unroll);
        }

        reduction_tv->split(0, 1);

        // [x-BIDx, x-Unswitch, x-Unroll, x-TIDx, rf-Leftover, r-1]
        //   0         1          2        3            4       5
        // [x-BIDx, x-Unswitch, x-TIDx, x-Vectorize, rf-Leftover, r-1]
        //   0         1          2        3              4        5

        if (rparams.vectorize) {
          reduction_tv->reorder({{1, 3}, {2, 1}, {3, 4}, {4, 2}});
        } else {
          reduction_tv->reorder({{1, 3}, {2, 4}, {3, 1}, {4, 2}});
        }

        // [x-BIDx, x-TIDx, rf-Leftover, x-Unswitch, x-Unroll, r-1(TIDy)]
        //   0       1            2           3          4      5

        reference_tv = ir_utils::rfactorHelper(reduction_tv, {2});
        if (rparams.persistent_kernel) {
          reference_tv->axis(5)->parallelize(ParallelType::TIDy);
        }

        reference_tv->axis(0)->parallelize(ParallelType::BIDx);
        reference_tv->axis(1)->parallelize(ParallelType::TIDx);
        reference_tv->axis(3)->parallelize(ParallelType::Unswitch);
        if (rparams.vectorize) {
          reference_tv->axis(4)->parallelize(ParallelType::Vectorize);
        } else {
          reference_tv->axis(4)->parallelize(ParallelType::Unroll);
        }
      }
    }
  }
  return reference_tv;
}

// Reset inputs and outputs to global memory, everything else to local.
void clearMemorySpace(Fusion* fusion) {
  for (auto tv : ir_utils::allTvs(fusion)) {
    if (tv->isFusionInput() || tv->isFusionOutput()) {
      tv->setMemoryType(MemoryType::Global);
    } else {
      tv->setMemoryType(MemoryType::Local);
    }
  }
}

// Returns cached after tensors of the fusion inputs if unrolled. Otherwise
// return empty vector.
std::vector<TensorView*> cacheInputs(Fusion* fusion, bool unroll) {
  if (!unroll) {
    return {};
  }

  std::vector<TensorView*> cached_inputs;
  // If we're going to unroll, make a cache of the inputs
  auto in_tvs = ir_utils::filterByType<TensorView>(fusion->inputs());
  for (auto tv : in_tvs) {
    if (tv->uses().empty()) {
      continue;
    }
    auto cached_tv = tv->cache_after();
    cached_inputs.emplace_back(cached_tv);
  }
  return cached_inputs;
}

// Returns the pairs of <cache of each fusion output, corresponding output> for
// all outputs.
std::vector<std::pair<TensorView*, TensorView*>> cacheAndForkOutputs(
    Fusion* fusion,
    bool unroll) {
  std::vector<std::pair<TensorView*, TensorView*>> cached_outputs;
  // For intermediate outputs, apply cache_fork
  for (const auto output :
       ir_utils::filterByType<TensorView>(fusion->outputs())) {
    if (output->definition() == nullptr) {
      continue;
    }
    if (!output->uses().empty()) {
      auto cached_output = output->as<TensorView>()->cache_fork();
      cached_outputs.emplace_back(std::make_pair(output, cached_output));
    } else if (unroll) {
      auto cached_output = output->as<TensorView>()->cache_before();
      cached_outputs.emplace_back(std::make_pair(cached_output, output));
    }
  }
  return cached_outputs;
}

void multiReductionInliner(
    Fusion* fusion,
    const ReductionParams& rparams,
    TensorView* reduction_tv,
    TensorView* reference_tv,
    std::vector<TensorView*> reduction_tvs,
    std::vector<TensorView*> cached_inputs,
    std::vector<std::pair<TensorView*, TensorView*>> cached_outputs) {
  TransformPropagator::from(reference_tv);

  // Apply rfactor to all reductions if applicable
  std::vector<TensorView*> rfactor_tvs;

  if (reference_tv != reduction_tv) {
    std::vector<int> rfactor_axes;
    for (size_t i = 0; i < reference_tv->nDims(); i++) {
      if (reference_tv->axis((int)i)->isReduction() &&
          reference_tv->axis((int)i)->isRFactorProduct()) {
        rfactor_axes.push_back((int)i);
      }
    }

    for (auto reduction_tv_ : reduction_tvs) {
      if (reduction_tv_ == reduction_tv) {
        // The reduction tv
        rfactor_tvs.push_back(reference_tv);
        continue;
      } else {
        rfactor_tvs.push_back(
            ir_utils::rfactorHelper(reduction_tv_, rfactor_axes));
      }
    }

    TORCH_INTERNAL_ASSERT(
        reduction_tvs.size() == rfactor_tvs.size(),
        "Expected all reductions to contain rfactor.");
  }

  // Propagate parallelization
  parallelizeAllLike(reference_tv, ir_utils::allTvs(fusion));

  // Find iter domains that are mapped to a trivial reduction, these should
  // never be inlined.
  std::unordered_set<IterDomain*> mapped_to_trivial_reduction =
      getTrivialReductionMap(fusion);

  if (rparams.loop_unroll > 1) {
    // Inline Input caches to their consumers outside unswitched/vectorization
    // position Inline consumers of input caches to rfactor tensors

    // Mark which tensor views are actual input caches to leave vectorization on
    // them
    std::unordered_set<TensorView*> keep_unrolled;

    std::vector<TensorView*> compute_from;

    // Grab all tensor views that should be vectorized
    auto vecotrizable_inputs_outputs =
        getVectorizableInputsOutputs(reference_tv);

    // Inputs to cache
    for (auto cached_input : cached_inputs) {
      auto consumers_of_input_cache = ir_utils::consumerTvsOf(cached_input);
      for (auto consumer : consumers_of_input_cache) {
        auto unswitch_it = std::find_if(
            consumer->domain()->domain().begin(),
            consumer->domain()->domain().end(),
            [&mapped_to_trivial_reduction](IterDomain* id) {
              return id->getParallelType() == ParallelType::Unswitch ||
                  id->getParallelType() == ParallelType::Unroll ||
                  id->getParallelType() == ParallelType::Vectorize ||
                  id->getParallelType() == ParallelType::MisalignedVectorize ||
                  mapped_to_trivial_reduction.count(id);
            });
        auto unswitch_pos = unswitch_it == consumer->domain()->domain().end()
            ? -1
            : std::distance(consumer->domain()->domain().begin(), unswitch_it) +
                1;

        cached_input->computeAt(
            consumer, unswitch_pos, ComputeAtMode::BestEffort);
        compute_from.push_back(consumer);

        if (rparams.vectorize) {
          auto producer_tvs = ir_utils::producerTvsOf(cached_input);
          if (producer_tvs.size() == 1 &&
              std::find(
                  vecotrizable_inputs_outputs.begin(),
                  vecotrizable_inputs_outputs.end(),
                  producer_tvs[0]) != vecotrizable_inputs_outputs.end()) {
            keep_unrolled.emplace(cached_input);
          }
        } else {
          keep_unrolled.emplace(cached_input);
        }
      }
    }

    // Inline output caches into outputs
    std::vector<TensorView*> compute_to;
    for (auto cached_output_pair : cached_outputs) {
      auto cached_output = cached_output_pair.first;
      auto output = cached_output_pair.second;

      // If an output has multiple consumers don't process here, we want only
      // terminating outputs
      if (cached_output->uses().size() > 1) {
        continue;
      }

      auto pos_it = std::find_if(
          output->domain()->domain().begin(),
          output->domain()->domain().end(),
          [&mapped_to_trivial_reduction](IterDomain* id) {
            return id->getParallelType() == ParallelType::Unswitch ||
                id->getParallelType() == ParallelType::Unroll ||
                id->getParallelType() == ParallelType::Vectorize ||
                id->getParallelType() == ParallelType::MisalignedVectorize ||
                mapped_to_trivial_reduction.count(id);
          });
      auto pos = pos_it == output->domain()->domain().end()
          ? -1
          : std::distance(output->domain()->domain().begin(), pos_it) + 1;

      cached_output->computeAt(output, pos, ComputeAtMode::BestEffort);

      compute_to.push_back(cached_output);
      if (rparams.vectorize) {
        if (std::find(
                vecotrizable_inputs_outputs.begin(),
                vecotrizable_inputs_outputs.end(),
                output) != vecotrizable_inputs_outputs.end()) {
          keep_unrolled.emplace(output);
        }
      } else {
        keep_unrolled.emplace(output);
      }
    }

    // Before compute at-ing the internal structure, remove vectorization
    // anywhere it doesn't belong. Otherwise it will mess up our inlining. Clear
    // explicit unroll or vectorization when not for input or output GMEM
    // transfers.
    for (auto tv : ir_utils::allTvs(fusion)) {
      if (!keep_unrolled.count(tv)) {
        for (size_t i = 0; i < tv->nDims(); i++) {
          auto id = tv->axis((int)i);
          if (id->getParallelType() == ParallelType::Unroll ||
              id->getParallelType() == ParallelType::Vectorize ||
              id->getParallelType() == ParallelType::MisalignedVectorize) {
            tv->axis((int)i)->parallelize(ParallelType::Serial);
          }
        }
      }
    }

    // Make sure not to completely inline if there's trivial reductions in the
    // fusion
    auto pos_it = std::find_if(
        reference_tv->domain()->domain().begin(),
        reference_tv->domain()->domain().end(),
        [&mapped_to_trivial_reduction](IterDomain* id) {
          return mapped_to_trivial_reduction.count(id);
        });

    auto pos = pos_it == reference_tv->domain()->domain().end()
        ? -1
        : std::distance(reference_tv->domain()->domain().begin(), pos_it) + 1;

    // Compute at inputs to rfactor dimensions
    computeAtBetween(
        compute_from, rfactor_tvs, pos, ComputeAtMode::MostInlined);

    // Inline rfactor into reduction
    if (reference_tv != reduction_tv) {
      // Compute at rfactor into following reduction, keep outside first
      // reduction iter domain in the rfactor tensor view
      for (size_t i = 0; i < rfactor_tvs.size(); i++) {
        if (!rparams.reduction_unroll) {
          auto rfactor_tv = rfactor_tvs[i];
          auto rfactor_tv_dom = rfactor_tv->domain()->domain();
          auto reduction_it = std::find_if(
              rfactor_tv_dom.begin(), rfactor_tv_dom.end(), [](IterDomain* id) {
                return id->isReduction();
              });
          TORCH_INTERNAL_ASSERT(
              reduction_it != rfactor_tv_dom.end(),
              "Expected reduction axis in ",
              rfactor_tv);
          auto pos = std::distance(rfactor_tv_dom.begin(), reduction_it);
          rfactor_tv->computeWith(
              reduction_tvs[i], pos, ComputeAtMode::Standard);
        } else {
          rfactor_tvs[i]->computeWith(
              reduction_tvs[i], -1, ComputeAtMode::BestEffort);
        }
      }
    }

    // Remove anything before a reduction from compute_from
    {
      auto producers_of_reductions = DependencyCheck::getAllValsBetween(
          {fusion->inputs().begin(), fusion->inputs().end()},
          {reduction_tvs.begin(), reduction_tvs.end()});

      auto producer_tvs_of_reductions =
          ir_utils::filterByType<TensorView>(producers_of_reductions);
      compute_from.erase(
          std::remove_if(
              compute_from.begin(),
              compute_from.end(),
              [&producer_tvs_of_reductions](TensorView* compute_from_tv) {
                return std::find(
                           producer_tvs_of_reductions.begin(),
                           producer_tvs_of_reductions.end(),
                           compute_from_tv) != producer_tvs_of_reductions.end();
              }),
          compute_from.end());
    }

    // Add reduction tensor views to compute from
    compute_from.insert(
        compute_from.end(), reduction_tvs.begin(), reduction_tvs.end());

    // Compute between reductions and output caches
    computeAtBetween(
        compute_from,
        compute_to,
        -1,
        ComputeAtMode::BestEffort,
        mapped_to_trivial_reduction);

  } else {
    // Want to inline, especially backwards based on reduction_tv, otherwise
    // rfactor tv may not be inlined correctly
    auto ref_tvs = rfactor_tvs.size() ? rfactor_tvs : reduction_tvs;
    for (auto red_tv : ref_tvs) {
      auto pos_it = std::find_if(
          red_tv->domain()->domain().begin(),
          red_tv->domain()->domain().end(),
          [&mapped_to_trivial_reduction](IterDomain* id) {
            return id->getParallelType() == ParallelType::Unswitch ||
                id->getParallelType() == ParallelType::Unroll ||
                id->getParallelType() == ParallelType::Vectorize ||
                id->getParallelType() == ParallelType::MisalignedVectorize ||
                mapped_to_trivial_reduction.count(id);
          });
      auto pos = pos_it == red_tv->domain()->domain().end()
          ? -1
          : std::distance(red_tv->domain()->domain().begin(), pos_it) + 1;

      computeAtInputs(red_tv, pos, ComputeAtMode::MostInlined);
      computeWithOutputs(red_tv, pos, ComputeAtMode::BestEffort);
    }
  }
}

FindAllMappedDims::FindAllMappedDims(TensorView* from, IterDomain* id)
    : starting_tv(from), starting_id(id) {
  std::deque<TensorView*> to_visit{starting_tv};
  std::unordered_set<TensorView*> visited;
  mapped_ids.emplace(std::make_pair(starting_tv, starting_id));

  // Propagate mapping of id
  while (!to_visit.empty()) {
    auto tv = to_visit.front();
    to_visit.pop_front();

    if (!visited.emplace(tv).second) {
      continue;
    }

    auto tv_id = mapped_ids.at(tv);

    for (auto consumer_tv : ir_utils::consumerTvsOf(tv)) {
      if (visited.find(consumer_tv) != visited.end()) {
        continue;
      }

      if (mapped_ids.find(consumer_tv) != mapped_ids.end()) {
        continue;
      }

      PairwiseRootDomainMap root_map(tv, consumer_tv);
      auto p2c_map =
          root_map.mapProducerToConsumer(tv->domain(), consumer_tv->domain());

      auto c_it = p2c_map.find(tv_id);
      if (c_it != p2c_map.end()) {
        mapped_ids.emplace(std::make_pair(consumer_tv, c_it->second));
        to_visit.emplace_back(consumer_tv);
      }
    }

    for (auto producer_tv : ir_utils::producerTvsOf(tv)) {
      if (visited.find(producer_tv) != visited.end()) {
        continue;
      }

      if (mapped_ids.find(producer_tv) != mapped_ids.end()) {
        continue;
      }

      PairwiseRootDomainMap root_map(producer_tv, tv);
      auto c2p_map =
          root_map.mapConsumerToProducer(tv->domain(), producer_tv->domain());
      auto p_it = c2p_map.find(tv_id);
      if (p_it != c2p_map.end()) {
        mapped_ids.emplace(std::make_pair(producer_tv, p_it->second));
        to_visit.emplace_back(producer_tv);
      }
    }
  }
}

std::unordered_set<IterDomain*> FindAllMappedDims::from(
    TensorView* tv,
    IterDomain* id) {
  TORCH_INTERNAL_ASSERT(
      std::find_if(
          tv->getRootDomain().begin(),
          tv->getRootDomain().end(),
          [&id](IterDomain* root_id) { return root_id == id; }) !=
          tv->getRootDomain().end(),
      "Tried to map out ",
      id,
      " from TV ",
      tv,
      " to the rest of the fusion, but id does not belong to this tv.");

  FindAllMappedDims mapped_dims(tv, id);

  std::unordered_set<IterDomain*> mapped_id_set;
  for (auto entry : mapped_dims.mapped_ids) {
    mapped_id_set.emplace(entry.second);
  }
  return mapped_id_set;
}

bool shouldVectorize(
    TensorView* tv,
    std::unordered_set<IterDomain*> vector_dims) {
  const auto& root_dom = TensorDomain::noBroadcasts(
      TensorDomain::noReductions(tv->getRootDomain()));

  // Don't vectorize 0-dim tensors
  if (root_dom.size() == 0) {
    return false;
  }

  auto inner_most_dim = root_dom[root_dom.size() - 1];

  // Make sure inner most dimension is in the vector_dim set
  if (vector_dims.count(inner_most_dim) == 0) {
    return false;
  }

  auto root_pos_it = std::find_if(
      tv->getRootDomain().begin(),
      tv->getRootDomain().end(),
      [&inner_most_dim](IterDomain* id) { return inner_most_dim == id; });

  TORCH_INTERNAL_ASSERT(root_pos_it != tv->getRootDomain().end());
  auto inner_most_dim_pos =
      std::distance(tv->getRootDomain().begin(), root_pos_it);

  const auto& contiguity = tv->domain()->contiguity();

  TORCH_INTERNAL_ASSERT(contiguity.size() == tv->getRootDomain().size());

  // Don't vectorize if inner most dimension is not contiguous
  if (!contiguity[inner_most_dim_pos]) {
    return false;
  }

  return true;
}

std::vector<TensorView*> getVectorizableInputsOutputs(
    TensorView* reference_tv) {
  if (reference_tv->nDims() == 0) {
    return {};
  }

  IterDomain* inner_most_id = nullptr;
  for (auto it = reference_tv->getRootDomain().rbegin();
       it != reference_tv->getRootDomain().rend();
       it++) {
    if ((*it)->isReduction() && reference_tv->isFusionInput()) {
      continue;
    }
    if ((*it)->isBroadcast() && inner_most_id == nullptr) {
      inner_most_id = *it;
    }
    inner_most_id = *it;
    break;
  }

  if (inner_most_id == nullptr) {
    return {};
  }

  auto vectorizable_dims = FindAllMappedDims::from(reference_tv, inner_most_id);

  std::vector<TensorView*> vectorizable_tensors;

  for (auto input_tv :
       ir_utils::filterByType<TensorView>(reference_tv->fusion()->inputs())) {
    if (shouldVectorize(input_tv, vectorizable_dims)) {
      vectorizable_tensors.push_back(input_tv);
    }
  }

  for (auto output_tv :
       ir_utils::filterByType<TensorView>(reference_tv->fusion()->outputs())) {
    if (shouldVectorize(output_tv, vectorizable_dims)) {
      vectorizable_tensors.push_back(output_tv);
    }
  }

  return vectorizable_tensors;
}

std::vector<int64_t> mappedInputsOutputs(TensorView* reference_tv) {
  auto fusion = reference_tv->fusion();
  FusionGuard fg(fusion);

  // All input or output tensor views
  std::vector<TensorView*> in_out_tvs;
  {
    auto inp_tvs = ir_utils::filterByType<TensorView>(fusion->inputs());
    in_out_tvs.insert(in_out_tvs.end(), inp_tvs.begin(), inp_tvs.end());
    auto out_tvs = ir_utils::filterByType<TensorView>(fusion->outputs());
    in_out_tvs.insert(in_out_tvs.end(), out_tvs.begin(), out_tvs.end());
  }

  // Shouldn't matter which compute at map we use
  auto ca_index_map = ComputeAtMap(ComputeAtMap::MappingMode::INDEX);
  ca_index_map.build(fusion);

  auto ref_root_domain = reference_tv->getMaybeRFactorDomain();
  std::vector<int64_t> mapping_count(ref_root_domain.size(), 0);

  // Map all inputs and output domains to reference tv domains
  for (auto in_out_tv : in_out_tvs) {
    auto in_out_tv_domain = in_out_tv->getRootDomain();
    auto in_out_tv_domain_list = std::list<IterDomain*>(
        in_out_tv_domain.begin(), in_out_tv_domain.end());
    auto in_out_dtype_size = dataTypeSize(in_out_tv->getDataType().value());

    for (size_t ref_i = 0; ref_i < ref_root_domain.size(); ref_i++) {
      auto ref_id = ref_root_domain[ref_i];

      // If reference id is broadcast or reduction
      if (ref_id->isBroadcast() || ref_id->isReduction()) {
        continue;
      }
      auto map_it = std::find_if(
          in_out_tv_domain_list.begin(),
          in_out_tv_domain_list.end(),
          [&ref_id, &ca_index_map](IterDomain* in_out_tv_id) {
            return ca_index_map.areMapped(in_out_tv_id, ref_id);
          });

      if (map_it == in_out_tv_domain_list.end()) {
        continue;
      }

      // If input/output id is broadcast or reduction
      if ((*map_it)->isBroadcast() || (*map_it)->isReduction()) {
        continue;
      }

      mapping_count[ref_i] = mapping_count[ref_i] + (int64_t)in_out_dtype_size;
      in_out_tv_domain_list.erase(map_it);
    }
  }
  return mapping_count;
}

} // namespace scheduler_utils
} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
