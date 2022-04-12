#include <torch/csrc/jit/codegen/cuda/scheduler/reduction_utils.h>

#include <torch/csrc/jit/codegen/cuda/expr_evaluator.h>
#include <torch/csrc/jit/codegen/cuda/ir_cloner.h>
#include <torch/csrc/jit/codegen/cuda/ir_utils.h>
#include <torch/csrc/jit/codegen/cuda/scheduler/registry.h>
#include <torch/csrc/jit/codegen/cuda/scheduler/utils.h>
#include <torch/csrc/jit/codegen/cuda/transform_replay.h>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

namespace reduction_scheduler_utils {

TensorView* scheduleReductionTV(
    const ReductionParams& rparams,
    TensorView* reduction_tv,
    bool has_iter_axis) {
  // Outer and inner reduction axis is relative. Outer reduce axis is only valid
  // in 3D scheduling. Otherwise inner_reduce_axis is the only reduction axis.
  // Inner here though is only relative to the other axis. When
  // rparams.fastest_dim == false, the reduction axis is logically outside the
  // iteration axis.
  const int iter_axis = 0;
  const int outer_reduce_axis = rparams.schedule_3D ? 1 : 0;
  const int inner_reduce_axis = rparams.schedule_3D ? 2 : has_iter_axis ? 1 : 0;

  TORCH_INTERNAL_ASSERT(
      (int)reduction_tv->nDims() >
          std::max(iter_axis, std::max(outer_reduce_axis, inner_reduce_axis)),
      "Issue in scheduling reduction tv, expecting >",
      std::max(iter_axis, std::max(outer_reduce_axis, inner_reduce_axis)),
      " dimensions, but found ",
      reduction_tv->nDims());

  TORCH_INTERNAL_ASSERT(
      !(rparams.fastest_dim && rparams.vectorize_iter_dom),
      "Cannot vectorize iteration domain on inner reductions.");

  TORCH_INTERNAL_ASSERT(
      !(!rparams.fastest_dim && rparams.vectorize_inner_reduction),
      "Cannot vectorize reduction domain on outer reductions.");

  TORCH_INTERNAL_ASSERT(
      !(rparams.multiple_reds_per_blk && !has_iter_axis),
      "Multiple reductions requires an iter domain, but one wasn't found.");

  TORCH_INTERNAL_ASSERT(
      !(rparams.cross_grid_inner_reduction && rparams.unroll_iter_dom),
      "Unrolling on iter domain not supported with cross grid reductions.");

  TORCH_INTERNAL_ASSERT(
      !(rparams.unroll_iter_dom && !has_iter_axis),
      "Unrolling on iter domain requires an iter domain.");

  auto vectorize = [&reduction_tv](int axis, int factor) {
    reduction_tv->split(axis, factor);
    reduction_tv->axis(axis + 1)->parallelize(ParallelType::Vectorize);
  };

  auto inner_parallel = [&reduction_tv](int axis, ParallelType ptype) {
    reduction_tv->split(axis, NamedScalar::getParallelDim(ptype));
    reduction_tv->axis(axis + 1)->parallelize(ptype);
  };

  auto inner_unswitch = [&reduction_tv](int axis) {
    reduction_tv->split(axis, 1);
    reduction_tv->axis(axis + 1)->parallelize(ParallelType::Unswitch);
  };

  auto inner_unroll = [&reduction_tv](int axis, int factor) {
    reduction_tv->split(axis, factor);
    reduction_tv->axis(axis + 1)->parallelize(ParallelType::Unroll);
  };

  auto outer_parallel = [&reduction_tv](int axis, ParallelType ptype) {
    reduction_tv->split(axis, NamedScalar::getParallelDim(ptype), false);
    reduction_tv->axis(axis)->parallelize(ptype);
  };

  auto outer_unswitch = [&reduction_tv](int axis) {
    reduction_tv->split(axis, 1, false);
    reduction_tv->axis(axis)->parallelize(ParallelType::Unswitch);
  };

  auto outer_unroll = [&reduction_tv](int axis, int factor) {
    reduction_tv->split(axis, factor, false);
    reduction_tv->axis(axis)->parallelize(ParallelType::Unroll);
  };

  if (rparams.persistent_kernel) {
    // Persistent Format:
    // [Grid Split, persistent buffer, unswitch, unroll, thread dim, vectorize]
    if (rparams.vectorize_inner_reduction) {
      vectorize(inner_reduce_axis, rparams.unroll_factor_inner_reduction);
    }
    auto outer_i = inner_reduce_axis;
    if (rparams.cross_grid_inner_reduction) {
      outer_parallel(outer_i++, rparams.grid_dim_inner_reduction);
    }

    reduction_tv->split(
        outer_i++, rparams.batches_per_block_inner_reduction, false);

    outer_unswitch(outer_i++);

    if (!rparams.vectorize_inner_reduction && rparams.unroll_inner_reduction) {
      outer_unroll(outer_i++, rparams.unroll_factor_inner_reduction);
    }

    reduction_tv->axis(outer_i)->parallelize(rparams.block_dim_inner_reduction);

    if (rparams.pad_inner_reduction_to_warp) {
      reduction_tv->axis(outer_i)->padToMultipleOfWarp();
    }

  } else {
    // Non-persistent format:
    // [Grid Split, Remainder, unswitch, unroll, thread dim, vectorize]
    if (rparams.vectorize_inner_reduction) {
      vectorize(inner_reduce_axis, rparams.unroll_factor_inner_reduction);
    }

    if (rparams.cross_block_inner_reduction) {
      inner_parallel(inner_reduce_axis, rparams.block_dim_inner_reduction);
      if (rparams.pad_inner_reduction_to_warp) {
        reduction_tv->axis(inner_reduce_axis + 1)->padToMultipleOfWarp();
      }
    }

    if (!rparams.vectorize_inner_reduction && rparams.unroll_inner_reduction) {
      inner_unroll(inner_reduce_axis, rparams.unroll_factor_inner_reduction);
    }

    inner_unswitch(inner_reduce_axis);
    if (rparams.cross_grid_inner_reduction) {
      if (rparams.split_grid_dim_inner_reduction) {
        outer_parallel(inner_reduce_axis, rparams.grid_dim_inner_reduction);
      } else {
        reduction_tv->axis(inner_reduce_axis)
            ->parallelize(rparams.grid_dim_inner_reduction);
      }
    }
  }

  // Outer reduction axis
  if (rparams.schedule_3D) {
    if (rparams.persistent_kernel) {
      // Persistent Format:
      // [Grid Split, persistent buffer, unroll, thread dim]
      auto outer_i = outer_reduce_axis;
      if (rparams.cross_grid_outer_reduction) {
        outer_parallel(outer_i++, rparams.grid_dim_outer_reduction);
      }

      reduction_tv->split(
          outer_i++, rparams.batches_per_block_outer_reduction, false);

      if (rparams.unroll_outer_reduction) {
        outer_unroll(outer_i++, rparams.unroll_factor_outer_reduction);
      }

      reduction_tv->axis(outer_i)->parallelize(
          rparams.block_dim_outer_reduction);
    } else {
      // Non-persistent format:
      // [Grid Split, Remainder, unroll, thread dim]
      if (rparams.cross_block_outer_reduction) {
        inner_parallel(outer_reduce_axis, rparams.block_dim_outer_reduction);
      }

      if (rparams.unroll_outer_reduction) {
        inner_unroll(outer_reduce_axis, rparams.unroll_factor_outer_reduction);
      }

      if (rparams.cross_grid_outer_reduction) {
        outer_parallel(outer_reduce_axis, rparams.grid_dim_outer_reduction);
      }
    }
  }

  // Iteration domain
  if (has_iter_axis) {
    // [Grid Split, unswitch, unroll, thread dim, vectorize]

    if (rparams.vectorize_iter_dom) {
      vectorize(iter_axis, rparams.unroll_factor_iter_dom);
    }

    if (isParallelTypeThread(rparams.block_dim_iter_dom)) {
      inner_parallel(iter_axis, rparams.block_dim_iter_dom);
    }

    if (!rparams.vectorize_iter_dom && rparams.unroll_iter_dom) {
      inner_unroll(iter_axis, rparams.unroll_factor_iter_dom);
    }

    if (rparams.unroll_iter_dom) {
      inner_unswitch(iter_axis);
    }

    if (isParallelTypeThread(rparams.grid_dim_iter_dom)) {
      if (rparams.split_grid_dim_iter_dom) {
        outer_parallel(iter_axis, rparams.grid_dim_iter_dom);
      } else {
        reduction_tv->axis(iter_axis)->parallelize(rparams.grid_dim_iter_dom);
      }
    }
  }

  return sortAndRFactor(reduction_tv);
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
    for (const auto i : c10::irange(reference_tv->nDims())) {
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
  scheduler_utils::parallelizeAllLike(reference_tv, ir_utils::allTvs(fusion));

  // Find iter domains that are mapped to a trivial reduction, these should
  // never be inlined.
  std::unordered_set<IterDomain*> mapped_to_trivial_reduction =
      scheduler_utils::getTrivialReductionMap(fusion);

  bool unroll = rparams.unroll_inner_reduction || rparams.unroll_iter_dom;

  bool vectorize =
      rparams.vectorize_inner_reduction || rparams.vectorize_iter_dom;

  if (unroll) {
    // Inline Input caches to their consumers outside unswitched/vectorization
    // position Inline consumers of input caches to rfactor tensors

    // Mark which tensor views are actual input caches to leave vectorization on
    // them
    std::unordered_set<TensorView*> keep_unrolled;

    std::vector<TensorView*> compute_from;

    // Grab all tensor views that should be vectorized
    auto vecotrizable_inputs_outputs =
        scheduler_utils::getInputsOutputsWithInnerDim(reference_tv, true);

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

        if (vectorize) {
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
      if (vectorize) {
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
        for (const auto i : c10::irange(tv->nDims())) {
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
    scheduler_utils::computeAtBetween(
        compute_from, rfactor_tvs, pos, ComputeAtMode::MostInlined);

    // Inline rfactor into reduction
    if (reference_tv != reduction_tv) {
      // Compute at rfactor into following reduction, keep outside first
      // reduction iter domain in the rfactor tensor view
      for (const auto i : c10::irange(rfactor_tvs.size())) {
        if (rparams.unroll_iter_dom) {
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
          // I would like computeAtMode here to be Standard. However, the
          // processing of welford rfactors in compute at ends up propating
          // compute at from reduction_tv->rfactor_tv to all outputs.
          rfactor_tv->computeWith(
              reduction_tvs[i], pos, ComputeAtMode::BestEffort);
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
    scheduler_utils::computeAtBetween(
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

      scheduler_utils::computeAtInputs(red_tv, pos, ComputeAtMode::MostInlined);
      scheduler_utils::computeWithOutputs(
          red_tv, pos, ComputeAtMode::BestEffort);
    }
    // For topologies where there may not be paths to all inputs/outputs from
    // the reductions, we need to take a similar approach to the unrolled
    // version and setup of compute at from inputs->outputs that are not
    // inputs/outputs of the reductions.
    std::vector<TensorView*> compute_to;
    std::unordered_set<TensorView*> outs_of_reds;
    {
      auto outs_of_red_vec = ir_utils::outputTvsOf(ref_tvs);
      outs_of_reds = std::unordered_set<TensorView*>(
          outs_of_red_vec.begin(), outs_of_red_vec.end());
    }
    for (auto out : ir_utils::filterByType<TensorView>(fusion->outputs())) {
      // only terminating outputs
      if (out->uses().size() || outs_of_reds.find(out) != outs_of_reds.end() ||
          out->isFusionInput()) {
        continue;
      }
      compute_to.push_back(out);
    }

    std::vector<TensorView*> compute_from;
    std::unordered_set<TensorView*> inps_of_reds;
    {
      auto inps_of_red_vec = ir_utils::inputTvsOf(ref_tvs);
      inps_of_reds = std::unordered_set<TensorView*>(
          inps_of_red_vec.begin(), inps_of_red_vec.end());
    }
    for (auto inp : ir_utils::filterByType<TensorView>(fusion->inputs())) {
      if (inps_of_reds.find(inp) != inps_of_reds.end()) {
        continue;
      }
      compute_from.push_back(inp);
    }

    scheduler_utils::computeAtBetween(
        compute_from,
        compute_to,
        -1,
        ComputeAtMode::MostInlined,
        mapped_to_trivial_reduction);
  }
}

namespace {

// Convert properties of an ID to a numeric value
int idPos(const IterDomain* id) {
  int inner_most = std::numeric_limits<int>::max();
  int outer_most = std::numeric_limits<int>::min();

  // Trivial reduction
  if (id->isReduction() && id->getParallelType() == ParallelType::Serial &&
      id->extent()->isOneInt()) {
    return inner_most;
  }
  inner_most--;

  // Reduction and unrolled
  if (id->isReduction() &&
      (id->getParallelType() == ParallelType::Unroll ||
       id->getParallelType() == ParallelType::Vectorize ||
       id->getParallelType() == ParallelType::MisalignedVectorize)) {
    return inner_most;
  }
  inner_most--;

  // Reduction and constant
  if (id->isReduction() && id->extent()->isConstScalar()) {
    return inner_most;
  }
  inner_most--;

  // Reduction and unswitched
  if (id->isReduction() && id->getParallelType() == ParallelType::Unswitch) {
    return inner_most;
  }
  inner_most--;

  // Reduction and thread
  if (id->isReduction() && id->isThread()) {
    return inner_most;
  }
  inner_most--;

  // Broadcast
  if (id->isBroadcast() || id->isImplicitBroadcast()) {
    return inner_most;
  }
  inner_most--;

  // Iter and unrolled
  if (!id->isReduction() &&
      (id->getParallelType() == ParallelType::Unroll ||
       id->getParallelType() == ParallelType::Vectorize ||
       id->getParallelType() == ParallelType::MisalignedVectorize)) {
    return inner_most;
  }
  inner_most--;

  // Iter and unswitched
  if (!id->isReduction() && id->getParallelType() == ParallelType::Unswitch) {
    return inner_most;
  }
  inner_most--;

  // Reduction and non-constant
  if (id->isReduction() && !id->extent()->isConstScalar()) {
    return inner_most;
  }
  inner_most--;

  // Iter and block (outer)
  if (!id->isReduction() && id->isBlockDim()) {
    return outer_most;
  }
  outer_most++;

  // Iter and thread (outer)
  if (!id->isReduction() && id->isThreadDim()) {
    return outer_most;
  }
  outer_most++;

  // Iter and constant
  if (!id->isReduction() && id->extent()->isConstScalar()) {
    return outer_most;
  }
  outer_most++;

  // Iter and non-constant
  if (!id->isReduction() && !id->extent()->isConstScalar()) {
    return outer_most;
  }
  outer_most++;

  return 0;
}

struct id_lt {
  // Return if id0 should be before id1
  inline bool operator()(const IterDomain* id0, const IterDomain* id1) {
    return idPos(id0) < idPos(id1);
  }
};
} // namespace

TensorView* sortAndRFactor(TensorView* reference_tv) {
  auto domain = reference_tv->domain()->domain();
  std::sort(domain.begin(), domain.end(), id_lt());
  std::unordered_map<int, int> reorder_map;
  std::unordered_map<IterDomain*, int> domain_pos;
  for (int axis_i = 0; axis_i < (int)domain.size(); axis_i++) {
    domain_pos[domain[axis_i]] = axis_i;
  }
  for (int old_i = 0; old_i < (int)reference_tv->nDims(); old_i++) {
    auto new_i_it = domain_pos.find(reference_tv->axis(old_i));
    TORCH_INTERNAL_ASSERT(
        new_i_it != domain_pos.end(),
        "Error in schedule reorder, didn't reorder all axes in provided tv.");
    auto new_i = new_i_it->second;
    reorder_map[old_i] = new_i;
  }
  reference_tv->reorder(reorder_map);

  std::vector<int> rfactor_axes;
  std::vector<int> rfactor_axes_no_unswitch;
  size_t reduction_dims = 0;
  for (int axis_i = 0; axis_i < (int)reference_tv->nDims(); axis_i++) {
    auto id = reference_tv->axis(axis_i);
    if (!id->isReduction()) {
      continue;
    }

    reduction_dims++;
    if (id->isThread()) {
      continue;
    }

    // We always want an rfactor axis because our inlining logic expects it. If
    // there's no parallelization to split out, just rfactor everything but the
    // unswitch dim.
    if (!(id->getParallelType() == ParallelType::Unswitch &&
          id->extent()->isOneInt())) {
      rfactor_axes_no_unswitch.emplace_back(axis_i);
    }
    rfactor_axes.emplace_back(axis_i);
  }

  if (reduction_dims == rfactor_axes.size()) {
    return ir_utils::rfactorHelper(reference_tv, rfactor_axes_no_unswitch);
  }

  return ir_utils::rfactorHelper(reference_tv, rfactor_axes);
}

void projectPersistentBuffers(Fusion* fusion) {
  auto persistent_info = scheduler_utils::persistentBuffers(fusion);

  // Convenience accessors
  const auto& persistent_buffers = persistent_info.persistent_buffers;
  const auto& persistent_resolution_points =
      persistent_info.persistent_buffer_resolution_points;
  const auto& projected_buffers =
      persistent_info.projectable_persistent_buffers;

  TORCH_INTERNAL_ASSERT(persistent_buffers.size() == persistent_buffers.size());

  // Iterate through projected buffers, tracking which index it corresponds too
  // since there's a resolution point entry for every buffer.
  for (auto buffer_i : c10::irange(persistent_buffers.size())) {
    auto buffer = persistent_buffers[buffer_i];
    if (std::find(projected_buffers.begin(), projected_buffers.end(), buffer) ==
        projected_buffers.end()) {
      continue;
    }

    auto resolution_points = persistent_resolution_points[buffer_i];

    std::vector<Val*> persistent_use_of_buffer;

    // Go through the resolution points one by one. Resolution points are points
    // in which the reduction branch meets the residual branch. These are points
    // where the persitent buffer may no longer be needed (one point could be
    // after another, and the buffer would be needed until the last resolution
    // points)
    for (auto resolution_point : resolution_points) {
      // Need to go through all paths from the persistent buffer to the
      // resolution point
      auto chains_to_resolution =
          DependencyCheck::getAllDependencyChains(buffer, resolution_point);
      for (auto chain : chains_to_resolution) {
        auto tv_chain = ir_utils::filterByType<TensorView>(chain);

        // To move the persistent buffers to the inputs, we need to recompute
        // the persistent buffer for all branches that don't go through a
        // reduction. If there's a reduction on the current path between the
        // persistent buffer and resolution, continue, there's no need to
        // replicate this use.
        if (std::any_of(tv_chain.begin(), tv_chain.end(), [](TensorView* tv) {
              return tv->hasReduction();
            })) {
          continue;
        }

        // Grab use of the buffer, chain[0] is the persistent buffer, chain[1]
        // is its first use.
        auto use = chain[1];

        // Only grab unique uses, a persistent buffer could be used multiple
        // times in the same expression.
        if (std::find(
                persistent_use_of_buffer.begin(),
                persistent_use_of_buffer.end(),
                use) != persistent_use_of_buffer.end()) {
          continue;
        }
        persistent_use_of_buffer.emplace_back(use);
      }

      // For all uses that do not go towards the reduction operations in the
      // persistent section of the graph, recompute the persistent buffer.
      for (auto use : persistent_use_of_buffer) {
        TORCH_INTERNAL_ASSERT(use->definition() != nullptr);
        auto buffer_replicate = RecomputeTv::recompute(buffer);
        ir_utils::replaceValInExpr(use->definition(), buffer, buffer_replicate);
      }
    }
  }
}

} // namespace reduction_scheduler_utils
} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
