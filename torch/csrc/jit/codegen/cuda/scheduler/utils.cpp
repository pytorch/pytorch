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

void computeWithOutputs(
    TensorView* producer,
    int pos,
    std::unordered_set<TensorView*> tv_filter,
    ComputeAtMode mode) {
  for (auto out_tv : ir_utils::outputTvsOf(producer)) {
    if (tv_filter.count(out_tv)) {
      producer->computeWith(out_tv, pos, mode);
    }
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
    ComputeAtMode mode) {
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

        // Assume we don't want to reset computeAt on tensors that have already
        // performed it.
        producer->computeAt(consumer, pos, mode);
      }
    }
  }
}

int64_t persistentBufferSize(
    Fusion* fusion,
    SchedulerRuntimeInfo& runtime_info) {
  auto persistent_buffers = scheduler_utils::persistentBuffers(fusion);

  if (persistent_buffers.buffers.empty()) {
    return true;
  }

  int64_t persistent_buffer_size = 0;
  // Measure at each output how much persistent memory is being used
  std::unordered_map<Val*, int64_t> scoped_persistence;

  for (auto tv : persistent_buffers.buffers) {
    int64_t tv_persistent_numel = -1;
    for (auto id : tv->getMaybeRFactorDomain()) {
      if (id->isReduction()) {
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

      if (scoped_persistence.find(val) != scoped_persistence.end()) {
        scoped_persistence.at(val) += persistent_buffer_size;
      } else {
        scoped_persistence[val] = persistent_buffer_size;
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

} // namespace scheduler_utils
} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
