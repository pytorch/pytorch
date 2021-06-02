#include <torch/csrc/jit/codegen/cuda/scheduler/registry.h>
#include <torch/csrc/jit/codegen/cuda/scheduler/utils.h>

#include <torch/csrc/jit/codegen/cuda/arith.h>
#include <torch/csrc/jit/codegen/cuda/compute_at_map.h>
#include <torch/csrc/jit/codegen/cuda/expr_evaluator.h>
#include <torch/csrc/jit/codegen/cuda/ir_iostream.h>
#include <torch/csrc/jit/codegen/cuda/ir_utils.h>
#include <torch/csrc/jit/codegen/cuda/iter_visitor.h>
#include <torch/csrc/jit/codegen/cuda/root_domain_map.h>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {
namespace scheduler_utils {
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
  if (tv->nDims() == 0) {
    return 0;
  }
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
  if (tv->definition() == nullptr) {
    return {};
  }
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

std::vector<TensorView*> inputTvsOf(TensorView* tv) {
  return inputTvsOf(std::vector<TensorView*>{tv});
}

std::vector<TensorView*> outputTvsOf(TensorView* tv) {
  return outputTvsOf(std::vector<TensorView*>{tv});
}

std::vector<TensorView*> inputTvsOf(std::vector<TensorView*> tvs) {
  auto inp_vals = IterVisitor::getInputsTo({tvs.begin(), tvs.end()});
  auto filtered = ir_utils::filterByType<TensorView>(inp_vals);
  std::vector<TensorView*> inp_tvs(filtered.begin(), filtered.end());
  return uniqueEntries(inp_tvs);
}

std::vector<TensorView*> outputTvsOf(std::vector<TensorView*> tvs) {
  auto out_vals = DependencyCheck::getAllOutputsOf({tvs.begin(), tvs.end()});
  auto filtered = ir_utils::filterByType<TensorView>(out_vals);
  std::vector<TensorView*> out_tvs(filtered.begin(), filtered.end());
  return uniqueEntries(out_tvs);
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
  for (auto inp_tv : inputTvsOf(consumer)) {
    inp_tv->computeAt(consumer, pos, mode);
  }
}

void computeWithOutputs(TensorView* producer, int pos, ComputeAtMode mode) {
  for (auto out_tv : outputTvsOf(producer)) {
    producer->computeWith(out_tv, pos, mode);
  }
}

std::vector<TensorView*> allTvs(Fusion* fusion) {
  auto used_vals = fusion->usedMathVals();
  auto used_tvs = ir_utils::filterByType<TensorView>(used_vals);
  return uniqueEntries({used_tvs.begin(), used_tvs.end()});
}

PersistentBufferInfo persistentBuffers(Fusion* fusion) {
  FusionGuard fg(fusion);

  PersistentBufferInfo info;

  ComputeAtRootDomainMap root_map;
  root_map.build();

  auto all_tvs = allTvs(fusion);

  for (auto producer : all_tvs) {
    bool mappable = true;
    auto consumers = consumerTvsOf(producer);
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
    ExpressionEvaluator& evaluator,
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

    auto inferred_val = evaluator.evaluate(id->extent());
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

bool registerPersistentBufferCheck(
    Fusion* fusion,
    SchedulerRuntimeInfo& runtime_info) {
  auto persistent_buffers = scheduler_utils::persistentBuffers(fusion);
  bool fits_register_persistence = true;

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
    auto consumers_of_tv = scheduler_utils::consumerTvsOf(tv);
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

  constexpr int64_t register_file_size = 256 * 1024;
  // Don't use more than 75% of register file for persistent buffers
  if (max_persistence_size * 4 > register_file_size * 3) {
    fits_register_persistence = false;
  }

  return fits_register_persistence;
}

} // namespace scheduler_utils
} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
