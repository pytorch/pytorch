#include <torch/csrc/jit/codegen/cuda/scheduler/registry.h>
#include <torch/csrc/jit/codegen/cuda/scheduler/utils.h>

#include <torch/csrc/jit/codegen/cuda/compute_at_map.h>
#include <torch/csrc/jit/codegen/cuda/expr_evaluator.h>
#include <torch/csrc/jit/codegen/cuda/instrumentation.h>
#include <torch/csrc/jit/codegen/cuda/ir_utils.h>
#include <torch/csrc/jit/codegen/cuda/root_domain_map.h>
#include <torch/csrc/jit/codegen/cuda/scheduler/mma_utils.h>
#include <torch/csrc/jit/codegen/cuda/transform_replay.h>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {
namespace scheduler_utils {

// Returns number of "valid" dimensions. e.g. if tv has
// [I1, R2, I3, I4, R3{1}]
// where R3{1} is in dont_merge, resulting domain should be:
// [I1, I3*I4, R2, R3{1}] with return value 3
//
// if tv has
// [R1, I2, R3, I4, R4, R5{1}, R6{1}]
//  where R5{1} and R6{1} are in dont_merge, resulting domain should be:
// [I2*I4, R1*R3, R4, R5{1}, R6{1}]
// with return value 3
size_t merge_3d(
    TensorView* tv,
    const std::unordered_set<IterDomain*>& dont_merge) {
  bool active_is_reduction = false;
  bool first_dim = true;
  int prev_i = -1;

  for (int i = static_cast<int>(tv->nDims()) - 1; i >= 0; i--) {
    if (dont_merge.count(tv->axis(i))) {
      continue;
    }

    if (first_dim) {
      active_is_reduction = tv->axis(i)->isReduction();
      prev_i = i;
      first_dim = false;
    } else {
      if (tv->axis(i)->isReduction() != active_is_reduction) {
        break;
      }
      tv->merge(i, prev_i);
      prev_i = i;
    }
  }

  if (prev_i == -1) {
    // Zero dimensional
    return 0;
  }

  // put inner most dimension as last dimension
  tv->reorder({{prev_i, -1}});
  active_is_reduction = false;
  first_dim = true;
  prev_i = -1;

  for (int i = static_cast<int>(tv->nDims()) - 2; i >= 0; i--) {
    auto id = tv->axis(i);
    if (dont_merge.count(id)) {
      continue;
    }

    if (first_dim) {
      active_is_reduction = id->isReduction();
      prev_i = i;
      first_dim = false;
    } else if (id->isReduction() == active_is_reduction) {
      tv->merge(i, prev_i);
      prev_i = i;
    }
  }

  // put second dimension as second to last dimension
  if (prev_i == -1) {
    // One dimensional, put merged dimension as first
    tv->reorder({{-1, 0}});
    return 1;
  } else {
    // put new dimension as second to last
    tv->reorder({{prev_i, -2}});
  }

  active_is_reduction = false;
  first_dim = true;
  prev_i = -1;

  for (int i = static_cast<int>(tv->nDims()) - 3; i >= 0; i--) {
    if (dont_merge.count(tv->axis(i))) {
      continue;
    }

    if (first_dim) {
      active_is_reduction = tv->axis(i)->isReduction();
      prev_i = i;
      first_dim = false;
    } else if (tv->axis(i)->isReduction() == active_is_reduction) {
      tv->merge(i, prev_i);
      prev_i = i;
    }
  }

  // put third dimension as second to last dimension
  if (prev_i == -1) {
    // Two dimensional, put merged dimensions first
    tv->reorder({{-1, 0}, {-2, 1}});
    // [outer, inner, dont_merge...]
    if (tv->axis(0)->isReduction()) {
      // put reductions as second axis
      tv->reorder({{0, 1}, {1, 0}});
    }
    return 2;
  } else {
    // put new dimension as third to last
    tv->reorder({{prev_i, -3}});
    // Stable sort to have iteration domains first, then reduction
    if (tv->axis(0)->isReduction() && !tv->axis(1)->isReduction()) {
      tv->reorder({{0, 1}, {1, 0}});
    }
    if (tv->axis(1)->isReduction() && !tv->axis(2)->isReduction()) {
      tv->reorder({{1, 2}, {2, 1}});
    }
    if (tv->axis(0)->isReduction() && !tv->axis(1)->isReduction()) {
      tv->reorder({{0, 1}, {1, 0}});
    }
    return 3;
  }
}

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
  if (prev_i != -1) {
    tv->reorder({{prev_i, 0}});
  }

  return prev_i == -1 ? 0 : num_merged + 1;
}

void parallelizeAllLike(
    TensorView* reference_tv,
    const std::vector<TensorView*>& all_tvs) {
  FusionGuard fg(reference_tv->fusion());

  // Use loop map as that is the most permissive.
  auto ca_loop_map = ComputeAtMap(ComputeAtMap::MappingMode::LOOP);
  ca_loop_map.build(FusionGuard::getCurFusion());
  for (auto id : reference_tv->domain()->domain()) {
    ca_loop_map.getConcreteMappedID(id)->parallelize(id->getParallelType());
    if (id->hasPaddingToMultipleOfWarp()) {
      ca_loop_map.getConcreteMappedID(id)->padToMultipleOfWarp(
          id->getMaybeSizeAfterPadding());
    }
  }

  for (auto tv : all_tvs) {
    if (tv->isFusionInput()) {
      continue;
    }
    for (const auto i : c10::irange(tv->domain()->domain().size())) {
      auto ca_id = ca_loop_map.getConcreteMappedID(tv->axis(i));
      tv->axis(i)->parallelize(ca_id->getParallelType());
      if (ca_id->hasPaddingToMultipleOfWarp()) {
        tv->axis(i)->padToMultipleOfWarp(ca_id->getMaybeSizeAfterPadding());
      }
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
    if (out_tv == producer) {
      continue;
    }
    producer->computeWith(out_tv, pos, mode);
  }
}

namespace {

// Find the resolution points of the persistent buffers in the provided
// persistent_buffer_info. Resolution points are identified by tracking if a
// tensor view is dependent on a reduction, or a persistent buffer. When an
// expression has inputs that are on both a reduction and persistent buffer
// path, that's a point where we may be resolving the persistent buffer. In
// other words, we know the persistent buffer has to be live at that point, but
// don't know if it has to be live after it.
//
// For example if we have:
//
// t0 = makeSymbolicTensor(2)
// t1 = set(t0)
// t2 = sum(t1, 1)
// t3 = broadcast(t2, {false, true})
// t4 = set(t1)
// t5 = add(t4, t3)
//
// In this case, t1 is the persistent buffer, that buffer is resolved at t5, so
// it needs to exist in full until t5 is "resolved". This class assumes all
// reduction patterns in the fusion are matching.
class PersistentBufferResolution : public IterVisitor {
 public:
  static std::vector<TensorView*> getResolutionPointsOf(
      Fusion* fusion,
      TensorView* persistent_buffer) {
    PersistentBufferResolution resolution(fusion, persistent_buffer);

    TORCH_INTERNAL_ASSERT(
        !resolution.resolution_points_.empty(),
        "Could not resolve persistent buffer: ",
        persistent_buffer);

    return resolution.resolution_points_;
  }

  PersistentBufferResolution() = delete;

 private:
  PersistentBufferResolution(Fusion* fusion, TensorView* persistent_buffer)
      : persistent_buffer_(persistent_buffer) {
    traverse(fusion);
  }

 private:
  void handle(Val* val) final {
    if (!val->isA<TensorView>()) {
      return;
    }
    auto tv = val->as<TensorView>();
    if (tv == persistent_buffer_) {
      persistent_buffer_hit = true;
      on_persitent_buffer_path_.emplace(tv);
      return;
    }

    if (!persistent_buffer_hit) {
      return;
    }

    if (tv->hasReduction()) {
      if (std::any_of(
              resolution_points_.begin(),
              resolution_points_.end(),
              [&tv](TensorView* resolution_point) {
                return DependencyCheck::isDependencyOf(resolution_point, tv);
              })) {
        // If already resolved, don't start a new reduction path.
        return;
      }
      on_reduction_path_.emplace(tv);
    }
  }

  void handle(Expr* expr) final {
    if (!persistent_buffer_hit) {
      return;
    }

    bool output_is_reduction =
        std::any_of(expr->outputs().begin(), expr->outputs().end(), [](Val* v) {
          if (!v->isA<TensorView>()) {
            return false;
          }
          return v->as<TensorView>()->hasReduction();
        });

    // Persistent buffers cannot be resolved on a reduction expression
    if (output_is_reduction) {
      return;
    }

    bool input_on_reduction_path = std::any_of(
        expr->inputs().begin(), expr->inputs().end(), [&](Val* inp) {
          return on_reduction_path_.count(inp);
        });

    auto input_on_persitent_buffer_path_it = std::find_if(
        expr->inputs().begin(), expr->inputs().end(), [&](Val* inp) {
          return on_persitent_buffer_path_.count(inp);
        });

    bool input_on_persistent_buffer_path =
        input_on_persitent_buffer_path_it != expr->inputs().end();

    if (input_on_reduction_path && input_on_persistent_buffer_path) {
      // Expression has inputs on both a reduction and persistent buffer path,
      // this is a resolution.
      auto out_tvs = ir_utils::filterByType<TensorView>(expr->outputs());

      // Add resolution point
      resolution_points_.insert(
          resolution_points_.end(), out_tvs.begin(), out_tvs.end());

      // Outputs are still on a persistent path
      for (auto out : expr->outputs()) {
        on_persitent_buffer_path_.emplace(out);
      }
    } else if (input_on_reduction_path) {
      // Propagate forward the reduction path
      on_reduction_path_.insert(expr->outputs().begin(), expr->outputs().end());
    } else if (input_on_persistent_buffer_path) {
      // Propagate forward the persistent path
      for (auto out : expr->outputs()) {
        on_persitent_buffer_path_.emplace(out);
      }
    }
  }

  // Don't do processing until we see the buffer we're looking for
  bool persistent_buffer_hit = false;

  // Track if key is dependent on a persistent reduction, resolves if
  // encountering a persistent buffer. For this analysis doesn't matter which
  // reduction the path is based on.
  std::unordered_set<Val*> on_reduction_path_;

  // Track if key is dependent on a persistent buffer, resolves if encountering
  // a persistent reduction or changes path if encountering another persistent
  // buffer
  std::unordered_set<Val*> on_persitent_buffer_path_;

  // Tracks where the persistent buffer (key) is resolved (values)
  std::vector<TensorView*> resolution_points_;

  const TensorView* persistent_buffer_;
};

} // namespace

PersistentBufferInfo persistentBuffers(Fusion* fusion) {
  FusionGuard fg(fusion);
  PersistentBufferInfo persistent_buffer_info;

  ComputeAtRootDomainMap root_map;
  root_map.build();

  auto all_tvs = ir_utils::allTvs(fusion);

  for (auto producer : all_tvs) {
    // Are all producer ids mappable to all consumers
    bool mappable = true;
    auto consumers = ir_utils::consumerTvsOf(producer);
    if (consumers.empty()) {
      continue;
    }

    // Track which consumers have unmappable dims from producer
    std::vector<TensorView*> unmappable_consumers;

    for (auto consumer : consumers) {
      bool consumer_mappable = true;
      auto mappable_roots =
          root_map.getMappableDims(producer->domain(), consumer->domain());

      auto p_root = producer->getMaybeRFactorDomain();

      for (auto p_root_id : p_root) {
        if (p_root_id->isReduction() || p_root_id->isBroadcast()) {
          continue;
        }
        if (!mappable_roots.count(p_root_id)) {
          mappable = false;
          consumer_mappable = false;
          persistent_buffer_info.unmappable_dims.emplace(p_root_id);
        }
      }

      if (!consumer_mappable) {
        unmappable_consumers.emplace_back(consumer);
      }
    }

    if (!mappable) {
      // If there's unmappable dims from producer to consumer, producer is a
      // persistent buffer.
      persistent_buffer_info.persistent_buffers.emplace_back(producer);
    }
  }

  // Set the persistent buffer resolution points
  persistent_buffer_info.persistent_buffer_resolution_points = {};
  for (auto buffer : persistent_buffer_info.persistent_buffers) {
    persistent_buffer_info.persistent_buffer_resolution_points.emplace_back(
        PersistentBufferResolution::getResolutionPointsOf(fusion, buffer));
  }

  // Find projectable persistent buffers
  auto reduction_tvs = getReductionTvs(fusion /*, ignore_trivial=true */);
  for (auto persistent_buffer : persistent_buffer_info.persistent_buffers) {
    // Inputs marked as persistent buffers can't be projected any further back
    if (persistent_buffer->isFusionInput()) {
      continue;
    }
    auto dep_vals = DependencyCheck::getAllValsBetween(
        {reduction_tvs.begin(), reduction_tvs.end()}, {persistent_buffer});

    // If there's a reduction between a persistent buffer and the inputs, it
    // can't be projected backwards.
    if (dep_vals.empty()) {
      persistent_buffer_info.projectable_persistent_buffers.push_back(
          persistent_buffer);
    }
  }

  // Get a list of inputs of the projectable buffers
  auto all_inputs = ir_utils::inputTvsOf(
      persistent_buffer_info.projectable_persistent_buffers);

  // Map unmappable dims to inputs, doesn't matter which compute at map used
  auto ca_index_map = ComputeAtMap(ComputeAtMap::MappingMode::INDEX);
  ca_index_map.build(fusion);

  std::unordered_set<IterDomain*> unmappable_concrete_ids;
  for (auto id : persistent_buffer_info.unmappable_dims) {
    unmappable_concrete_ids.emplace(ca_index_map.getConcreteMappedID(id));
  }

  for (auto input : all_inputs) {
    bool has_unmappable_dim = false;
    for (auto input_id : input->getMaybeRFactorDomain()) {
      auto concrete_input_id = ca_index_map.getConcreteMappedID(input_id);
      if (unmappable_concrete_ids.find(concrete_input_id) !=
          unmappable_concrete_ids.end()) {
        persistent_buffer_info.unamppable_dims_projected_to_inputs.emplace(
            input_id);
        has_unmappable_dim = true;
      }
    }
    if (has_unmappable_dim) {
      persistent_buffer_info.projectable_buffer_inputs.emplace_back(input);
    }
  }

  return persistent_buffer_info;
}

TvProperties getProperties(
    Fusion* fusion,
    SchedulerRuntimeInfo& runtime_info,
    TensorView* tv) {
  FusionGuard fg(fusion);

  TORCH_INTERNAL_ASSERT(tv != nullptr);

  auto root_dom = tv->getRootDomain();
  bool fastest_dim_reduction = true;

  // Is there a non trivial reduction on the inner most dimension or is there an
  // iteration domain.
  for (size_t i = root_dom.size(); i > 0; i--) {
    if (root_dom[i - 1]->isBroadcast() ||
        root_dom[i - 1]->isTrivialReduction()) {
      continue;
    } else if (root_dom[i - 1]->isReduction()) {
      fastest_dim_reduction = true;
      break;
    } else {
      fastest_dim_reduction = false;
      break;
    }
  }

  // Tracks the dimensionality of the problem starts on inner most dim and works
  // outward
  int64_t dimensionality = 1;
  // Initialize for dimensionality analysis
  bool cur_dim_is_reduction = fastest_dim_reduction;
  // Compute the size of the inner most dimension
  int64_t inner_most_dimension_numel = 1;

  // Start from the inner most dimension, and work outwards. If this is a 3D
  // pattern, i.e. theres a pattern like [r0, r1, i2, r3] or [i0, r1, r2, i3,
  // i4] then compute the inner most dimension to compute separately.
  for (size_t i = root_dom.size(); i > 0; i--) {
    auto id = root_dom[i - 1];
    if (id->isBroadcast() || id->isTrivialReduction()) {
      continue;
    }
    if (id->isReduction() != cur_dim_is_reduction) {
      dimensionality++;
      cur_dim_is_reduction = !cur_dim_is_reduction;
    } else if (dimensionality == 1) {
      auto inferred_val =
          runtime_info.expressionEvaluator().evaluate(id->extent());
      TORCH_INTERNAL_ASSERT(
          inferred_val.has_value(), "Error inferring reduction size.");
      inner_most_dimension_numel =
          inner_most_dimension_numel * inferred_val.value();
    }
  }

  // Non reduction element count
  int64_t total_iteration_numel = 1;
  // Reduction element count
  int64_t total_reduction_numel = 1;

  for (auto id : tv->getRootDomain()) {
    auto inferred_val =
        runtime_info.expressionEvaluator().evaluate(id->extent());
    TORCH_INTERNAL_ASSERT(
        inferred_val.has_value(),
        "Error inferring dimensions of reduction fusion.");
    if (id->isReduction()) {
      total_reduction_numel *= inferred_val.value();
    } else {
      total_iteration_numel *= inferred_val.value();
    }
  }

  TvProperties properties;
  properties.total_reduction_numel = total_reduction_numel;
  properties.total_iteration_numel = total_iteration_numel;
  properties.fastest_dim_reduction = fastest_dim_reduction;
  properties.inner_most_dimension_numel = inner_most_dimension_numel;
  properties.dimensionality = dimensionality;

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

        auto consumer_pos = pos_it == consumer->domain()->domain().end()
            ? pos
            : std::min(
                  (int)std::distance(
                      consumer->domain()->domain().begin(), pos_it) +
                      1,
                  (pos < 0 ? pos + (int)consumer->nDims() : pos));
        // Assume we don't want to reset computeAt on tensors that have already
        // performed it.
        producer->computeAt(consumer, consumer_pos, mode);
      }
    }
  }
}

namespace {

// Figure out which persistent buffers are active at the generation of values in
// the fusion. This will be used at runtime to compute the size and max size of
// the persistent buffers.
std::unique_ptr<HeuristicCompileTime::ScopedPersistenceBufferMap>
getScopePersistenceFactors(
    Fusion* fusion,
    PersistentBufferInfo& persistent_buffer_info) {
  auto new_persistent_factor_map_ptr =
      std::make_unique<HeuristicCompileTime::ScopedPersistenceBufferMap>();
  auto& new_persistent_factor_map = *new_persistent_factor_map_ptr;

  // Convenience accessors
  const auto& persistent_buffers = persistent_buffer_info.persistent_buffers;
  const auto& projectable_buffer_inputs =
      persistent_buffer_info.projectable_buffer_inputs;
  const auto& projectable_persistent_buffers =
      persistent_buffer_info.projectable_persistent_buffers;
  const auto& persistent_buffer_resolution_points =
      persistent_buffer_info.persistent_buffer_resolution_points;

  // Append projectable buffer inputs, going to compute size of those as well.
  auto persistent_buffers_and_inputs = persistent_buffers;
  persistent_buffers_and_inputs.insert(
      persistent_buffers_and_inputs.end(),
      projectable_buffer_inputs.begin(),
      projectable_buffer_inputs.end());

  for (auto persistent_buffer_i : c10::irange(persistent_buffers.size())) {
    auto persistent_buffer = persistent_buffers[persistent_buffer_i];
    // All expressions between tv and its resolution points must have tv's
    // persistent buffer allocated. This is an optimistic view on how many
    // registers we need allocated in the kernel, since if we ordered two
    // persistent buffers that are completely independent to somehow overlap
    // with eachothers loop nests both persistent buffers would have to be
    // allocated at the same time even though this function would assume they
    // don't.
    //
    // Unfortunately this limitation is hard to work around as we would have
    // to actually generate the kernel before we know if it would fit
    // persistently in registers. In practice, though, this should not happen
    // as inlining loop structures where the persistent buffer is used should
    // prevent muiltiple persistent buffers from being merged togther if not
    // necessary.
    auto resolution_points =
        persistent_buffer_resolution_points[persistent_buffer_i];
    for (auto val : DependencyCheck::getAllValsBetween(
             {persistent_buffer},
             {resolution_points.begin(), resolution_points.end()})) {
      // Persistent normalization kernels imply that all persistent buffers
      // have the same dimensionality. Assume if a persistent buffer is
      // consumed by another we can alias and reuse the memory.
      if (val == persistent_buffer) {
        continue;
      }

      // All vals between resolution point and the corresponding buffer have
      // that buffer live during their generation.
      if (new_persistent_factor_map.find(val) ==
          new_persistent_factor_map.end()) {
        new_persistent_factor_map[val] =
            std::vector<bool>(persistent_buffers_and_inputs.size(), false);
      }
      new_persistent_factor_map.at(val)[persistent_buffer_i] = true;
    }
  }

  // Processing projectable persistent buffers is a little more complex, simply
  // because we have to line up inputs with their persistent buffers.

  // Offset into the bool vector
  size_t bool_vector_offset = persistent_buffers.size();
  for (auto projectable_persistent_buffer_i :
       c10::irange(projectable_persistent_buffers.size())) {
    auto projectable_persistent_buffer =
        projectable_persistent_buffers[projectable_persistent_buffer_i];
    auto inputs = ir_utils::inputTvsOf(projectable_persistent_buffer);

    for (auto input : inputs) {
      auto input_it = std::find(
          projectable_buffer_inputs.begin(),
          projectable_buffer_inputs.end(),
          input);
      // If input wasn't recorded as a projectable buffer input, then it doesn't
      // have any persistent dims, so ignore it.
      if (input_it == projectable_buffer_inputs.end()) {
        continue;
      }

      // get inuput index entry in the buffer inputs vector
      auto input_i = std::distance(projectable_buffer_inputs.begin(), input_it);

      // Get the offset in the bool vector for this input
      input_i += bool_vector_offset;

      // If we project persistence from the persistent buffers to the inputs,
      // then it would have to be active from the resolution points of the
      // persistent buffer all the way back to the projected inputs.
      auto resolution_points =
          persistent_buffer_resolution_points[projectable_persistent_buffer_i];

      for (auto val : DependencyCheck::getAllValsBetween(
               {input}, {resolution_points.begin(), resolution_points.end()})) {
        // Persistent normalization kernels imply that all persistent buffers
        // have the same dimensionality. Assume if a persistent buffer is
        // consumed by another we can alias and reuse the memory.
        if (val == input) {
          continue;
        }

        if (new_persistent_factor_map.find(val) ==
            new_persistent_factor_map.end()) {
          new_persistent_factor_map[val] =
              std::vector<bool>(persistent_buffers_and_inputs.size(), false);
        }
        new_persistent_factor_map.at(val)[input_i] = true;
      }
    }
  }
  return new_persistent_factor_map_ptr;
}

} // namespace

PersistentBufferSizeReturn persistentBufferSize(
    Fusion* fusion,
    SchedulerRuntimeInfo& runtime_info,
    PersistentBufferInfo& persistent_buffer_info,
    HeuristicSummary* data_cache) {
  FUSER_PERF_SCOPE("scheduler_utils::persistentBufferSize");

  if (persistent_buffer_info.persistent_buffers.empty()) {
    PersistentBufferSizeReturn empty_sizes;
    return empty_sizes;
  }

  // Compute size of all the buffers
  const auto& persistent_buffers = persistent_buffer_info.persistent_buffers;
  const auto& projectable_buffers =
      persistent_buffer_info.projectable_persistent_buffers;
  const auto& projectable_buffers_inputs =
      persistent_buffer_info.projectable_buffer_inputs;
  const auto& unmappable_dims = persistent_buffer_info.unmappable_dims;
  const auto& input_unmappable_dims =
      persistent_buffer_info.unamppable_dims_projected_to_inputs;

  std::vector<TensorView*> all_buffers = persistent_buffers;
  all_buffers.insert(
      all_buffers.end(),
      projectable_buffers_inputs.begin(),
      projectable_buffers_inputs.end());

  std::vector<int64_t> persistent_buffer_sizes(all_buffers.size(), -1);

  for (auto buffer_i : c10::irange(all_buffers.size())) {
    bool is_input = buffer_i >= persistent_buffers.size();
    auto buffer = all_buffers[buffer_i];

    for (auto id : buffer->getMaybeRFactorDomain()) {
      if (id->isReduction() || id->isBroadcast()) {
        continue;
      }
      // Unmappable dimensions are those that we cannot inline into other
      // tensor views. So they're the ones that need to be persistent.
      if (!is_input && !unmappable_dims.count(id)) {
        continue;
      }

      if (is_input && !input_unmappable_dims.count(id)) {
        continue;
      }

      auto id_size = runtime_info.expressionEvaluator().evaluate(id->extent());
      TORCH_INTERNAL_ASSERT(
          id_size.has_value(), "Could not infer persistent buffer size.");
      if (persistent_buffer_sizes[buffer_i] == -1) {
        persistent_buffer_sizes[buffer_i] = id_size.value();
      } else {
        persistent_buffer_sizes[buffer_i] *= id_size.value();
      }
    }

    persistent_buffer_sizes[buffer_i] = persistent_buffer_sizes[buffer_i] == -1
        ? 0
        : persistent_buffer_sizes[buffer_i] *
            dataTypeSize(buffer->getDataType().value());
  }

  // Buffers involved in normal persistence
  std::vector<bool> persistent_mask(all_buffers.size(), false);

  for (auto buffer_i : c10::irange(persistent_buffers.size())) {
    persistent_mask[buffer_i] = true;
  }

  // Buffers involved in projected to inputs
  std::vector<bool> projected_mask(all_buffers.size(), true);

  for (auto buffer_i : c10::irange(persistent_buffers.size())) {
    auto buffer = persistent_buffers[buffer_i];
    // Not a projectable buffer, or an input of a projectable buffer
    if (std::find(
            projectable_buffers.begin(), projectable_buffers.end(), buffer) !=
        projectable_buffers.end()) {
      projected_mask[buffer_i] = false;
    }
  }

  // Function to take the mask of active buffers at a val, the mask (for if this
  // is a normal persistent calculation, or a calculation projected on to the
  // input buffers), and sizes, and returns total persistent buffer size.
  auto masked_dot_product = [](const std::vector<bool>& mask0,
                               const std::vector<bool>& mask1,
                               const std::vector<int64_t>& sizes) {
    int64_t buffer_size = 0;
    TORCH_INTERNAL_ASSERT(
        mask0.size() == mask1.size() && mask0.size() == sizes.size());
    for (auto buffer_i : c10::irange(sizes.size())) {
      if (mask0[buffer_i] && mask1[buffer_i]) {
        buffer_size += sizes[buffer_i];
      }
    }
    return buffer_size;
  };

  auto persistent_buffer_info_entry =
      HeuristicSummaryEntry<HeuristicCompileTime::ScopePersistentFactorInfo>(
          data_cache, [&fusion, &persistent_buffer_info]() {
            return getScopePersistenceFactors(fusion, persistent_buffer_info);
          });

  auto& scoped_persistence_factor = persistent_buffer_info_entry.get();

  // Go through all values, compute the size of the active persistent buffers,
  // do both without and with projection
  int64_t max_persistence_size = 0;
  int64_t max_proj_persistence_size = 0;
  for (const auto& entry : scoped_persistence_factor) {
    auto active_buffers = entry.second;
    auto persistent_buffer_size = masked_dot_product(
        persistent_mask, active_buffers, persistent_buffer_sizes);
    max_persistence_size =
        std::max(max_persistence_size, persistent_buffer_size);

    auto projected_buffer_size = masked_dot_product(
        projected_mask, active_buffers, persistent_buffer_sizes);
    max_proj_persistence_size =
        std::max(max_proj_persistence_size, projected_buffer_size);
  }

  PersistentBufferSizeReturn persistent_buffer_size;
  persistent_buffer_size.persistent_buffer_size = max_persistence_size;
  persistent_buffer_size.projected_persistent_buffer_size =
      max_proj_persistence_size;
  return persistent_buffer_size;
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
    // Use the loop map as that is the most permissive
    auto ca_loop_map = ComputeAtMap(ComputeAtMap::MappingMode::LOOP);
    ca_loop_map.build(fusion);
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
                [&ca_loop_map, &id](IterDomain* trivial_id) {
                  return ca_loop_map.areMapped(id, trivial_id);
                })) {
          mapped_to_trivial_reduction.emplace(id);
        }
      }
    }
  }
  return mapped_to_trivial_reduction;
}

std::pair<bool, bool> canonicalDimReduction(
    Fusion* fusion,
    TensorView* tv,
    bool schedule_3D) {
  std::unordered_set<IterDomain*> mapped_to_trivial_reduction =
      getTrivialReductionMap(fusion);

  TORCH_INTERNAL_ASSERT(tv != nullptr);

  if (!schedule_3D) {
    // We coalesce all reduction axes to the right;
    bool has_red_axis = mergeReduction(tv, mapped_to_trivial_reduction) > 0;

    bool has_iter_axis = mergeNonReduction(tv, mapped_to_trivial_reduction) > 0;
    return {has_iter_axis, has_red_axis};
  } else {
    TORCH_INTERNAL_ASSERT(
        merge_3d(tv, mapped_to_trivial_reduction) == 3,
        "Tried 3D merge, but result is not 3D.");
    return {true, true};
  }
}

std::vector<TensorView*> getReductionTvs(Fusion* fusion, bool ignore_trivial) {
  auto all_tvs = ir_utils::allTvs(fusion);
  std::vector<TensorView*> reduction_tvs;
  for (auto tv : all_tvs) {
    if (!tv->isFusionInput() &&
        std::any_of(
            tv->domain()->domain().begin(),
            tv->domain()->domain().end(),
            [&ignore_trivial](IterDomain* id) {
              return id->isReduction() &&
                  !(ignore_trivial && id->isTrivialReduction());
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

std::vector<TensorView*> getViewTVs(Fusion* fusion) {
  std::vector<TensorView*> view_tvs;
  auto fusion_vals = fusion->usedMathVals();
  for (auto producer_tv : ir_utils::filterByType<TensorView>(fusion_vals)) {
    auto consumer_tvs = ir_utils::consumerTvsOf(producer_tv);
    for (auto consumer_tv : consumer_tvs) {
      if (consumer_tv->isDefinitionType(ExprType::ViewOp)) {
        view_tvs.push_back(consumer_tv);
      }
    }
  }
  return view_tvs;
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
    if (tv->uses().empty() || tv->isFusionOutput()) {
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
      auto cached_output = output->cache_fork();
      cached_outputs.emplace_back(std::make_pair(output, cached_output));
    } else if (unroll) {
      auto cached_output = output->cache_before();
      cached_outputs.emplace_back(std::make_pair(cached_output, output));
    }
  }
  return cached_outputs;
}

namespace {
// If this is an rfactored reduction domain, actually check the root domain,
// this is because the rfactored reduction tensorview has the vectorized
// dimension, but that means the rfactor domain could have reordered what we
// consider the "inner most" allocated position on it if we consider the rfactor
// dimension.
IterDomain* innerMostRootDim(TensorView* tv) {
  if (tv->nDims() == 0) {
    return nullptr;
  }

  IterDomain* inner_most_id = nullptr;
  auto root_domain = tv->hasReduction() && tv->hasRFactor()
      ? tv->getRootDomain()
      : tv->getMaybeRFactorDomain();

  for (auto it = root_domain.rbegin(); it != root_domain.rend(); it++) {
    if ((*it)->isReduction() && tv->isFusionInput()) {
      continue;
    }
    if ((*it)->isBroadcast()) {
      if (inner_most_id == nullptr) {
        inner_most_id = *it;
      }
      continue;
    }
    if ((*it)->isTrivialReduction()) {
      if (inner_most_id == nullptr) {
        inner_most_id = *it;
      }
      continue;
    }
    inner_most_id = *it;
    break;
  }

  return inner_most_id;
}

// Take the inner most rfactor id from innerMostRootDim and project it to the
// root domain if the provided domain is on the rfactor domain. If vectorize,
// will not project if not following the inner most path.
IterDomain* projectIdToRoot(
    TensorView* tv,
    IterDomain* reference_id,
    bool vectorize) {
  if (reference_id == nullptr) {
    return nullptr;
  }

  if (!tv->hasRFactor()) {
    return reference_id;
  }

  auto replay_exprs = StmtSort::getExprs(tv->fusion(), {reference_id}, false);
  if (replay_exprs.empty()) {
    return reference_id;
  }

  IterDomain* projected_id = reference_id;
  for (auto expr_it = replay_exprs.rbegin(); expr_it != replay_exprs.rend();
       ++expr_it) {
    auto expr = *expr_it;
    if (expr->isA<Merge>()) {
      auto merge = expr->as<Merge>();
      if (merge->out() == projected_id) {
        projected_id = merge->inner();
      }
    } else if (expr->isA<Split>()) {
      auto split = expr->as<Split>();
      if (split->inner() == projected_id) {
        projected_id = split->in();
      } else if (split->outer() == projected_id) {
        if (vectorize) {
          projected_id = nullptr;
        } else {
          projected_id = split->in();
        }
      }
    } else {
      TORCH_INTERNAL_ASSERT(
          false, "Didn't recognize the iterdomain expression: ", expr);
    }
    if (projected_id == nullptr) {
      break;
    }
  }
  return projected_id;
}
} // namespace

FindAllMappedDims::FindAllMappedDims(
    TensorView* from,
    IterDomain* id,
    bool vectorize_pass)
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

    // For producers, project to root
    tv_id = projectIdToRoot(tv, tv_id, vectorize_pass);
    // If projection fails, don't map to producers
    if (tv_id == nullptr) {
      continue;
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
    IterDomain* id,
    bool vectorize_pass) {
  auto root_domain = tv->hasReduction() && tv->hasRFactor()
      ? tv->getRootDomain()
      : tv->getMaybeRFactorDomain();

  TORCH_INTERNAL_ASSERT(
      std::find_if(
          root_domain.begin(),
          root_domain.end(),
          [&id](IterDomain* root_id) { return root_id == id; }) !=
          root_domain.end(),
      "Tried to map out ",
      id,
      " from TV ",
      tv,
      " to the rest of the fusion, but id does not belong to this tv.");

  FindAllMappedDims mapped_dims(tv, id, vectorize_pass);

  std::unordered_set<IterDomain*> mapped_id_set;
  for (auto entry : mapped_dims.mapped_ids) {
    mapped_id_set.emplace(entry.second);
  }
  return mapped_id_set;
}

bool hasInnerDim(
    TensorView* tv,
    std::unordered_set<IterDomain*> vector_dims,
    bool should_vectorize) {
  const auto& inner_most_dim = innerMostRootDim(tv);
  if (inner_most_dim == nullptr || inner_most_dim->isReduction()) {
    return false;
  }

  // Make sure inner most dimension is in the vector_dim set
  if (vector_dims.count(inner_most_dim) == 0) {
    return false;
  }

  if (!should_vectorize) {
    return true;
  }

  auto root_pos_it = std::find_if(
      tv->getMaybeRFactorDomain().begin(),
      tv->getMaybeRFactorDomain().end(),
      [&inner_most_dim](IterDomain* id) { return inner_most_dim == id; });

  TORCH_INTERNAL_ASSERT(root_pos_it != tv->getMaybeRFactorDomain().end());
  auto inner_most_dim_pos =
      std::distance(tv->getMaybeRFactorDomain().begin(), root_pos_it);

  const auto& contiguity = tv->domain()->contiguity();

  TORCH_INTERNAL_ASSERT(
      contiguity.size() == tv->getMaybeRFactorDomain().size());

  // Don't vectorize if inner most dimension is not contiguous
  if (!contiguity[inner_most_dim_pos]) {
    return false;
  }

  return true;
}

std::vector<TensorView*> getInputsOutputsWithInnerDim(
    TensorView* reference_tv,
    bool vectorize_pass) {
  auto inner_most_id = innerMostRootDim(reference_tv);

  if (inner_most_id == nullptr) {
    return {};
  }

  auto vectorizable_dims =
      FindAllMappedDims::from(reference_tv, inner_most_id, vectorize_pass);

  std::vector<TensorView*> vectorizable_tensors;

  for (auto input_tv :
       ir_utils::filterByType<TensorView>(reference_tv->fusion()->inputs())) {
    if (hasInnerDim(input_tv, vectorizable_dims, vectorize_pass)) {
      vectorizable_tensors.push_back(input_tv);
    }
  }

  for (auto output_tv :
       ir_utils::filterByType<TensorView>(reference_tv->fusion()->outputs())) {
    if (hasInnerDim(output_tv, vectorizable_dims, vectorize_pass)) {
      vectorizable_tensors.push_back(output_tv);
    }
  }

  return vectorizable_tensors;
}

std::vector<BroadcastMultiple> getBroadcastMultiples(TensorView* reference_tv) {
  auto fusion = reference_tv->fusion();
  FusionGuard fg(fusion);

  std::vector<BroadcastMultiple> multiples(
      reference_tv->getMaybeRFactorDomain().size());

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

  // Map all inputs and output domains to reference tv domains
  for (auto in_out_tv : in_out_tvs) {
    std::vector<bool> mapped_axes(ref_root_domain.size(), false);

    auto in_out_tv_domain = in_out_tv->getRootDomain();
    auto in_out_tv_domain_list = std::list<IterDomain*>(
        in_out_tv_domain.begin(), in_out_tv_domain.end());

    for (const auto ref_i : c10::irange(ref_root_domain.size())) {
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

      mapped_axes[ref_i] = true;
      in_out_tv_domain_list.erase(map_it);
    }

    // For each break point position if there an lhs or rhs multiple based on
    // this tensor add it to the global multiplier
    {
      bool rhs = false;
      bool lhs = false;
      auto dtype_size = dataTypeSize(in_out_tv->getDataType().value());
      for (size_t mapped_axes_i = 0; mapped_axes_i < mapped_axes.size();
           mapped_axes_i++) {
        auto lhs_i = mapped_axes_i;
        auto rhs_i = mapped_axes.size() - 1 - mapped_axes_i;

        if (lhs) {
          multiples[lhs_i].lhs_multiple += dtype_size;
        } else if (mapped_axes[lhs_i]) {
          lhs = true;
        }

        if (rhs || mapped_axes[rhs_i]) {
          multiples[rhs_i].rhs_multiple += dtype_size;
          rhs = true;
        }
      }
    }
  }

  return multiples;
}

namespace matmul_utils {

void scheduleWarpTileWithReduction(TensorView* tv, MatMulTileOptions tile) {
  // Assumes
  // [M, N, K]
  auto cta_tile = tile.cta_tile;
  auto warp_tile = tile.warp_tile;
  auto instruction_tile = tile.instruction_tile;

  TORCH_CHECK(
      warp_tile.k == cta_tile.k,
      "schedule warp tile: currently no support for splitting k dimension to different warps");

  mma_util::checkDimSize(
      tv, {-3, -2, -1}, {cta_tile.m, cta_tile.n, cta_tile.k});

  //       -3   -2  -1
  //[...    M,   N,  K]

  // Distribute warp tile:
  tv->split(-3, warp_tile.m);
  tv->split(-2, warp_tile.n);

  //  -5   -4   -3   -2   -1
  // [Mwo  Mw  Nwo   Nw   K]
  tv->split(-4, instruction_tile.m);
  tv->split(-2, instruction_tile.n);
  tv->split(-1, instruction_tile.k);

  //   -8  -7 -6 -5 -4 -3 -2 -1
  // [Mwo Mw Mi Nwo Nw Ni Ko Ki]

  tv->reorder({{-7, -5}, {-6, -3}, {-5, -7}, {-3, -2}, {-2, -6}});

  //   -8  -7  -6 -5 -4 -3 -2 -1
  // [Mwo  Nwo Ko Mw Nw Mi Ni Ki]
}

void scheduleWarpTileWithNoReduction(TensorView* tv, MatMulTileOptions tile) {
  // Assumes
  // [M, N, K]
  auto cta_tile = tile.cta_tile;
  auto warp_tile = tile.warp_tile;
  auto instruction_tile = tile.instruction_tile;

  mma_util::checkDimSize(tv, {-2, -1}, {cta_tile.m, cta_tile.n});

  //        -2  -1
  //[...    M,   N]

  // Distribute warp tile:
  tv->split(-2, warp_tile.m);
  tv->split(-1, warp_tile.n);

  //  -4   -3   -2   -1
  // [Mwo  Mw  Nwo   Nw ]
  tv->split(-3, instruction_tile.m);
  tv->split(-1, instruction_tile.n);

  //  -6 -5  -4 -3 -2 -1
  // [Mwo Mw Mi Nwo Nw Ni]

  tv->reorder({{-5, -4}, {-4, -2}, {-3, -5}, {-2, -3}});

  //  -6   -5  -4 -3 -2 -1
  // [Mwo  Nwo Mw Nw Mi Ni]
}

//! Split the innermost dim to a vectorized load
void scheduleContiguousVectorLoad(
    TensorView* tv,
    MatMulTileOptions tile,
    int vector_word) {
  auto warp_dims = tile.cta_tile / tile.warp_tile;
  int num_of_thread = warp_dims.m * warp_dims.n * warp_dims.k * 32;

  tv->split(-1, num_of_thread * vector_word);
  tv->split(-1, vector_word);
  // [..., thread, vec]
  // distribute to warp:
  tv->split(-2, 32);
  tv->split(-3, warp_dims.n * warp_dims.k);

  tv->axis(-1)->parallelize(ParallelType::Vectorize);
  tv->axis(-2)->parallelize(ParallelType::TIDx);
  tv->axis(-3)->parallelize(ParallelType::TIDy);
  tv->axis(-4)->parallelize(ParallelType::TIDz);
}

} // namespace matmul_utils

} // namespace scheduler_utils
} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
