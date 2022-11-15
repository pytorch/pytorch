#include <torch/csrc/jit/codegen/cuda/scheduler/registry.h>
#include <torch/csrc/jit/codegen/cuda/scheduler/utils.h>
#include <torch/csrc/jit/codegen/cuda/scheduler/vectorize_helper.h>

#include <torch/csrc/jit/codegen/cuda/contiguity.h>
#include <torch/csrc/jit/codegen/cuda/expr_evaluator.h>
#include <torch/csrc/jit/codegen/cuda/instrumentation.h>
#include <torch/csrc/jit/codegen/cuda/ir_utils.h>
#include <torch/csrc/jit/codegen/cuda/root_domain_map.h>
#include <torch/csrc/jit/codegen/cuda/scheduler/mma_utils.h>
#include <torch/csrc/jit/codegen/cuda/transform_replay.h>

#include <algorithm>

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
size_t merge_3d(TensorView* tv) {
  bool active_is_reduction = false;
  bool first_dim = true;
  int prev_i = -1;

  for (int i = static_cast<int>(tv->nDims()) - 1; i >= 0; i--) {
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

void splitDims(
    TensorView* tv,
    std::vector<std::pair<size_t, size_t>> to_split, // (dim, size)
    std::vector<size_t>& to_update) {
  std::stable_sort(
      to_split.begin(),
      to_split.end(),
      [](const std::pair<size_t, size_t>& p1,
         const std::pair<size_t, size_t>& p2) { return p1.first < p2.first; });
  size_t dim_offset = 0;
  size_t pending_dim_offset = 0;
  int64_t prev_dim = -1;
  for (auto entry : to_split) {
    size_t dim = entry.first;
    size_t size = entry.second;
    if (dim != prev_dim) {
      dim_offset += pending_dim_offset;
      pending_dim_offset = 0;
    }
    size_t actual_dim = dim_offset + dim;
    tv->split(actual_dim, size);
    pending_dim_offset++;
    for (auto& i : to_update) {
      if (i > actual_dim) {
        i++;
      }
    }
    prev_dim = dim;
  }
}

c10::optional<size_t> mergeDims(
    TensorView* tv,
    std::vector<size_t> to_merge,
    std::vector<size_t>& to_update) {
  if (to_merge.empty()) {
    return c10::nullopt;
  }
  if (to_merge.size() == 1) {
    return to_merge[0];
  }
  std::sort(to_merge.begin(), to_merge.end());
  size_t left = to_merge[0];
  int64_t offset = 0;
  for (auto right = to_merge.begin() + 1; right != to_merge.end(); right++) {
    auto actual_right = offset-- + *right;
    tv->merge(left, actual_right);
    for (auto& i : to_update) {
      if (i == actual_right) {
        i = left;
      } else if (i > actual_right) {
        i--;
      }
    }
  }
  return left;
}

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
  if (prev_i != 0) {
    tv->reorder({{prev_i, 0}});
  }

  return prev_i == -1 ? 0 : num_merged + 1;
}

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
  if (prev_i != -1) {
    tv->reorder({{prev_i, 0}});
  }

  return prev_i == -1 ? 0 : num_merged + 1;
}

void parallelizeAllLike(
    TensorView* reference_tv,
    int64_t pos,
    std::vector<TensorView*> selected_tvs,
    const std::unordered_set<ParallelType>& selected_parallel_types,
    bool propagate_padding) {
  FusionGuard fg(reference_tv->fusion());

  if (pos < 0) {
    pos += reference_tv->nDims() + 1;
  }
  TORCH_CHECK(
      pos >= 0 && pos <= reference_tv->nDims(),
      "parallelizeAllLike called on an position outside valid range.");

  std::unordered_map<IterDomain*, IterDomain*> concrete_to_reference_map;

  auto ca_map = ComputeAtMap(FusionGuard::getCurFusion());

  const auto& reference_dom = reference_tv->domain()->domain();
  for (auto it = reference_dom.begin(); it != reference_dom.begin() + pos;
       it++) {
    auto ca_id = ca_map.getConcreteMappedID(*it, IdMappingMode::PERMISSIVE);
    concrete_to_reference_map[ca_id] = *it;
  }

  if (selected_tvs.empty()) {
    selected_tvs = ir_utils::allTvs(reference_tv->fusion());
  }
  for (auto tv : selected_tvs) {
    if (tv->isFusionInput()) {
      continue;
    }
    for (const auto i : c10::irange(tv->domain()->domain().size())) {
      auto ca_id =
          ca_map.getConcreteMappedID(tv->axis(i), IdMappingMode::PERMISSIVE);
      if (concrete_to_reference_map.count(ca_id) > 0) {
        auto reference_id = concrete_to_reference_map.at(ca_id);
        auto reference_parallel_type = reference_id->getParallelType();
        if (selected_parallel_types.empty() ||
            selected_parallel_types.count(reference_parallel_type)) {
          tv->axis(i)->parallelize(reference_parallel_type);
        }
        if (propagate_padding) {
          if (reference_id->hasPaddingToMultipleOfWarp()) {
            tv->axis(i)->padToMultipleOfWarp(
                reference_id->getMaybeSizeAfterPadding());
          }
        }
      }
    }
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
      if (dynamic_cast<SelectOp*>(consumer->definition())) {
        continue;
      }
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
  auto reduction_tvs = getReductionTvs(fusion);
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
  auto ca_map = ComputeAtMap(fusion);

  std::unordered_set<IterDomain*> unmappable_concrete_ids;
  for (auto id : persistent_buffer_info.unmappable_dims) {
    unmappable_concrete_ids.emplace(
        ca_map.getConcreteMappedID(id, IdMappingMode::EXACT));
  }

  for (auto input : all_inputs) {
    bool has_unmappable_dim = false;
    for (auto input_id : input->getMaybeRFactorDomain()) {
      auto concrete_input_id =
          ca_map.getConcreteMappedID(input_id, IdMappingMode::EXACT);
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
    if (root_dom[i - 1]->isBroadcast()) {
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
  int64_t inner_most_dimension_ndims = 0;

  // Start from the inner most dimension, and work outwards. If this is a 3D
  // pattern, i.e. theres a pattern like [r0, r1, i2, r3] or [i0, r1, r2, i3,
  // i4] then compute the inner most dimension to compute separately.
  for (size_t i = root_dom.size(); i > 0; i--) {
    auto id = root_dom[i - 1];
    if (id->isBroadcast()) {
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
          inner_most_dimension_numel * inferred_val->as<int64_t>();
      inner_most_dimension_ndims++;
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
      total_reduction_numel *= inferred_val->as<int64_t>();
    } else {
      total_iteration_numel *= inferred_val->as<int64_t>();
    }
  }

  TvProperties properties;
  properties.total_reduction_numel = total_reduction_numel;
  properties.total_iteration_numel = total_iteration_numel;
  properties.fastest_dim_reduction = fastest_dim_reduction;
  properties.inner_most_dimension_numel = inner_most_dimension_numel;
  properties.inner_most_dimension_ndims = inner_most_dimension_ndims;
  properties.dimensionality = dimensionality;

  return properties;
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
        persistent_buffer_sizes[buffer_i] = id_size->as<int64_t>();
      } else {
        persistent_buffer_sizes[buffer_i] *= id_size->as<int64_t>();
      }
    }

    persistent_buffer_sizes[buffer_i] = persistent_buffer_sizes[buffer_i] == -1
        ? 0
        : persistent_buffer_sizes[buffer_i] *
            dataTypeSize(
                buffer->getDataType().value(),
                indexModeToDtype(runtime_info.getIndexMode()));
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

std::pair<bool, bool> canonicalDimReduction(
    Fusion* fusion,
    TensorView* tv,
    bool schedule_3D) {
  TORCH_INTERNAL_ASSERT(tv != nullptr);

  if (!schedule_3D) {
    // We coalesce all reduction axes to the right;
    bool has_red_axis = mergeReduction(tv) > 0;

    bool has_iter_axis = mergeNonReduction(tv) > 0;
    return {has_iter_axis, has_red_axis};
  } else {
    TORCH_INTERNAL_ASSERT(
        merge_3d(tv) == 3, "Tried 3D merge, but result is not 3D.");
    return {true, true};
  }
}

std::vector<TensorView*> getReductionTvs(Fusion* fusion) {
  auto all_tvs = ir_utils::allTvs(fusion);
  std::vector<TensorView*> reduction_tvs;
  for (auto tv : all_tvs) {
    if (!tv->isFusionInput() &&
        std::any_of(
            tv->domain()->domain().begin(),
            tv->domain()->domain().end(),
            [](IterDomain* id) { return id->isReduction(); })) {
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
    if (tv->uses().empty() || tv->isFusionOutput() ||
        ir_utils::isSelectInput(tv)) {
      // Right now, tensors that are input to the select op can't be cached as
      // they must be in global memory.
      continue;
    }
    auto cached_tv = tv->cacheAfter();
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
  // For intermediate outputs, apply cacheFork
  for (auto output : ir_utils::filterByType<TensorView>(fusion->outputs())) {
    if (output->definition() == nullptr) {
      continue;
    }
    if (!output->uses().empty()) {
      output = output->cacheFork();
    }
    // We shouldn't necessarily need to fork and cache for unrolling, but
    // compute at best effort replay doesn't look at multiple outputs to limit
    // itself by, so to make sure vectorization is done correctly we fork and
    // cache. This is partially a compute at issue, but even with that fixed,
    // we'd likely want to cache a forked output to make sure our inlining
    // strategy is optimal.
    if (unroll) {
      auto cached_output = output->cacheBefore();
      cached_outputs.emplace_back(std::make_pair(cached_output, output));
    }
  }
  return cached_outputs;
}

namespace {

// Take the inner most rfactor id from innerMostRootDim and project it to the
// root domain if the provided domain is on the rfactor domain. If vectorize,
// will not project if not following the inner most path.
IterDomain* projectIdToRoot(
    TensorView* tv,
    IterDomain* reference_id,
    bool inner_only) {
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
        if (!merge->inner()->isBroadcast()) {
          projected_id = merge->inner();
        } else {
          projected_id = merge->outer();
        }
      }
    } else if (expr->isA<Split>()) {
      auto split = expr->as<Split>();
      if (split->inner() == projected_id) {
        projected_id = split->in();
      } else if (split->outer() == projected_id) {
        if (inner_only) {
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

// Take the inner most root id from innerMostRootDim and project it to the
// rfactor domain if the provided domain is on the rfactor domain. If vectorize,
// will not project if not following the inner most path.
IterDomain* projectIdToRFactor(
    TensorView* tv,
    IterDomain* reference_id,
    bool inner_only) {
  if (reference_id == nullptr) {
    return nullptr;
  }

  if (!tv->hasRFactor()) {
    return reference_id;
  }

  auto replay_exprs = StmtSort::getExprs(
      tv->fusion(),
      {tv->getRFactorDomain().begin(), tv->getRFactorDomain().end()},
      false);
  if (replay_exprs.empty()) {
    return reference_id;
  }

  IterDomain* projected_id = reference_id;
  for (auto expr_it = replay_exprs.begin(); expr_it != replay_exprs.end();
       ++expr_it) {
    auto expr = *expr_it;
    if (expr->isA<Merge>()) {
      auto merge = expr->as<Merge>();
      if (merge->inner() == projected_id) {
        projected_id = merge->out();
      } else if (merge->outer() == projected_id) {
        if (merge->inner()->isBroadcast() || !inner_only) {
          projected_id = merge->out();
        } else {
          projected_id = nullptr;
        }
      }
    } else if (expr->isA<Split>()) {
      auto split = expr->as<Split>();
      if (split->in() == projected_id) {
        projected_id = split->inner();
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

IterDomain* innerMostRootDim(TensorView* tv) {
  // This is backwards from how we normally think about grabbing root dimensions
  // to process. If we're in a reduction scheduler and we're using the rfactored
  // reduction tensor view, we don't care about the rfactor domain, we care
  // about the root domain because we're looking to vectorize the reads (input
  // tensor views). Otherwise we do want the rfactor domain. So this is the
  // reverse of our typical check, we actually want to selectively ignore the
  // rfactor domain.
  const auto& root_domain = tv->hasReduction() && tv->hasRFactor()
      ? tv->getRootDomain()
      : tv->getMaybeRFactorDomain();

  if (tv->nDims() == 0) {
    return nullptr;
  }

  IterDomain* inner_most_id = nullptr;

  for (auto it = root_domain.rbegin(); it != root_domain.rend(); it++) {
    // If we're looking at a reduction domain on an input because of
    // segmentation we don't want to consider those reduction domains as a
    // vectorization opportunity. If we're looking at a reduction reference
    // tensor we want to consider the reduction iteration domains as domains we
    // can vectorize on.
    if ((*it)->isReduction() && tv->isFusionInput()) {
      continue;
    }
    if ((*it)->isBroadcast()) {
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

FindAllMappedDims::FindAllMappedDims(
    TensorView* from,
    IterDomain* id,
    bool inner_only)
    : starting_tv_(from), starting_id_(id), inner_only_(inner_only) {}

void FindAllMappedDims::setUp() {
  mapped_root_ids_[starting_tv_] =
      projectIdToRoot(starting_tv_, starting_id_, inner_only_);
  mapped_rfactor_ids_[starting_tv_] =
      projectIdToRFactor(starting_tv_, starting_id_, inner_only_);
}

void FindAllMappedDims::propagateC2P(TensorView* from, TensorView* to) {
  auto from_id = mapped_root_ids_.at(from);
  PairwiseRootDomainMap root_map(to, from);
  auto c2p_map = root_map.mapConsumerToProducer(from->domain(), to->domain());
  auto p_it = c2p_map.find(from_id);
  if (p_it != c2p_map.end()) {
    mapped_root_ids_[to] = projectIdToRoot(to, p_it->second, inner_only_);
    mapped_rfactor_ids_[to] = p_it->second;
  } else {
    mapped_root_ids_[to] = nullptr;
    mapped_rfactor_ids_[to] = nullptr;
  }
}

void FindAllMappedDims::propagateP2C(TensorView* from, TensorView* to) {
  auto from_id = mapped_rfactor_ids_.at(from);
  PairwiseRootDomainMap root_map(from, to);
  auto p2c_map = root_map.mapProducerToConsumer(from->domain(), to->domain());
  auto c_it = p2c_map.find(from_id);
  if (c_it != p2c_map.end()) {
    mapped_root_ids_[to] = c_it->second;
    mapped_rfactor_ids_[to] = projectIdToRFactor(to, c_it->second, inner_only_);
  } else {
    mapped_root_ids_[to] = nullptr;
    mapped_rfactor_ids_[to] = nullptr;
  }
}

void FindAllMappedDims::propagateSibling(TensorView* from, TensorView* to) {
  auto from_id = mapped_root_ids_.at(from);
  if (from_id == nullptr) {
    mapped_root_ids_[to] = nullptr;
  } else {
    for (auto i : c10::irange(from->getRootDomain().size())) {
      if (from_id == from->getRootDomain()[i]) {
        mapped_root_ids_[to] = to->getRootDomain()[i];
        break;
      }
    }
  }
  from_id = mapped_rfactor_ids_.at(from);
  if (from_id == nullptr) {
    mapped_root_ids_[to] = nullptr;
  } else {
    for (auto i : c10::irange(from->getMaybeRFactorDomain().size())) {
      if (from_id == from->getMaybeRFactorDomain()[i]) {
        mapped_rfactor_ids_[to] = to->getMaybeRFactorDomain()[i];
        return;
      }
    }
  }
  TORCH_INTERNAL_ASSERT(false, "Unable to find mapped root/rfactor domain");
}

std::unordered_set<IterDomain*> FindAllMappedDims::get() const {
  std::unordered_set<IterDomain*> mapped_id_set;
  for (auto entry : mapped_root_ids_) {
    mapped_id_set.emplace(entry.second);
  }
  for (auto entry : mapped_rfactor_ids_) {
    mapped_id_set.emplace(entry.second);
  }
  return mapped_id_set;
}

bool hasInnerDim(
    TensorView* tv,
    std::unordered_set<IterDomain*> inner_dims,
    bool should_vectorize) {
  const auto& inner_most_dim = innerMostRootDim(tv);
  // TODO: Why "|| inner_most_dim->isReduction()"
  if (inner_most_dim == nullptr || inner_most_dim->isReduction()) {
    return false;
  }

  // Make sure inner most dimension is in the inner_dims set
  if (inner_dims.count(inner_most_dim) == 0) {
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
    bool inner_only,
    bool vectorize_pass) {
  if (vectorize_pass) {
    TORCH_INTERNAL_ASSERT(
        inner_only, "Can only vectorize inner-most dimensions");
  }

  auto inner_most_id = innerMostRootDim(reference_tv);

  if (inner_most_id == nullptr) {
    return {};
  }

  FindAllMappedDims all_mapped_root_dims(
      reference_tv, inner_most_id, inner_only);
  MaxRootDomainInfoSpanningTree tree(reference_tv);
  tree.traverse(&all_mapped_root_dims);

  auto vectorizable_dims = all_mapped_root_dims.get();

  std::vector<TensorView*> vectorizable_tensors;

  // We put outputs in front of inputs because this would make the transpose
  // scheduler prefer to use output instead of input as reference tensor.
  for (auto output_tv :
       ir_utils::filterByType<TensorView>(reference_tv->fusion()->outputs())) {
    if (hasInnerDim(output_tv, vectorizable_dims, vectorize_pass)) {
      vectorizable_tensors.push_back(output_tv);
    }
  }

  for (auto input_tv :
       ir_utils::filterByType<TensorView>(reference_tv->fusion()->inputs())) {
    if (hasInnerDim(input_tv, vectorizable_dims, vectorize_pass)) {
      vectorizable_tensors.push_back(input_tv);
    }
  }

  return vectorizable_tensors;
}

// Returns disjoint view sets mapped onto the given reference. Returns a pair
// of vectors of size rfactorDomain of reference. Vector of
// VectorOfUniqueEntries returns a const* to the disjoint set in
// disjoint_view_set the iterdomain is mapped to. Integer vector represents
// which disjoint view group the rfactor id belongs to. It's straight forward
// to map from the former to the latter, but not the latter to former.
//
// Since we return a const* to entries in disjoint_view_set, it must be passed
// in as a reference. Algorithm is N^2 based on number of dims in reference,
// but generating the disjoint view set is likely the limiter on perf of this
// function.
DisjointViewSetInfo getDisjointViewSetsOf(
    Fusion* fusion,
    TensorView* of,
    DisjointSets<IterDomain*>& disjoint_view_set) {
  auto rfactor_dom = of->getMaybeRFactorDomain();
  if (rfactor_dom.size() == 0) {
    return {};
  }

  // Start naming id's based on 0 so the inner most dimension will always be
  // 0, then as groups are discovered marching to the left their id will
  // increase. i.e. we could have something like [0, 3, 1, 2, 1, 0] as a
  // result.
  std::vector<int> disjoint_group_ids(rfactor_dom.size(), -1);
  std::vector<const VectorOfUniqueEntries<IterDomain*>*> disjoint_set_of_id(
      rfactor_dom.size(), nullptr);
  int current_group_id = 0;
  int ref_dim_i = rfactor_dom.size() - 1;

  while (ref_dim_i >= 0) {
    if (disjoint_group_ids[ref_dim_i] != -1) {
      // Already put in a group, continue
      ref_dim_i--;
      continue;
    }

    const auto& ref_group =
        disjoint_view_set.getDisjointSetOf(rfactor_dom[ref_dim_i]);

    int other_dim_i = ref_dim_i;
    while (other_dim_i >= 0) {
      const auto& other_group =
          disjoint_view_set.getDisjointSetOf(rfactor_dom[other_dim_i]);
      if (&ref_group == &other_group) {
        disjoint_group_ids[other_dim_i] = current_group_id;
        disjoint_set_of_id[other_dim_i] = &ref_group;
      }
      other_dim_i--;
    }

    ref_dim_i--;
    current_group_id++;
  }

  TORCH_INTERNAL_ASSERT(
      std::none_of(
          disjoint_group_ids.begin(),
          disjoint_group_ids.end(),
          [](int i) { return i == -1; }),
      "Failed to generate the view disjoint groups of the reference ",
      of->toString());

  TORCH_INTERNAL_ASSERT(
      std::none_of(
          disjoint_set_of_id.begin(),
          disjoint_set_of_id.end(),
          [](const VectorOfUniqueEntries<IterDomain*>* ptr) {
            return ptr == nullptr;
          }),
      "Failed to generate the view disjoint groups of the reference ",
      of->toString());

  DisjointViewSetInfo info;
  info.disjoint_sets_of_ref = disjoint_set_of_id;
  info.disjoint_set_ids = disjoint_group_ids;
  info.ref = of;

  return info;
}

BroadcastMultipleInformation getBroadcastMultiples(
    TensorView* reference_tv,
    DataType index_type) {
  auto fusion = reference_tv->fusion();
  FusionGuard fg(fusion);

  // We always cacheBefore output at the beginning of the scheduling. And after
  // cacheBefore, the reference tensor will have all reduction IDs removed.
  // TODO: clean this up when we kill trivial reduction.
  auto ref_root_domain =
      TensorDomain::noReductions(reference_tv->getMaybeRFactorDomain());

  std::vector<BroadcastMultiple> multiples(ref_root_domain.size());

  auto disjoint_view_sets = disjointViewSets(fusion);
  auto disjoint_set_information = scheduler_utils::getDisjointViewSetsOf(
      fusion, reference_tv, disjoint_view_sets);

  auto ref_disjoint_sets = disjoint_set_information.disjoint_sets_of_ref;
  auto ref_disjoint_set_ids = disjoint_set_information.disjoint_set_ids;

  // All input or output tensor views
  std::vector<TensorView*> in_out_tvs;
  {
    auto inp_tvs = ir_utils::filterByType<TensorView>(fusion->inputs());
    in_out_tvs.insert(in_out_tvs.end(), inp_tvs.begin(), inp_tvs.end());
    auto out_tvs = ir_utils::filterByType<TensorView>(fusion->outputs());
    in_out_tvs.insert(in_out_tvs.end(), out_tvs.begin(), out_tvs.end());
  }

  // Shouldn't matter if we use EXACT or PERMISSIVE mapping mode for compute
  // at map as we're just looking at the root mappings.
  auto ca_map = ComputeAtMap(fusion);

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

      bool ref_id_has_view_transforms = std::count(
                                            ref_disjoint_set_ids.begin(),
                                            ref_disjoint_set_ids.end(),
                                            ref_disjoint_set_ids[ref_i]) > 1;

      // Could have multiple mappings if there's view transforms
      std::vector<IterDomain*> mapped_ids;
      if (!ref_id_has_view_transforms) {
        auto mapped_it = std::find_if(
            in_out_tv_domain_list.begin(),
            in_out_tv_domain_list.end(),
            [&ref_id, &ca_map](IterDomain* in_out_tv_id) {
              return ca_map.areMapped(
                  in_out_tv_id, ref_id, IdMappingMode::EXACT);
            });
        if (mapped_it != in_out_tv_domain_list.end()) {
          mapped_ids.push_back(*mapped_it);
        }
      } else {
        for (auto in_out_id : in_out_tv_domain) {
          if (ref_disjoint_sets[ref_i]->has(in_out_id)) {
            mapped_ids.push_back(in_out_id);
          }
        }
      }

      // Nothing maps to reference, no contribution to multiples for this dim
      if (mapped_ids.empty()) {
        continue;
      }

      if (std::all_of(mapped_ids.begin(), mapped_ids.end(), [](IterDomain* id) {
            return id->isReduction() || id->isBroadcast();
          })) {
        continue;
      }

      // If any iteration domain in the input or output that's mapped through
      // the view disjoint set is not a reduction or broadcast, assume it's a
      // full dimension for the sake of the pointwise scheduler.
      mapped_axes[ref_i] = true;
    }

    // For each break point position if there an lhs or rhs multiple based on
    // this tensor add it to the global multiplier. The only time we consider
    // we can benefit from broadcast is if the entire left or right side the
    // break point is all broadcasts.
    {
      bool rhs = false;
      bool lhs = false;
      auto dtype_size =
          dataTypeSize(in_out_tv->getDataType().value(), index_type);
      for (auto mapped_axes_i : c10::irange(mapped_axes.size())) {
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
  BroadcastMultipleInformation bcast_info;
  bcast_info.view_disjoint_set_ids = ref_disjoint_set_ids;
  bcast_info.broadcast_multiples = multiples;
  return bcast_info;
}

namespace matmul_utils {

void scheduleWarpTileWithReduction(TensorView* tv, MatMulTileOptions tile) {
  // Assumes
  // [M, N, K]
  auto cta_tile = tile.cta_tile;
  auto warp_tile = tile.warp_tile;
  auto instruction_tile = tile.instruction_tile;

  TORCH_CHECK(
      cta_tile.k % warp_tile.k == 0,
      "Number of warp on k dimension need to be integer");

  int num_warp_k = cta_tile.k / warp_tile.k;

  mma_util::checkDimSize(
      tv, {-3, -2, -1}, {cta_tile.m, cta_tile.n, cta_tile.k});

  if (num_warp_k == 1) {
    // Non split K over warp case:

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
  } else {
    // Split K over warp case:
    // Main difference is that an additional
    //  thread dimension needs to be reserved
    //  for cross warp reduction:
    //       -3   -2  -1
    //[...    M,   N,  K]
    // Distribute warp tile:
    tv->split(-3, warp_tile.m);
    tv->split(-2, warp_tile.n);
    tv->split(-1, warp_tile.k);

    //   -6  -5   -4   -3   -2 -1
    // [Mwo  Mw  Nwo   Nw   K, Kw]
    tv->split(-5, instruction_tile.m);
    tv->split(-3, instruction_tile.n);
    tv->split(-1, instruction_tile.k);

    //  -9  -8  -7 -6 -5 -4 -3 -2 -1
    // [Mwo Mw Mi Nwo Nw Ni Kwo Kw Ki]

    tv->reorder({{-8, -6}, {-7, -3}, {-6, -8}, {-4, -2}, {-3, -7}, {-2, -4}});
    //  -9   -8  -7 -6 -5 -4 -3 -2 -1
    // [Mwo  Nwo Ko Mw Nw Kw, Mi Ni Ki]

    tv->merge(-9);
    //  -8  -7 -6 -5 -4   -3 -2 -1
    // [MNwo Ko Mw Nw Kw, Mi Ni Ki]
  }
}

void scheduleWarpTileWithNoReduction(TensorView* tv, MatMulTileOptions tile) {
  // Assumes
  // [M, N, K]
  auto cta_tile = tile.cta_tile;
  auto warp_tile = tile.warp_tile;
  auto instruction_tile = tile.instruction_tile;

  mma_util::checkDimSize(tv, {-2, -1}, {cta_tile.m, cta_tile.n});

  TORCH_CHECK(
      cta_tile.k % warp_tile.k == 0,
      "Number of warp on k dimension need to be integer");

  int num_warp_k = cta_tile.k / warp_tile.k;

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

  if (num_warp_k != 1) {
    // The non reduction warps are merged together
    //  to save one thread dim for cross dim reduce.
    tv->merge(-6);
    //  -5  -4 -3 -2 -1
    // [MNo Mw Nw Mi Ni]
  }
}

//! Split the innermost dim to a vectorized load
void scheduleContiguousVectorLoad(
    TensorView* tv,
    MatMulTileOptions tile,
    int vector_word,
    bool vectorize) {
  auto warp_dims = tile.cta_tile / tile.warp_tile;
  int num_of_thread = warp_dims.m * warp_dims.n * warp_dims.k * 32;

  tv->split(-1, num_of_thread * vector_word);
  tv->split(-1, vector_word);
  // [..., thread, vec]
  // distribute to warp: for tidx
  tv->split(-2, 32);

  //      -3    -2    -1
  // [...warp, lane, vec]

  if (warp_dims.k == 1) {
    //      -4     -3    -2    -1
    // [...warpM, warpN, lane, vec]
    tv->split(-3, warp_dims.n);
  } else {
    //      -4     -3    -2    -1
    // [...warpMN, warpR, lane, vec]
    tv->split(-3, warp_dims.k);
  }

  if (vectorize) {
    tv->axis(-1)->parallelize(ParallelType::Vectorize);
  }

  tv->axis(-2)->parallelize(ParallelType::TIDx);
  tv->axis(-3)->parallelize(ParallelType::TIDy);
  tv->axis(-4)->parallelize(ParallelType::TIDz);
}

void makeTile(TensorView* tv, std::vector<int> tile_sizes) {
  TORCH_CHECK(
      tv->domain()->domain().size() >= tile_sizes.size(),
      "Tensor dimension less than tile dimension!");

  // Number of inner dimensions we are tiling.
  const auto tile_dimension_size = tile_sizes.size();

  // Split the inner dimensions:
  for (auto idx : c10::irange(tile_dimension_size)) {
    // Using negative indexing to accomodate potential batching
    //  dimensions on the further left. Eg.:
    //  0, 1, 2   ->         -3,-2,-1
    // [M, N, K]  -> [B0, B1, M, N, K]
    tv->split(idx - tile_dimension_size, tile_sizes.at(idx));
  }

  // The transformation happened should look like:
  //   Before               After
  // [..., M, N, K] -> [..., Mo, Mi, No, Ni, Ko, Ki]

  // Re-order the tiles so that all the outer tiles are
  //  on the left of all the inner tiles
  std::unordered_map<int, int> reorder_map_old_to_new;

  // Number of tiled inner dimensions after we split.
  const auto split_tile_dimension_size = 2 * tile_dimension_size;
  for (auto idx : c10::irange(split_tile_dimension_size)) {
    // We want to reorder as follows:
    //           Before
    //
    // [..., Mo, Mi, No, Ni, Ko, Ki] ->
    //                 After
    //      vvv group0 vvv  vvv group1 vvv
    // [..., Mo, No, Ko,     Mi, Ni, Ki]

    // The index offset within group of current
    //  iterdomain, with grouping specified above.
    auto index_within_group = idx / 2;

    // The index of the group the current id belongs
    //  to, as specified above.
    auto group_index = idx % 2;

    // Calculate the actual index after reordering
    auto index_after_reorder =
        group_index * tile_dimension_size + index_within_group;

    // Add pair {idx_before, idx_after} to re-order map.
    reorder_map_old_to_new.insert(std::make_pair(
        idx - split_tile_dimension_size,
        index_after_reorder - split_tile_dimension_size));
  }

  // Apply the re-order map to tensor
  tv->reorder(reorder_map_old_to_new);
}

namespace {

c10::optional<IterDomain*> getMaybeRootIfInnermostTiled(
    IterDomain* id,
    const std::unordered_set<IterDomain*>& maybe_rfactor_id_set) {
  // Root id defaults to an "innermost id".
  while (id->definition() && !maybe_rfactor_id_set.count(id)) {
    if (auto split = dynamic_cast<Split*>(id->definition())) {
      if (id == split->inner()) {
        id = split->in();
        continue;
      }
    }
    // Didn't pass the inner most check, return empty.
    return c10::nullopt;
  }

  return id;
}

} // namespace

void orderTiledConcreteIdAsRoot(TensorView* tv) {
  auto ndims = tv->nDims();

  // Keep track of the left most position where we will
  //  be reordering the axes.
  auto leftmost_pos = ndims;

  // Pull the root id's of the given tv.
  std::unordered_set<IterDomain*> maybe_rfactor_id_set{
      tv->getMaybeRFactorDomain().begin(), tv->getMaybeRFactorDomain().end()};

  // Keep track of leaf positions that is either a reduction
  //  or a broadcast.
  // Note: Currently don't really see a case where this function
  //  should be called on a reduction output tv, but adding them
  //  here for completeness.
  std::deque<int> broadcast_or_reduction_pos;

  // Map the root id's to their innermost concrete id's
  //  on the leaf.
  std::unordered_map<IterDomain*, int> root_id_to_inner_leaf_pos;

  // Try to re-order inner iterdomains from the innermost
  //  position backward. This utility only tries to re-order
  //  inner tiles on the innermost positions, like the resulting
  //  tensor from makeTile utility.
  // The re-ordering would first try to decide the inner iterdomains
  //  we want to re-order. For this we start from the innermost position
  //  and move back and collect all the iterdomains that we know
  //  are inner tiles of some root domain or broadcast/reduction domains
  //  that won't affect the concrete id layout.
  // The collection process would stop whenever a iterdomain that is
  //  neither an inner tile nor reduction/broadcast is found, and would
  //  not re-order any iterdomain beyond that point to keep the
  //  outer loop structure unchanged.
  for (int64_t i = static_cast<int64_t>(ndims) - 1; i >= 0; i--) {
    auto leaf_id = tv->axis(i);
    if (leaf_id->isBroadcast() || leaf_id->isReduction()) {
      // Register this reduction or broadcast axis
      //  to reorder.
      broadcast_or_reduction_pos.push_front(i);
      leftmost_pos = i;
      continue;
    }
    auto maybe_root =
        getMaybeRootIfInnermostTiled(leaf_id, maybe_rfactor_id_set);

    if (maybe_root.has_value()) {
      // Found an innermost id, add them to the
      //  axes to reorder.
      TORCH_INTERNAL_ASSERT(
          root_id_to_inner_leaf_pos
              .insert(std::make_pair(maybe_root.value(), i))
              .second,
          "Multiple \"innermost\" id seen for root id :",
          maybe_root.value()->toString(),
          " on ",
          tv->toString(),
          " very likely an invariant is broken.");
      leftmost_pos = i;
    } else {
      break;
    }
  }

  // Calculate the ordering:

  // pointer to the current target postion after
  //  repordering
  int current_pos = leftmost_pos;
  std::unordered_map<int, int> reorder_map_old_to_new;

  // first place all the broadcast and reduction on the left:
  for (auto original_broadcast_or_reduction_pos : broadcast_or_reduction_pos) {
    reorder_map_old_to_new[original_broadcast_or_reduction_pos] = current_pos++;
  }

  // Next put all the innermost leaf id's, we make sure that
  //  the inner tile ordering follows the corresponding root
  //  domain ordering by iterating on the root domain and
  //  find their corresponding inner tile iterdomains from
  //  the populated root_id_to_inner_leaf_pos.
  for (auto root_id : tv->getMaybeRFactorDomain()) {
    auto leaf_id_pos_it = root_id_to_inner_leaf_pos.find(root_id);
    if (leaf_id_pos_it != root_id_to_inner_leaf_pos.end()) {
      reorder_map_old_to_new[leaf_id_pos_it->second] = current_pos++;
    }
  }

  // Validate that we have processed all inner ids or broadcast/reduction
  //  ids we have registered.
  TORCH_INTERNAL_ASSERT(current_pos == ndims, "Inconsistent ordering logic");

  // Apply the new order:
  tv->reorder(reorder_map_old_to_new);
}

} // namespace matmul_utils

//! Propagate current transformations on from_tv to all graphs
void transformPropagateToAllFrom(TensorView* from_tv, int pos) {
  TransformPropagator propagator(from_tv, pos);
  MaxRootDomainInfoSpanningTree(from_tv, nullptr).traverse(&propagator);
}

namespace {

//! Utility enum to signify which direction
//! BoundedDirectionalTransformPropagator
//!  passes will propagate the transforms.
enum class PropagateDirection { Backward = 0, Forward };

//! Returns true if the given tensorview is a fake boundary
//!  TensorView, see Note [Fake Boundary Tensorview].
//! This function assumes and would not check that tv is a boundary
//!  of the select_tv set.
bool isFakeBoundaryTensorview(
    TensorView* tv,
    const std::unordered_set<TensorView*>& selected_tv_set,
    PropagateDirection direction) {
  if (direction == PropagateDirection::Forward) {
    // In the case of forward propagation,
    //  a boundary tv is a fake boundary if
    //  it has any consumer tv that's in the selected
    //  set.
    for (auto consumer_tv : ir_utils::consumerTvsOf(tv)) {
      if (selected_tv_set.count(consumer_tv)) {
        // Found a consumer that's in selected tv set.
        return true;
      }
    }

  } else {
    // In the case of backward propagation,
    //  a boundary tv is a fake boundary if it has any producer
    //  that is within the selected set.
    for (auto producer_tv : ir_utils::producerTvsOf(tv)) {
      if (selected_tv_set.count(producer_tv)) {
        // Found a producer that's in selected tv set.
        return true;
      }
    }
  }

  // Didn't find any producer/consumer in the selected tv set.
  //  The given tv is not a fake boundary tv.
  return false;
}

//! Utility function to generate the set of tensorviews to propagate
//!  transform to by BoundedDirectionalTransformPropagator.
std::unordered_set<TensorView*> getDirectionalPropagatePathSet(
    TensorView* from_tv,
    std::vector<TensorView*> boundary_tvs,
    BoundedDirectionalTransformPropagator::Options options,
    PropagateDirection direction) {
  // Prepare to collect all candidate tensorviews
  //  within the specified boundary.
  std::vector<Val*> propagate_candidate;

  // Collect boundary tvs in a set.
  std::unordered_set<TensorView*> boundary_tv_set(
      boundary_tvs.begin(), boundary_tvs.end());

  if (direction == PropagateDirection::Forward) {
    // In the case of forward propagation, collect all tvs
    //  that are consumers of `from_tv` and producers of
    //  boundary tvs.
    propagate_candidate = DependencyCheck::getAllValsBetween(
        {from_tv}, {boundary_tvs.begin(), boundary_tvs.end()});
  } else {
    // In the case of backward propagation, collect all tvs
    //  that are producers of `from_tv` and consumers of
    //  boundary tvs.
    propagate_candidate = DependencyCheck::getAllValsBetween(
        {boundary_tvs.begin(), boundary_tvs.end()}, {from_tv});
  }

  // Populate initial selected tensorviews in a set.
  auto propagate_candidate_tv_view =
      ir_utils::filterByType<TensorView>(propagate_candidate);
  // Prepare to filter out un-wanted tensorviews according
  //  to the option parameters.
  std::unordered_set<TensorView*> propagate_path_set{
      propagate_candidate_tv_view.begin(), propagate_candidate_tv_view.end()};

  // Remove boundary tensorviews if we don't want to transform
  //  tensorviews on the boundary.
  if (!options.transform_boundary) {
    // Additional refining step to identify "fake boundary" tensorviews.
    //  We don't want to erase fake boundary tensorviews from the selected
    //  set when we are erasing boundary tvs.
    //
    // Note [Fake Boundary Tensorview]
    // A tensorview, tv0, is defined as fake boundary tv if
    //  1. Tv0 is on the given boundary set.
    //  2. There is a path from another boundary tv, Tv1 to from_tv that
    // goes through Tv0.
    //
    // In this case the propagation behavior is not precisely defined.
    // Our current decision is to treat such tensorview as non-boundary
    //  tv to make sure the propagation paths are not blocked. E.g.:
    //
    //  T1 = T0
    //  T2 = T1
    //  T3 = T2 + T1
    // if we propagate with from_tv = {T3}, boundary_tv = {T0, T2},
    // transform_boundary=false
    //
    // Here T2 is a fake boundary and we will still transform T2 as it is
    //  on the path between T3 and T0.

    // Initialize set of fake boundary tvs.
    std::unordered_set<TensorView*> fake_boundary_set;

    // Populate the set of fake boundary tvs.
    std::copy_if(
        boundary_tvs.begin(),
        boundary_tvs.end(),
        std::inserter(fake_boundary_set, fake_boundary_set.end()),
        [&propagate_path_set, direction](TensorView* tv) {
          return isFakeBoundaryTensorview(tv, propagate_path_set, direction);
        });

    // Remove boundary tvs from the selected set, keeping fake boundary tvs.
    for (auto boundary_tv : boundary_tvs) {
      if (!fake_boundary_set.count(boundary_tv)) {
        propagate_path_set.erase(boundary_tv);
      }
    }
  }

  return propagate_path_set;
}

} // namespace

void BoundedDirectionalTransformPropagator::propagate(
    TensorView* from_tv,
    int pos,
    std::unordered_set<TensorView*> included_tvs,
    Options options) {
  // Run transform propagation using the custom selector.
  SetSelector selector(included_tvs);
  TransformPropagator propagator(from_tv, pos);
  MaxRootDomainInfoSpanningTree(from_tv, &selector).traverse(&propagator);

  // Propagate parallel type if requested by option parameters.
  if (options.propagate_parallel_type) {
    scheduler_utils::parallelizeAllLike(
        from_tv,
        options.parallel_propagation_pos,
        {included_tvs.begin(), included_tvs.end()},
        allParallelTypesExcept({ParallelType::Vectorize, ParallelType::Mma}));
  }
}

void BoundedDirectionalTransformPropagator::backward(
    TensorView* from,
    int pos,
    std::vector<TensorView*> to,
    c10::optional<Options> options) {
  if (!options.has_value()) {
    options = Options();
  }
  TORCH_INTERNAL_ASSERT(
      !to.empty(),
      "Propagation needs to be bounded, so no support for empty boundary.");

  // Collect all tvs to included on the backward path as specified
  //  by boundary and options.
  auto included_tvs = getDirectionalPropagatePathSet(
      from, to, *options, PropagateDirection::Backward);
  // Actually run the propagation.
  propagate(from, pos, included_tvs, *options);
}

void BoundedDirectionalTransformPropagator::forward(
    TensorView* from,
    int pos,
    std::vector<TensorView*> to,
    c10::optional<Options> options) {
  if (!options.has_value()) {
    options = Options();
  }
  TORCH_INTERNAL_ASSERT(
      !to.empty(),
      "Propagation needs to be bounded, so no support for empty boundary.")

  // Collect all tvs to included on the forward path as specified
  //  by boundary and options.
  auto included_tvs = getDirectionalPropagatePathSet(
      from, to, *options, PropagateDirection::Forward);

  // Actually run the propagation.
  propagate(from, pos, included_tvs, *options);
}

void BoundedDirectionalTransformPropagator::bothWays(
    TensorView* from,
    int pos,
    std::vector<TensorView*> backward_to,
    std::vector<TensorView*> forward_to,
    c10::optional<Options> options) {
  if (!options.has_value()) {
    options = Options();
  }
  TORCH_INTERNAL_ASSERT(
      !backward_to.empty() && !forward_to.empty(),
      "Propagation needs to be bounded, so no support for empty boundary.")

  // Collect all tvs to included on the backward and forward path as specified
  //  by boundary and options.
  auto backward_included_tvs = getDirectionalPropagatePathSet(
      from, backward_to, *options, PropagateDirection::Backward);
  auto forward_included_tvs = getDirectionalPropagatePathSet(
      from, forward_to, *options, PropagateDirection::Forward);

  // Combined the included tvs on both paths.
  auto included_tvs = backward_included_tvs;
  included_tvs.insert(forward_included_tvs.begin(), forward_included_tvs.end());

  // Run the propagation on the combined set of tvs.
  propagate(from, pos, included_tvs, *options);
}

DisjointSets<IterDomain*> disjointViewSets(Fusion* fusion) {
  // Start from the exact iter domain graph of the fusion
  IterDomainGraph id_graph(fusion);
  auto disjoint_view_ids = id_graph.exactNodes();

  // If iter domains are involved in any transformation from root domains to
  // rfactor domains they should be considered "contaminated".
  for (auto tv : ir_utils::allTvs(fusion)) {
    for (auto expr : StmtSort::getExprs(
             fusion,
             {tv->getMaybeRFactorDomain().begin(),
              tv->getMaybeRFactorDomain().end()})) {
      if (expr->isA<Merge>()) {
        auto merge = expr->as<Merge>();
        disjoint_view_ids.mapEntries(merge->inner(), merge->out());
        disjoint_view_ids.mapEntries(merge->outer(), merge->out());
      } else if (expr->isA<Split>()) {
        auto split = expr->as<Split>();
        disjoint_view_ids.mapEntries(split->in(), split->inner());
        disjoint_view_ids.mapEntries(split->in(), split->outer());
      } else {
        TORCH_INTERNAL_ASSERT(
            false, "Expression type: ", expr->toString(), " not supported.");
      }
    }
  }
  return disjoint_view_ids;
}

bool breakIsDisjoint(std::vector<int> group_ids, int pos) {
  if (pos < 0) {
    pos += group_ids.size();
  }
  TORCH_INTERNAL_ASSERT(
      pos >= 0 && pos <= group_ids.size(),
      "Invalid position, size of vec is ",
      group_ids.size(),
      " but position is ",
      pos);

  if (pos == 0 || pos == group_ids.size()) {
    return true;
  }

  std::unordered_set<int> left_ints(group_ids.begin(), group_ids.begin() + pos);

  for (auto i = pos; i < group_ids.size(); i++) {
    if (left_ints.count(group_ids[i]) > 0) {
      return false;
    }
  }
  return true;
}

std::unordered_map<int, int> domainReorderAsRfactorMap(TensorView* tv) {
  FusionGuard fg(tv->fusion());
  auto transform_exprs = StmtSort::getExprs(
      tv->fusion(),
      {tv->domain()->domain().begin(), tv->domain()->domain().end()});
  // simply update this vector of id's as progressing through the transformation
  // expressions. We'll always insert the result of split in the location of the
  // input, and insert the merge result in the position of the inner dimension.

  auto reordered_ids = tv->getMaybeRFactorDomain();
  for (const auto* expr : transform_exprs) {
    if (const Split* split = dynamic_cast<const Split*>(expr)) {
      auto find_it =
          std::find(reordered_ids.begin(), reordered_ids.end(), split->in());
      if (find_it == reordered_ids.end()) {
        // Transformations before rfactor, ignore those.
        continue;
      }
      auto pos = std::distance(reordered_ids.begin(), find_it);
      reordered_ids[pos] = split->inner();
      reordered_ids.insert(reordered_ids.begin() + pos, split->outer());
    } else if (const Merge* merge = dynamic_cast<const Merge*>(expr)) {
      auto find_it_0 =
          std::find(reordered_ids.begin(), reordered_ids.end(), merge->outer());
      auto find_it_1 =
          std::find(reordered_ids.begin(), reordered_ids.end(), merge->inner());
      if (find_it_0 == reordered_ids.end() &&
          find_it_1 == reordered_ids.end()) {
        // Transformations before rfactor, ignore those.
        continue;
      }
      TORCH_INTERNAL_ASSERT(
          find_it_0 != reordered_ids.end() && find_it_1 != reordered_ids.end(),
          "Error in transformations of ",
          tv->toString(),
          "\nTransformations before rfactor should not mix with transformations after rfactor.");
      auto pos0 = std::distance(reordered_ids.begin(), find_it_0);
      auto pos1 = std::distance(reordered_ids.begin(), find_it_1);
      if (pos0 > pos1) {
        std::swap(pos0, pos1);
      }
      // Should be impossible.
      TORCH_INTERNAL_ASSERT(
          pos0 != pos1,
          "Didn't expect merge inputs to be the same iteratrion domain:\n",
          merge->toString());

      reordered_ids.erase(reordered_ids.begin() + pos0);
      reordered_ids[--pos1] = merge->out();
    }
  }

  std::unordered_map<int, int> old2new;
  for (auto id_i : c10::irange(tv->domain()->domain().size())) {
    auto leaf_id = tv->axis(id_i);
    auto find_it =
        std::find(reordered_ids.begin(), reordered_ids.end(), leaf_id);
    TORCH_INTERNAL_ASSERT(
        find_it != reordered_ids.end(),
        "Reordering map creation failed, uninitialized iterdomain,",
        " likely something is wrong with the transformations between the rfactor domain and the leaves.");
    int new_pos = (int)std::distance(reordered_ids.begin(), find_it);
    int old_pos = (int)id_i;
    old2new[old_pos] = new_pos;
  }
  return old2new;
}

void propagateViewTransforms(Fusion* fusion, const ComputeAtMap& ca_map) {
  std::unordered_set<std::shared_ptr<VectorOfUniqueEntries<IterDomain*>>>
      transformed_disjoint_sets;

  // If iter domains are involved in any transformation from root domains to
  // rfactor domains they should be considered "contaminated".
  for (auto tv : ir_utils::allTvs(fusion)) {
    for (auto expr : StmtSort::getExprsBetween(
             fusion,
             {tv->getRootDomain().begin(), tv->getRootDomain().end()},
             {tv->getMaybeRFactorDomain().begin(),
              tv->getMaybeRFactorDomain().end()})) {
      for (auto id : ir_utils::filterByType<IterDomain>(expr->inputs())) {
        transformed_disjoint_sets.emplace(
            ca_map.disjointSetOf(id, IdMappingMode::EXACT));
      }
    }
  }

  std::unordered_set<IterDomain*> terminating_rfactor_dims;
  for (const auto& disjoint_set_shared_ptr :
       ca_map.idGraph().exactNodes().disjointSets()) {
    if (std::none_of(
            disjoint_set_shared_ptr->vector().begin(),
            disjoint_set_shared_ptr->vector().end(),
            [](IterDomain* id) { return id->isRFactorProduct(); })) {
      continue;
    }
    if (transformed_disjoint_sets.find(disjoint_set_shared_ptr) !=
        transformed_disjoint_sets.end()) {
      // Disjoint set was transformed for view, ignore it
      continue;
    }
    for (auto id : disjoint_set_shared_ptr->vector()) {
      terminating_rfactor_dims.emplace(id);
    }
  }

  // If iter domains are involved in any transformation from root domains to
  // rfactor domains they should be considered "contaminated".
  for (auto tv : ir_utils::allTvs(fusion)) {
    if (!tv->hasRFactor()) {
      continue;
    }

    std::unordered_map<int, int> old2new;
    // Make sure rfactor dims we need are in domain, and reorder them in domain
    // so they're consecutive starting from the left of domain. TODO: We could
    // improve this so that if there's transformations replayed after the
    // rfactor dims we could try and pull those through the fusion instead of
    // enforcing rfactor dims are in domain.
    for (auto rfactor_id : tv->getMaybeRFactorDomain()) {
      if (terminating_rfactor_dims.find(rfactor_id) !=
          terminating_rfactor_dims.end()) {
        auto find_it = std::find(
            tv->domain()->domain().begin(),
            tv->domain()->domain().end(),
            rfactor_id);
        TORCH_INTERNAL_ASSERT(
            find_it != tv->domain()->domain().end(),
            "Require ",
            rfactor_id,
            " is in the active domain of ",
            tv->toString(),
            " for view propagation.");
        auto old_pos = std::distance(tv->domain()->domain().begin(), find_it);

        old2new[old_pos] = old2new.size();
      }
    }

    if (old2new.empty()) {
      continue;
    }

    // Propagate the view transformations
    tv->reorder(old2new);
    //! Propagate current transformations on from_tv to all graphs
    transformPropagateToAllFrom(tv, old2new.size());
  }
}

} // namespace scheduler_utils

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
