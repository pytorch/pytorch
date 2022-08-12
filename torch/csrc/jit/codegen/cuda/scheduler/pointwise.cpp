#include <torch/csrc/jit/codegen/cuda/scheduler/pointwise.h>

#include <torch/csrc/jit/codegen/cuda/executor_utils.h>
#include <torch/csrc/jit/codegen/cuda/instrumentation.h>
#include <torch/csrc/jit/codegen/cuda/ir_iostream.h>
#include <torch/csrc/jit/codegen/cuda/ir_utils.h>
#include <torch/csrc/jit/codegen/cuda/lower_utils.h>
#include <torch/csrc/jit/codegen/cuda/scheduler/registry.h>
#include <torch/csrc/jit/codegen/cuda/scheduler/utils.h>
#include <torch/csrc/jit/codegen/cuda/scheduler/vectorize_helper.h>
#include <torch/csrc/jit/codegen/cuda/transform_replay.h>
#include <torch/csrc/jit/codegen/cuda/utils.h>

#include <ATen/cuda/CUDAContext.h>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

namespace {
// constexpr int64_t x_grid_limit = ((int64_t)1 << (int64_t)31) - (int64_t)1;
// Unused at the moment, commenting for clang tidy
constexpr int64_t kThreadX = 128;

// DomainMap uses the ComputeAtMap to find a reference TensorView
// that maps to all iterDomains in the fusion.
class DomainMap {
 public:
  DomainMap(Fusion* fusion) : fusion_(fusion), ca_map_(ComputeAtMap(fusion)) {
    view_tvs_ = scheduler_utils::getViewTVs(fusion);
  }

  // The pointwise scheduler heuristics requires a minimum number of axes.
  // The output reference tensor should respect this requirement.
  TensorView* findReferenceTensorView(int minimum_num_axes = 0) const {
    for (auto output_tv :
         ir_utils::filterByType<TensorView>(fusion_->outputs())) {
      if (isValidReference(output_tv) &&
          hasMinimumSize(output_tv, minimum_num_axes) &&
          !output_tv->isFusionInput()) {
        return output_tv;
      }
    }
    return nullptr;
  }

  static bool hasReferenceTensorView(Fusion* fusion) {
    FusionGuard fg(fusion);
    DomainMap domain_map(fusion);
    return domain_map.findReferenceTensorView() != nullptr;
  }

  // Determine if output TensorView is a valid reference tensor for this fusion.
  // The reference tensor must map to all the iterDomains in each input.
  bool isValidReference(TensorView* output_tv) const {
    if (output_tv->isFusionInput()) {
      return false;
    }
    for (auto input_tv :
         ir_utils::filterByType<TensorView>(fusion_->inputs())) {
      if (input_tv->uses().empty()) {
        continue;
      }

      if (fusion_->getOutputAlias(output_tv) == input_tv) {
        continue;
      }

      if (!areAllMapped(input_tv, output_tv)) {
        return false;
      }
    }
    return true;
  }

 private:
  bool hasMinimumSize(TensorView* tv, int num_axes) const {
    TORCH_INTERNAL_ASSERT(tv != nullptr);
    return (num_axes == 0 || tv->getMaybeRFactorDomain().size() > num_axes);
  }

  // Determine if all iterDomains are mapped between input and output tvs
  bool areAllMapped(TensorView* input_tv, TensorView* output_tv) const {
    // Get concrete IDs for input root or rfactor domain
    std::unordered_set<IterDomain*> in_concrete_ids;
    for (auto in_id : input_tv->getMaybeRFactorDomain()) {
      if (!ca_map_.getConcreteMappedID(in_id, IdMappingMode::EXACT)
               ->isBroadcast() &&
          !in_id->isReduction()) {
        in_concrete_ids.insert(
            ca_map_.getConcreteMappedID(in_id, IdMappingMode::EXACT));
      }
    }

    // Erase all input concrete IDs mapped to the output domain
    // Ignore unresolved broadcast dimensions
    for (auto out_id : output_tv->getMaybeRFactorDomain()) {
      if (!out_id->isBroadcast()) {
        if (!eraseIfMapped(in_concrete_ids, out_id)) {
          eraseIfMappedThroughView(in_concrete_ids, out_id);
        }
      }
    }
    return in_concrete_ids.empty();
  }

  // Erase input concrete ID if it is mapped to output ID
  bool eraseIfMapped(
      std::unordered_set<IterDomain*>& in_concrete_ids,
      IterDomain* out_id) const {
    auto out_concrete_id =
        ca_map_.getConcreteMappedID(out_id, IdMappingMode::EXACT);
    auto in_concrete_id_iter = in_concrete_ids.find(out_concrete_id);
    bool found_match = in_concrete_id_iter != in_concrete_ids.end();
    if (found_match) {
      in_concrete_ids.erase(in_concrete_id_iter);
    }
    return found_match;
  }

  // Check if in_id is mapped to out_id through any view rfactor domain
  void eraseIfMappedThroughView(
      std::unordered_set<IterDomain*>& in_concrete_ids,
      IterDomain* out_id) const {
    for (auto view : view_tvs_) {
      // Find any ID in view rfactor domain that is mapped to output ID
      auto view_rfactor_id = anyMapped(view->getRFactorDomain(), out_id);
      if (view_rfactor_id == nullptr) {
        continue;
      }

      if (view_rfactor_id->isRFactorProduct()) {
        // Check if input ID is mapped to any input IDs of the view rfactor ID
        auto root_inputs = InputsOf::outputs(fusion_, {view_rfactor_id});
        auto filtered_root_ids =
            ir_utils::filterByType<IterDomain>(root_inputs);
        for (auto view_root_id : filtered_root_ids) {
          eraseIfMapped(in_concrete_ids, view_root_id);
        }
      } else {
        // Otherwise, the input ID must map to the view rfactor ID
        eraseIfMapped(in_concrete_ids, view_rfactor_id);
      }
    }
  }

  // Find any id in domain that maps with target id
  IterDomain* anyMapped(
      const std::vector<IterDomain*> domain,
      IterDomain* target) const {
    for (auto id : domain) {
      if (ca_map_.areMapped(id, target, IdMappingMode::EXACT)) {
        return id;
      }
    }
    return nullptr;
  }

  Fusion* fusion_ = nullptr;
  ComputeAtMap ca_map_;
  std::vector<TensorView*> view_tvs_;
};

} // namespace

c10::optional<PointwiseParams> getPointwiseHeuristics(
    Fusion* fusion,
    const at::ArrayRef<c10::IValue>& runtime_inputs,
    HeuristicSummary* data_cache) {
  SchedulerRuntimeInfo runtime_info(fusion, runtime_inputs, true);
  return getPointwiseHeuristics(fusion, runtime_info, data_cache);
}

c10::optional<PointwiseParams> getPointwiseHeuristics(
    Fusion* fusion,
    SchedulerRuntimeInfo& runtime_info,
    HeuristicSummary* data_cache) {
  FUSER_PERF_SCOPE("getPointwiseHeuristics");

  FusionGuard fg(fusion);

  // Incase any buffer is of type DataType::Index
  DataType index_type = indexModeToDtype(runtime_info.getIndexMode());

  TensorView* largest_out = nullptr;
  int max_dims = -1;

  auto in_tvs = ir_utils::filterByType<TensorView>(fusion->inputs());
  // Will want to access this with direct indexing later, convert now.
  std::vector<TensorView*> out_tvs;
  // Only use valid reference tensors during heuristics analysis
  DomainMap domain_map(fusion);
  for (auto out_tv : ir_utils::filterByType<TensorView>(fusion->outputs())) {
    if (domain_map.isValidReference(out_tv)) {
      out_tvs.push_back(out_tv);
    }
  }
  TORCH_INTERNAL_ASSERT(
      !out_tvs.empty(), "No valid reference outputs were found!");

  for (auto out_tv : out_tvs) {
    int n_dims = 0;
    for (auto id : out_tv->getMaybeRFactorDomain()) {
      if (id->isReduction() || id->isBroadcast()) {
        continue;
      }
      n_dims++;
    }
    if (n_dims > max_dims) {
      largest_out = out_tv;
      max_dims = n_dims;
    }
  }

  TORCH_INTERNAL_ASSERT(largest_out != nullptr);

  const int64_t device_multiprocessor_count =
      (int64_t)at::cuda::getCurrentDeviceProperties()->multiProcessorCount;

  // TODO: Set to 1?
  int64_t max_input_dtype_size = 2;
  size_t n_tensors = 0;

  for (auto inp : in_tvs) {
    max_input_dtype_size = std::max(
        max_input_dtype_size,
        (int64_t)dataTypeSize(inp->getDataType().value(), index_type));
    n_tensors++;
  }
  n_tensors += std::distance(out_tvs.begin(), out_tvs.end());

  auto ref_root = largest_out->getMaybeRFactorDomain();
  std::vector<int64_t> elem_counts(ref_root.size(), 1);
  int64_t n_elems = 1;
  for (size_t ref_i = 0; ref_i < ref_root.size(); ref_i++) {
    auto inferred_val =
        runtime_info.expressionEvaluator().evaluate(ref_root[ref_i]->extent());
    TORCH_INTERNAL_ASSERT(
        inferred_val.has_value(),
        "Error inferring size for pointwise scheduler: ",
        ref_root[ref_i]->extent()->toInlineString());
    elem_counts[ref_i] = inferred_val.value();
    n_elems *= inferred_val.value();
  }

  // If zero dimensional or zero size, return default parameters
  if (TensorDomain::noReductions(
          TensorDomain::noBroadcasts(largest_out->domain()->domain()))
              .size() == 0 ||
      n_elems == 0) {
    auto vectorizable_inputs_outputs_entry = HeuristicSummaryEntry<
        HeuristicCompileTime::VectorizableInputsAndOutputs>(data_cache, []() {
      return std::make_unique<std::vector<TensorView*>>();
    });
    vectorizable_inputs_outputs_entry.get();

    auto broadcast_byte_multiples_entry =
        HeuristicSummaryEntry<HeuristicCompileTime::BroadcastMultiples>(
            data_cache, []() {
              return std::make_unique<
                  std::vector<scheduler_utils::BroadcastMultiple>>();
            });
    broadcast_byte_multiples_entry.get();

    PointwiseParams params;
    params.tag = "Pointwise heuristics";
    return params;
  }

  // Find all vectorizable inputs/outputs
  auto vectorizable_inputs_outputs_entry =
      HeuristicSummaryEntry<HeuristicCompileTime::VectorizableInputsAndOutputs>(
          data_cache, [&largest_out]() {
            return std::make_unique<std::vector<TensorView*>>(
                scheduler_utils::getInputsOutputsWithInnerDim(
                    largest_out, true));
          });

  constexpr int64_t kSixteen = 16; // clang tidy

  auto max_unroll_factor = ceilDiv(
      // Available unrolling based on size of data type
      (int64_t)kSixteen / max_input_dtype_size,
      // Reduce max unrolling factor if we have many inputs/outputs to unroll
      // as it could start consuming a lot of registers.
      std::max(
          (scheduler_utils::lastPow2(
               (int64_t)vectorizable_inputs_outputs_entry.get().size()) >>
           2),
          (int64_t)1));

  // Don't unroll at the cost of getting a full wave on the GPU
  if (n_elems < device_multiprocessor_count * kThreadX &&
      max_unroll_factor > 1) {
    max_unroll_factor = std::min(
        max_unroll_factor,
        ceilDiv(n_elems, device_multiprocessor_count * kThreadX));
  }

  // If we use RNG don't unroll so we can do correctness testing
  if (fusion->isStochastic() && isDisabled(DisableOption::UnrollWithRng)) {
    max_unroll_factor = 1;
  }

  PointwiseParams params;
  params.tag = "Pointwise heuristics";

  /*
   * 2D pointwise scheduling logic. What is expected is there's some
   * broadcasting pattern which would make scheduling as a 2D problem more
   * efficient than scheduling simply as a 1D problem.
   *
   * Mapping count holds how many bytes are in each dimension for both inputs
   * and outputs relative to the reference tensor. What we're looking for is a
   * break point in reference_tvs dimensions which separates the outer dimension
   * and inner dimension of the problem mapped to 2D.
   *
   * break_point is computed assuming no reuse, ignoring parallelization
   * limitations, and simply figures out which point best separates broadcasted
   * dimensions. In other words, where's the point where we isolate the most
   * broadcasted elements to one side.
   *
   * Once a break point is found, simply schedule the pointwise op as 2D
   * balancing parallelization as best as possible.
   */

  // Ideal break point location
  int break_point = 0;

  // If break_point, mark if BIDy and BIDx should be positionally reversed
  // relative to root domains
  bool flip_grid_binding = false;

  // Elements on the right of break point (without break point all are on the
  // right)
  int64_t right_elem_count = 0;

  int64_t bdimx = kThreadX;

  // bdimy may be used if the right side of the break point is not large and we
  // need to expand block level parallelism into the left side of the break
  // point.
  int64_t bdimy = 1;

  // In 2D scheduler gdim_left is used to parallelize the left side of the break
  // point.
  int64_t gdim_left = 1;

  // gdim_right is used if there's too much parallelization in the right side of
  // the break point. We will expand grid parallelization into the right side of
  // the break point with gdim_left and use gdim_right for the left side of the
  // break point.
  int64_t gdim_right = 1;

  auto broadcast_byte_multiples_entry =
      HeuristicSummaryEntry<HeuristicCompileTime::BroadcastMultiples>(
          data_cache, [&largest_out, &index_type]() {
            return std::make_unique<
                std::vector<scheduler_utils::BroadcastMultiple>>(
                scheduler_utils::getBroadcastMultiples(
                    largest_out, index_type));
          });

  auto& broadcast_byte_multiples = broadcast_byte_multiples_entry.get();

  TORCH_INTERNAL_ASSERT(broadcast_byte_multiples.size() == ref_root.size());

  int64_t dtype_sum = 0;
  for (auto inp : ir_utils::filterByType<TensorView>(fusion->inputs())) {
    dtype_sum += dataTypeSize(inp->getDataType().value(), index_type);
  }
  for (auto out : ir_utils::filterByType<TensorView>(fusion->outputs())) {
    dtype_sum += dataTypeSize(out->getDataType().value(), index_type);
  }

  { // Figure out break point position. Empty scope, consider moving to a
    // separate function.
    //
    // How much would this transfer cost if it was done as a 1-D schedule
    int64_t transfer_size_1d = 1;

    for (const auto i : c10::irange(ref_root.size())) {
      transfer_size_1d = transfer_size_1d * elem_counts[i] * dtype_sum;
    }

    // If there isn't very much parallelism available, just use 1D scheduler
    if (true || n_elems * 2 > device_multiprocessor_count * kThreadX) {
      int64_t min_total_transfer = std::numeric_limits<int64_t>::max();

      for (const auto break_point_i : c10::irange(ref_root.size())) {
        // Number of elements in the right side of reference tv with
        // break_point_i
        int64_t cur_right_elem_count = 1;
        for (const auto right_i : c10::irange(break_point_i, ref_root.size())) {
          cur_right_elem_count = cur_right_elem_count * elem_counts[right_i];
        }

        auto cur_left_elem_count = n_elems / cur_right_elem_count;
        if (cur_left_elem_count <= 1) {
          continue;
        }

        auto lhs_byte_multiple =
            broadcast_byte_multiples[break_point_i].lhs_multiple;
        auto rhs_byte_multiple =
            broadcast_byte_multiples[break_point_i].rhs_multiple;

        // Estimate transfer cost with this break point
        int64_t cur_transfer_size = 1;
        int64_t right_transfer_size = 1;

        for (const auto left_i : c10::irange(break_point_i)) {
          cur_transfer_size =
              cur_transfer_size * elem_counts[left_i] * lhs_byte_multiple;
        }

        for (const auto right_i : c10::irange(break_point_i, ref_root.size())) {
          right_transfer_size =
              right_transfer_size * elem_counts[right_i] * rhs_byte_multiple;
        }
        cur_transfer_size *= right_transfer_size;

        //  Continue if this break point doesn't save at least 10% of 1D
        //  scheduling or isn't better than previous break_points found.
        if (cur_transfer_size >= min_total_transfer ||
            cur_transfer_size * 10 >= transfer_size_1d * 9) {
          continue;
        }

        // Need to be able to parallelize, don't use break if there's not
        // at least an unrolled warp.
        if (ceilDiv(cur_right_elem_count, max_unroll_factor) <=
            at::cuda::getCurrentDeviceProperties()->warpSize) {
          continue;
        }

        // If outer broadcast, or balanced broadcast:
        if (lhs_byte_multiple <= rhs_byte_multiple &&
            // If right transfer size is bigger than half of L2
            at::cuda::getCurrentDeviceProperties()->l2CacheSize <
                right_transfer_size * 2) {
          // flip BIDx and BIDy bindings
          flip_grid_binding = true;
        } else {
          flip_grid_binding = false;
        }
        // Min transfer found, start setting values
        bdimx = std::min(
            ceilDiv(cur_right_elem_count, max_unroll_factor), kThreadX);
        bdimy = 1;
        gdim_right = 1;
        // Put remainder in bdimy if there's at least a wave of grid level
        // parallelism.
        if (cur_left_elem_count > device_multiprocessor_count) {
          bdimy = kThreadX / bdimx;
        }
        auto remainder_left = ceilDiv(cur_left_elem_count, bdimy);
        auto remainder_right =
            ceilDiv(cur_right_elem_count, bdimx * max_unroll_factor);
        // Use this break point
        break_point = static_cast<int>(break_point_i);
        min_total_transfer = cur_transfer_size;
        right_elem_count = cur_right_elem_count;

        gdim_left = remainder_left;
        gdim_right = remainder_right;
      }
    }
  }

  // Vectorizing innermost domains

  // Don't try to vectorize if it's not recommended
  params.unroll_factor = 1;

  // Compute maximum vectorize factor that can be used
  size_t vectorize_factor = max_unroll_factor;
  auto& vectorizable_inputs_outputs = vectorizable_inputs_outputs_entry.get();

  for (auto tv : vectorizable_inputs_outputs) {
    const auto tv_vectorize_factor =
        runtime_info.getInnerDimVectorizableWidth(tv);
    vectorize_factor = std::min(vectorize_factor, tv_vectorize_factor);
  }

  // Try expanding vectorization to contig merged domains
  auto expanded_vector_word_size =
      scheduler_utils::expandVectorizationToContigMergedDomains(
          fusion,
          runtime_info,
          vectorizable_inputs_outputs,
          largest_out,
          break_point,
          vectorize_factor);

  expanded_vector_word_size = std::min(
      static_cast<size_t>(max_unroll_factor), expanded_vector_word_size);

  if (expanded_vector_word_size > vectorize_factor) {
    vectorize_factor = expanded_vector_word_size;
  }

  if (vectorize_factor == 1) {
    params.vectorize = false;
    params.unroll_factor = max_unroll_factor;
  } else {
    params.vectorize = true;
    params.unroll_factor = vectorize_factor;
  }

  TORCH_INTERNAL_ASSERT(right_elem_count > 0 || break_point == 0);
  TORCH_INTERNAL_ASSERT(!(bdimy > 1 && gdim_right > 1));

  params.break_point = break_point;
  params.flip_grid_binding = flip_grid_binding;
  params.split_block = bdimy > 1;

  params.lparams.bind(bdimx, ParallelType::TIDx);
  if (params.split_block) {
    params.lparams.bind(bdimy, ParallelType::TIDy);
  }
  if ((flip_grid_binding && gdim_right > 65535) ||
      (!flip_grid_binding && gdim_left > 65535)) {
    params.split_grid_y_dim = true;
  }

  if (isDebugDumpEnabled(DebugDumpOption::SchedulerDebug)) {
    std::cerr << "\n===== Pointwise Stats ========\n"
              << "num_elems: " << n_elems << "\n"
              << "elem_counts: " << elem_counts << "\n"
              << "n_tensor_inputs: " << n_tensors << "\n"
              << "max_input_dtype_size: " << max_input_dtype_size << "\n"
              << "vectorize_factor: " << vectorize_factor << std::endl;
    std::cerr << "broadcast_byte_multiples: ";
    for (auto multiple : broadcast_byte_multiples) {
      std::cerr << "(" << multiple.lhs_multiple << ", " << multiple.rhs_multiple
                << "), ";
    }
    std::cerr << "LHS elems: "
              << (right_elem_count > 0 ? n_elems / right_elem_count : 0)
              << " RHS elems: " << right_elem_count << std::endl;
    std::cerr << std::endl;
    std::cerr << params.toString() << std::endl;
  }

  return params;
}

// TODO: remove or return launch parameters
LaunchParams schedulePointwise(
    Fusion* fusion,
    const at::ArrayRef<c10::IValue>& runtime_inputs) {
  FUSER_PERF_SCOPE("scheduleFusion");
  auto params = getPointwiseHeuristics(fusion, runtime_inputs);
  TORCH_INTERNAL_ASSERT(
      params.has_value(), "Could not schedule pointwise operation.");
  schedulePointwise(fusion, params.value());
  return params.value().lparams;
}

namespace {
// Returns number of non-reduction/non-broadcast dims in rfactor domain
size_t nRootDims(const TensorView* tv) {
  auto root_dom = tv->getMaybeRFactorDomain();
  size_t tv_n_dims = 0;
  for (auto dim : root_dom) {
    if (!dim->isReduction() && !dim->isBroadcast()) {
      tv_n_dims++;
    }
  }
  return tv_n_dims;
}
} // namespace

bool hasReferenceTensorView(Fusion* fusion) {
  return DomainMap::hasReferenceTensorView(fusion);
}

// TODO: Inline intermediate operations (avoid inlining unrolled/vectorized
// input/output caches)
void schedulePointwise(Fusion* fusion, const PointwiseParams& params) {
  FusionGuard fg(fusion);

  // Make sure we don't have global memory set on intermediate tensors from
  // fusion segmentation
  scheduler_utils::clearMemorySpace(fusion);

  // maybe has_reduction for scheduling should be done on a per output tensor
  // basis.
  TORCH_INTERNAL_ASSERT(
      ir_utils::getReductionOps(fusion /*, ignore_trivial=true */).empty(),
      "This scheduler only handles pointwise ops.");

  // For intermediate outputs, apply cacheFork
  auto outs = fusion->outputs();
  for (const auto output : outs) {
    if (!output->uses().empty() && output->definition() != nullptr) {
      if (output->getValType().value() == ValType::TensorView) {
        output->as<TensorView>()->cacheFork();
      }
    }
  }

  std::vector<TensorView*> input_tvs;
  {
    auto filtered_tvs = ir_utils::filterByType<TensorView>(fusion->inputs());
    // Remove hanging tensor views
    for (auto tv : filtered_tvs) {
      if (tv->uses().empty()) {
        continue;
      }
      input_tvs.push_back(tv);
    }
  }
  auto output_tvs = ir_utils::filterByType<TensorView>(fusion->outputs());

  size_t max_dims = 0;
  for (auto inp : input_tvs) {
    max_dims = std::max(nRootDims(inp), max_dims);
  }

  for (auto out : output_tvs) {
    max_dims = std::max(nRootDims(out), max_dims);
  }

  // If everything is zero dim tensors, just return.
  if (max_dims == 0) {
    return;
  }

  DomainMap domain_map(fusion);
  TensorView* reference_tv =
      domain_map.findReferenceTensorView(params.break_point);

  TORCH_INTERNAL_ASSERT(
      reference_tv != nullptr,
      "Could not find a fully broadcasted output to reference schedule on.");

  IterDomain* inner_most_id = nullptr;
  for (auto it = reference_tv->domain()->domain().rbegin();
       it != reference_tv->domain()->domain().rend();
       it++) {
    if ((*it)->isReduction()) {
      continue;
    }
    if ((*it)->isBroadcast() && inner_most_id == nullptr) {
      inner_most_id = *it;
    }
    inner_most_id = *it;
    break;
  }

  TORCH_INTERNAL_ASSERT(inner_most_id != nullptr);

  // Caches of inputs
  std::vector<TensorView*> cached_inputs;

  // Output, cacheBefore of output
  std::vector<std::pair<TensorView*, TensorView*>> cached_outputs;

  // Track what should be vectorized versus unrolled
  std::unordered_set<TensorView*> vectorized_tensor;

  // Figure out which inputs to cache for unrolling or vectorization
  for (auto inp : input_tvs) {
    if (inp->uses().empty() || inp->isFusionOutput()) {
      continue;
    }
    cached_inputs.emplace_back(inp->cacheAfter());
  }

  // Figure out which outputs to cache for unrolling or vectorization
  for (auto out : output_tvs) {
    if (out->definition() == nullptr) {
      continue;
    }
    cached_outputs.emplace_back(std::make_pair(out, out->cacheBefore()));
  }

  auto all_tvs = ir_utils::allTvs(fusion);

  // Merge right side of break point
  int rhs_i = -1;
  for (int i = (int)reference_tv->nDims(); i > (int)params.break_point; i--) {
    auto axis_i = i - 1;
    if (reference_tv->axis(axis_i)->isBroadcast() ||
        reference_tv->axis(axis_i)->isReduction()) {
      continue;
    }
    if (rhs_i == -1) {
      rhs_i = axis_i;
    } else {
      reference_tv->merge(axis_i, rhs_i);
      rhs_i = axis_i;
    }
  }
  if (rhs_i >= 0) {
    // If there's an rhs
    reference_tv->reorder({{rhs_i, -1}});
  }

  // Merge left side of break point
  int lhs_i = -1;
  for (int i = (int)params.break_point; i > 0; i--) {
    auto axis_i = i - 1;
    if (reference_tv->axis(axis_i)->isBroadcast() ||
        reference_tv->axis(axis_i)->isReduction()) {
      continue;
    }
    if (lhs_i == -1) {
      lhs_i = axis_i;
    } else {
      reference_tv->merge(axis_i, lhs_i);
      lhs_i = axis_i;
    }
  }

  if (params.break_point) {
    // 2D parallelization scheme
    TORCH_INTERNAL_ASSERT(rhs_i >= 0 && lhs_i >= 0);

    // Right (inner merged) dimension is at inner most position, left (outer
    // merged) dimension is at lhs_i. Order as [lhs_i, rhs_i, unmerged...]
    reference_tv->reorder({{lhs_i, 0}, {-1, 1}});

    if (params.vectorize) {
      reference_tv->split(1, params.unroll_factor);
      reference_tv->split(1, NamedScalar::getParallelDim(ParallelType::TIDx));
      reference_tv->split(0, 1);
      // [outer, Unswitch | i-remainder, TIDx, Vectorization]
      reference_tv->axis(1)->parallelize(ParallelType::Unswitch);
      reference_tv->axis(3)->parallelize(ParallelType::TIDx);

      // Aggressively mark with vectorized and cleanup later. That way we
      // don't have to manually specify parallelization outside the reference.
      reference_tv->axis(4)->parallelize(ParallelType::Vectorize);

      // [outer, Unswitch | i-remainder, TIDx, Vectorization]
      // To make consistent with unrolling:
      reference_tv->reorder({{1, 2}, {2, 1}, {3, 4}, {4, 3}});
      //[outer | i-remainder, Unswitch, Vectorization, TIDx]
    } else {
      reference_tv->split(1, NamedScalar::getParallelDim(ParallelType::TIDx));
      reference_tv->split(1, params.unroll_factor);

      reference_tv->split(0, 1);
      // [outer, unswitch | i-remainder, unroll, TIDx ]
      reference_tv->reorder({{1, 2}});
      // [outer, i-remainder, unswitch, unroll, TIDx ]
      reference_tv->axis(2)->parallelize(ParallelType::Unswitch);
      reference_tv->axis(4)->parallelize(ParallelType::TIDx);

      //[outer | i-remainder, Unswitch, Unroll, TIDx]
    }

    // Move out of the way to furthest left point
    reference_tv->reorder({{1, 0}});

    //[i-remainder | outer | Unswitch, Unroll, TIDx]
    if (params.split_block) {
      reference_tv->split(1, NamedScalar::getParallelDim(ParallelType::TIDy));
      if (params.flip_grid_binding) {
        // [BIDy | BIDx, TIDy | Unswitch, Unroll, TIDx]
        reference_tv->axis(1)->parallelize(ParallelType::BIDx);
        reference_tv->axis(2)->parallelize(ParallelType::TIDy);
        if (params.split_grid_y_dim) {
          // [i-remainder, BIDy{65535} | BIDx, TIDy | Unswitch, Unroll, TIDx]
          reference_tv->split(0, 65535);
          reference_tv->axis(1)->parallelize(ParallelType::BIDy);
        } else {
          reference_tv->axis(0)->parallelize(ParallelType::BIDy);
        }
      } else {
        // [BIDx | BIDy TIDy | Unswitch, Unroll, TIDx]
        reference_tv->axis(0)->parallelize(ParallelType::BIDx);
        reference_tv->axis(2)->parallelize(ParallelType::TIDy);
        if (params.split_grid_y_dim) {
          // [BIDx | i-remainder, BIDy{65535}, TIDy | Unswitch, Unroll, TIDx]
          reference_tv->split(1, 65535);
          reference_tv->axis(2)->parallelize(ParallelType::BIDy);
        } else {
          reference_tv->axis(1)->parallelize(ParallelType::BIDy);
        }
      }
    } else {
      // [BIDy | BIDx | Unswitch, Unroll, TIDx]
      if (params.flip_grid_binding) {
        // [BIDy | BIDx | Unswitch, Unroll, TIDx]
        reference_tv->axis(1)->parallelize(ParallelType::BIDx);
        if (params.split_grid_y_dim) {
          // [i-remainder, BIDy{65535} | BIDx | Unswitch, Unroll, TIDx]
          reference_tv->split(0, 65535);
          reference_tv->axis(1)->parallelize(ParallelType::BIDy);
        } else {
          reference_tv->axis(0)->parallelize(ParallelType::BIDy);
        }
      } else {
        // [BIDx | BIDy | Unswitch, Unroll, TIDx]
        reference_tv->axis(0)->parallelize(ParallelType::BIDx);
        if (params.split_grid_y_dim) {
          // [BIDx | i-remainder, BIDy{65535} | Unswitch, Unroll, TIDx]
          reference_tv->split(1, 65535);
          reference_tv->axis(2)->parallelize(ParallelType::BIDy);
        } else {
          reference_tv->axis(1)->parallelize(ParallelType::BIDy);
        }
      }
    }
  } else {
    // 1D Scheduler
    TORCH_INTERNAL_ASSERT(rhs_i >= 0 && lhs_i == -1);

    // right hand side exists and is the only axis we care to schedule, move
    // it from the inner most position to left most. Order as [rhs_i,
    // unmerged...]
    reference_tv->reorder({{-1, 0}});

    if (params.vectorize) {
      // Vectorize
      reference_tv->split(0, params.unroll_factor);
      // Unswitch
      reference_tv->split(0, 1);
      // Threads
      reference_tv->split(0, kThreadX);

      reference_tv->axis(0)->parallelize(ParallelType::BIDx);
      reference_tv->axis(1)->parallelize(ParallelType::TIDx);
      reference_tv->axis(2)->parallelize(ParallelType::Unswitch);
      // Aggressively mark with vectorized and cleanup later. That way we
      // don't have to manually specify parallelization outside the reference.
      reference_tv->axis(3)->parallelize(ParallelType::Vectorize);

      //[BIDx, TIDx, Unswitch, Vectorization]
      // To make consistent with unrolling:
      reference_tv->reorder({{1, 3}, {2, 1}, {3, 2}});
      //[BIDx, Unswitch, Vectorization, TIDx]
    } else {
      // Threads
      reference_tv->split(0, kThreadX);
      // Unroll
      reference_tv->split(0, params.unroll_factor);
      // Unswitch
      reference_tv->split(0, 1);

      // [BIDx, Unswitch, Unroll, TIDx]
      reference_tv->axis(0)->parallelize(ParallelType::BIDx);
      reference_tv->axis(1)->parallelize(ParallelType::Unswitch);
      reference_tv->axis(3)->parallelize(ParallelType::TIDx);
    }
  }

  TransformPropagator propagator(reference_tv);
  MaxRootDomainInfoSpanningTree(reference_tv).traverse(&propagator);
  scheduler_utils::parallelizeAllLike(reference_tv, all_tvs);

  if (params.vectorize) {
    // Grab all tensor views that should be vectorized
    auto vectorized_tvs =
        scheduler_utils::getInputsOutputsWithInnerDim(reference_tv, true);
    // Going to move inputs to consumers of inputs, need a copy as we'll modify
    // the original.
    {
      auto vectorized_tvs_copy = vectorized_tvs;
      for (auto inp : vectorized_tvs_copy) {
        if (!inp->isFusionInput()) {
          continue;
        }
        vectorized_tvs.erase(
            std::find(vectorized_tvs.begin(), vectorized_tvs.end(), inp));
        auto consumer_tvs = ir_utils::consumerTvsOf(inp);
        vectorized_tvs.insert(
            vectorized_tvs.end(), consumer_tvs.begin(), consumer_tvs.end());
      }
    }
    // Clear vectorize on tensors that shouldn't have it
    for (auto tv : all_tvs) {
      if (std::find(vectorized_tvs.begin(), vectorized_tvs.end(), tv) ==
          vectorized_tvs.end()) {
        for (auto id : tv->domain()->domain()) {
          if (id->getParallelType() == ParallelType::Vectorize) {
            id->parallelize(ParallelType::Serial);
          }
        }
      }
    }
  }

  // Compute at into cached inputs
  std::vector<TensorView*> consumers_of_cached_inputs;
  // Cache of input, and one of its consumers
  std::vector<std::pair<TensorView*, TensorView*>> input_cache_and_consumer;
  {
    // Avoid duplicate additions, so track what we add
    std::unordered_set<TensorView*> added;
    for (auto cached_input : cached_inputs) {
      auto consumer_tvs = ir_utils::consumerTvsOf(cached_input);
      TORCH_INTERNAL_ASSERT(
          consumer_tvs.size(),
          "Input was not succesfully filtered out for scheduling but wasn't used.");

      // Grab a consumer which will be used for computeAt structure of cached
      // input into a consumer
      input_cache_and_consumer.emplace_back(
          std::make_pair(cached_input, consumer_tvs[0]));

      // Grab all consumers which will be used for inlining computeAt for the
      // body of the computation (excluding caching inputs/outputs)
      for (auto consumer_tv : consumer_tvs) {
        // Don't duplicate
        if (added.insert(consumer_tv).second) {
          consumers_of_cached_inputs.emplace_back(consumer_tv);
        }
      }
    }
  }

  for (auto entry : input_cache_and_consumer) {
    // Compute at inside unswitch position:
    auto input_cache = entry.first;
    auto input_cache_consumer = entry.second;

    auto unswitch_it = std::find_if(
        input_cache_consumer->domain()->domain().begin(),
        input_cache_consumer->domain()->domain().end(),
        [](IterDomain* id) {
          return id->getParallelType() == ParallelType::Unswitch;
        });
    auto unswitch_pos =
        unswitch_it == input_cache_consumer->domain()->domain().end()
        ? -1
        : std::distance(
              input_cache_consumer->domain()->domain().begin(), unswitch_it) +
            1;

    input_cache->computeAt(
        input_cache_consumer, unswitch_pos, ComputeAtMode::BestEffort);
  }

  // Producers for inlined computeAt
  std::vector<TensorView*> compute_from = consumers_of_cached_inputs;

  // Consumers for inlined computeAt
  std::vector<TensorView*> compute_to;
  // Compute at cached outputs
  //[BIDx, Unswitch, Vectorization, TIDx]
  for (auto entry : cached_outputs) {
    auto cached_output = entry.second;
    auto output = entry.first;

    auto unswitch_it = std::find_if(
        output->domain()->domain().begin(),
        output->domain()->domain().end(),
        [](IterDomain* id) {
          return id->getParallelType() == ParallelType::Unswitch;
        });
    auto unswitch_pos = unswitch_it == output->domain()->domain().end()
        ? -1
        : std::distance(output->domain()->domain().begin(), unswitch_it) + 1;

    cached_output->computeAt(output, unswitch_pos, ComputeAtMode::BestEffort);
    compute_to.push_back(cached_output);
  }

  scheduler_utils::computeAtBetween(
      compute_from, compute_to, -1, ComputeAtMode::BestEffort);
}

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
