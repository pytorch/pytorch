#include <torch/csrc/jit/codegen/cuda/scheduler/transpose.h>

#include <torch/csrc/jit/codegen/cuda/executor_utils.h>
#include <torch/csrc/jit/codegen/cuda/inline_propagator.h>
#include <torch/csrc/jit/codegen/cuda/instrumentation.h>
#include <torch/csrc/jit/codegen/cuda/ir_iostream.h>
#include <torch/csrc/jit/codegen/cuda/ir_utils.h>
#include <torch/csrc/jit/codegen/cuda/lower_utils.h>
#include <torch/csrc/jit/codegen/cuda/scheduler/pointwise_utils.h>
#include <torch/csrc/jit/codegen/cuda/scheduler/registry.h>
#include <torch/csrc/jit/codegen/cuda/scheduler/utils.h>
#include <torch/csrc/jit/codegen/cuda/scheduler/vectorize_helper.h>
#include <torch/csrc/jit/codegen/cuda/transform_replay.h>
#include <torch/csrc/jit/codegen/cuda/utils.h>

#include <ATen/cuda/CUDAContext.h>

#include <algorithm>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

namespace {

constexpr int64_t kThreadsPerBlock = 128;

// DomainMap uses the ComputeAtMap to find a reference TensorView
// that maps to all iterDomains in the fusion.
class DomainMap : public pointwise_utils::DomainMap {
 public:
  using pointwise_utils::DomainMap::DomainMap;

  TensorView* findReferenceFor(const std::vector<TensorView*>& group) const {
    TensorView* result = nullptr;
    int max_dims = -1;
    for (auto tv : group) {
      if (isValidReference(tv)) {
        int dims = pointwise_utils::nRootDims(tv);
        if (dims > max_dims) {
          result = tv;
          max_dims = dims;
        }
      }
    }
    return result;
  }

  static bool hasAtLeastTwoValidGroups(Fusion* fusion) {
    FusionGuard fg(fusion);
    DomainMap domain_map(fusion);
    auto grouped_inputs_outputs = domain_map.groupInputsOutputsByInnerDim();
    if (grouped_inputs_outputs.size() < 2) {
      return false;
    }
    return domain_map.findReferenceFor(grouped_inputs_outputs[0]) != nullptr &&
        domain_map.findReferenceFor(grouped_inputs_outputs[1]) != nullptr;
  }

  int getPosMappedTo(TensorView* tv, IterDomain* id) {
    const auto& dom = tv->domain()->domain();
    for (auto i : c10::irange(dom.size())) {
      if (areExactMapped(tv->axis(i), id)) {
        return i;
      }
    }
    TORCH_INTERNAL_ASSERT(
        false, "Can not find ID mapped to ", id, " in tensor ", tv);
  }

  // Group inputs and outputs of a fusion by its inner most domain. For example
  //   inputs: t0, t1
  //   t2 = transpose(t1)
  //   t3 = t0 + t2
  //   t4 = sin(t0)
  //   t5 = cos(t1)
  //   outputs: t3, t4, t5
  //
  // Then we should have group {t0, t3, t4} and {t1, t5}
  //
  // The returned groups are sorted in descending size. If the sizes of two
  // group are equal, then we sort them by their members in the following order:
  //   output[0], output[1], ..., input[0], input[1], ...
  // That is, {ouput[0], output[2]} will be in front of {ouput[1], output[3]}
  // The order here must be deterministic, because in transpose heuristics, we
  // have `vectorize_factor1` and `vectorize_factor2` and we need to be sure
  // that `1` and `2` are assigned to the same group across runs.
  std::vector<std::vector<TensorView*>> groupInputsOutputsByInnerDim() const {
    std::vector<std::vector<TensorView*>> groups;
    auto output_tvs = ir_utils::filterByType<TensorView>(fusion_->outputs());
    auto input_tvs = ir_utils::filterByType<TensorView>(fusion_->inputs());
    std::unordered_map<size_t, IterDomain*> group_to_inner_dim_map;
    decltype(input_tvs)* tv_filtered_group[2] = {&output_tvs, &input_tvs};
    for (auto view : tv_filtered_group) {
      for (auto tv : *view) {
        auto inner_most_id = scheduler_utils::innerMostRootDim(tv);
        bool found = false;
        for (auto gi : c10::irange(groups.size())) {
          auto& g = groups[gi];
          auto group_inner_dim = group_to_inner_dim_map.at(gi);
          if (areExactMapped(inner_most_id, group_inner_dim)) {
            g.emplace_back(tv);
            found = true;
            break;
          }
        }
        if (!found) {
          group_to_inner_dim_map[groups.size()] = inner_most_id;
          groups.push_back({tv});
        }
      }
    }
    std::stable_sort(
        groups.begin(),
        groups.end(),
        [](const std::vector<TensorView*>& v1,
           const std::vector<TensorView*>& v2) {
          return v1.size() > v2.size();
        });
    return groups;
  }
};

} // namespace

bool hasAtLeastTwoValidGroups(Fusion* fusion) {
  return DomainMap::hasAtLeastTwoValidGroups(fusion);
}

std::shared_ptr<TransposeParams> getTransposeHeuristics(
    Fusion* fusion,
    const at::ArrayRef<c10::IValue>& runtime_inputs,
    HeuristicSummary* data_cache) {
  SchedulerRuntimeInfo runtime_info(fusion, runtime_inputs, true);
  return getTransposeHeuristics(fusion, runtime_info, data_cache);
}

std::shared_ptr<TransposeParams> getTransposeHeuristics(
    Fusion* fusion,
    SchedulerRuntimeInfo& runtime_info,
    HeuristicSummary* data_cache) {
  FUSER_PERF_SCOPE("getTransposeHeuristics");

  FusionGuard fg(fusion);

  // Incase any buffer is of type DataType::Index
  DataType index_type = indexModeToDtype(runtime_info.getIndexMode());

  auto domain_map_entry =
      HeuristicSummaryEntry<HeuristicCompileTime::DomainMap>(
          data_cache,
          [fusion]() { return std::make_unique<DomainMap>(fusion); });
  const auto& domain_map = dynamic_cast<DomainMap&>(domain_map_entry.get());

  auto grouped_inputs_outputs_entry =
      HeuristicSummaryEntry<HeuristicCompileTime::InputsOutputsInnerDimGroups>(
          data_cache, [&domain_map]() {
            return std::make_unique<std::vector<std::vector<TensorView*>>>(
                domain_map.groupInputsOutputsByInnerDim());
          });
  auto grouped_inputs_outputs = grouped_inputs_outputs_entry.get();

  TORCH_INTERNAL_ASSERT(
      grouped_inputs_outputs.size() >= 2,
      "Can not find mismatched inner most dim, should use pointwise scheduler.");

  auto largest_entry =
      HeuristicSummaryEntry<HeuristicCompileTime::ReferenceTensors>(
          data_cache, [&domain_map, &grouped_inputs_outputs]() {
            std::vector<TensorView*> data{
                domain_map.findReferenceFor(grouped_inputs_outputs[0]),
                domain_map.findReferenceFor(grouped_inputs_outputs[1])};
            return std::make_unique<std::vector<TensorView*>>(std::move(data));
          });
  auto& largest = largest_entry.get();
  TORCH_INTERNAL_ASSERT(largest.size() == 2);
  TensorView* largest1 = largest[0];
  TensorView* largest2 = largest[1];
  TORCH_INTERNAL_ASSERT(
      largest1 != nullptr, "Unable to find reference tensor for group 1");
  TORCH_INTERNAL_ASSERT(
      largest2 != nullptr, "Unable to find reference tensor for group 2");

  const int64_t device_multiprocessor_count =
      (int64_t)at::cuda::getCurrentDeviceProperties()->multiProcessorCount;

  int64_t max_input_dtype_size = 1;

  size_t n_input_tensors = 0;
  for (auto inp : ir_utils::filterByType<TensorView>(fusion->inputs())) {
    max_input_dtype_size = std::max(
        max_input_dtype_size,
        (int64_t)dataTypeSize(inp->getDataType().value(), index_type));
    n_input_tensors++;
  }

  auto ref_root = largest1->getMaybeRFactorDomain();
  int64_t n_elems = 1;
  for (size_t ref_i = 0; ref_i < ref_root.size(); ref_i++) {
    auto inferred_val =
        runtime_info.expressionEvaluator().evaluate(ref_root[ref_i]->extent());
    TORCH_INTERNAL_ASSERT(
        inferred_val.has_value(),
        "Error inferring size for pointwise scheduler: ",
        ref_root[ref_i]->extent()->toInlineString());
    n_elems *= inferred_val.value().as<int64_t>();
  }

  auto params = std::make_shared<TransposeParams>("Transpose heuristics");

  // Note [vectorization and unroll of input and output]
  //
  // The choice of vectorization size, block size and tile sizes needs to be
  // consistent with each other. Consider the following:
  //
  // The number of threads in one block is
  //   num_threads = blockDim.x * blockDim.y
  // and the number of elements per each tile is
  //   num_elems_per_tile = params->tile_size1 * params->tile_size2
  // So each thread needs to process
  //   num_elems_per_thread = num_elems_per_tile / num_threads
  // elements. That is, once the tile sizes and block size are determined, the
  // `num_elems_per_thread` is determined, regardless of vectorizability of
  // input/output tensors.
  //
  // To make the selection of tile sizes othogonal to vectorizability, we
  // support having both vectorization and unrolling in the same tensor. For
  // example, if we have num_elems_per_tile == 1024 and num_threads = 256, then
  // we have num_elems_per_thread being 4. And if we have vector size 2, then we
  // will do unroll 2 * vectorize 2 at the same tensor.
  //
  // Also, since the inner most dim of different groups are not the same, it is
  // natural to consider their vectorizability separately and allow them to have
  // different vectorize/unroll sizes.

  constexpr int64_t kSixteen = 16; // clang tidy

  auto max_unroll_factor = ceilDiv(
      // Available unrolling based on size of data type
      (int64_t)kSixteen / max_input_dtype_size,
      // Reduce max unrolling factor if we have many inputs/outputs to unroll
      // as it could start consuming a lot of registers.
      std::max(
          (scheduler_utils::lastPow2(
               (int64_t)grouped_inputs_outputs[0].size() +
               (int64_t)grouped_inputs_outputs[1].size()) >>
           2),
          (int64_t)1));

  // Don't unroll at the cost of getting a full wave on the GPU
  auto max_unroll_factor_occupancy = ceilDiv(
      n_elems,
      device_multiprocessor_count * params->tile_size1 * params->tile_size2);
  max_unroll_factor = std::min(max_unroll_factor, max_unroll_factor_occupancy);

  // Compute maximum vectorize factor that can be used
  size_t vectorize_factor1 = max_unroll_factor;
  size_t vectorize_factor2 = max_unroll_factor;

  for (auto tv : grouped_inputs_outputs[0]) {
    const auto tv_vectorize_factor =
        runtime_info.getInnerDimVectorizableWidth(tv);
    vectorize_factor1 = std::min(vectorize_factor1, tv_vectorize_factor);
  }
  for (auto tv : grouped_inputs_outputs[1]) {
    const auto tv_vectorize_factor =
        runtime_info.getInnerDimVectorizableWidth(tv);
    vectorize_factor2 = std::min(vectorize_factor2, tv_vectorize_factor);
  }

  // Try expanding vectorization to contig merged domains
  auto expanded_vector_word_size1 =
      scheduler_utils::expandVectorizationToContigMergedDomains(
          fusion,
          runtime_info,
          grouped_inputs_outputs[0],
          largest1,
          0,
          vectorize_factor1);
  auto expanded_vector_word_size2 =
      scheduler_utils::expandVectorizationToContigMergedDomains(
          fusion,
          runtime_info,
          grouped_inputs_outputs[1],
          largest2,
          0,
          vectorize_factor2);

  expanded_vector_word_size1 = std::min(
      static_cast<size_t>(max_unroll_factor), expanded_vector_word_size1);
  expanded_vector_word_size2 = std::min(
      static_cast<size_t>(max_unroll_factor), expanded_vector_word_size2);

  vectorize_factor1 = std::max(vectorize_factor1, expanded_vector_word_size1);
  vectorize_factor2 = std::max(vectorize_factor2, expanded_vector_word_size2);

  params->vectorize_factor1 = vectorize_factor1;
  params->vectorize_factor2 = vectorize_factor2;

  // TODO: should we adjust tile size according to max_unroll_factor?

  params->lparams.bind(kThreadsPerBlock, ParallelType::TIDx);

  if (isDebugDumpEnabled(DebugDumpOption::SchedulerDebug)) {
    std::cerr << "\n===== Transpose Stats ========\n"
              << "num_elems: " << n_elems << "\n"
              << "n_input_tensors: " << n_input_tensors << "\n"
              << "max_input_dtype_size: " << max_input_dtype_size << "\n"
              << "group 1: " << ir_utils::toString(grouped_inputs_outputs[0])
              << "\n"
              << "group 2: " << ir_utils::toString(grouped_inputs_outputs[1])
              << std::endl;
    std::cerr << std::endl;
    std::cerr << params->toString() << std::endl;
  }

  return params;
}

// TODO: remove or return launch parameters
LaunchParams scheduleTranspose(
    Fusion* fusion,
    const at::ArrayRef<c10::IValue>& runtime_inputs) {
  FUSER_PERF_SCOPE("scheduleFusion");
  auto params = getTransposeHeuristics(fusion, runtime_inputs);
  TORCH_INTERNAL_ASSERT(
      params != nullptr, "Could not schedule pointwise operation.");
  scheduleTranspose(fusion, *params);
  return params->lparams;
}

void scheduleTranspose(Fusion* fusion, const TransposeParams& params) {
  FusionGuard fg(fusion);

  // Make sure we don't have global memory set on intermediate tensors from
  // fusion segmentation
  scheduler_utils::clearMemorySpace(fusion);

  // maybe has_reduction for scheduling should be done on a per output tensor
  // basis.
  // TODO: add support for trivial reduction
  TORCH_INTERNAL_ASSERT(
      ir_utils::getReductionOps(fusion, /*ignore_trivial=*/false).empty(),
      "This scheduler only handles pointwise ops.");

  // Cache inputs
  auto cached_inputs = scheduler_utils::cacheInputs(fusion, true);

  // Cache and fork outputs
  auto cached_outputs = scheduler_utils::cacheAndForkOutputs(fusion, true);

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
    max_dims = std::max(pointwise_utils::nRootDims(inp), max_dims);
  }

  for (auto out : output_tvs) {
    max_dims = std::max(pointwise_utils::nRootDims(out), max_dims);
  }

  // If everything is zero dim tensors, just return.
  if (max_dims == 0) {
    return;
  }

  DomainMap domain_map(fusion);
  auto grouped_inputs_outputs = domain_map.groupInputsOutputsByInnerDim();
  TORCH_INTERNAL_ASSERT(grouped_inputs_outputs.size() >= 2);

  // We need something similar to `cacheFork` for input tensors in group 2. We
  // need this because we will want to propagate to the entire DAG except group
  // 2 and its cached inputs, so we need to make sure the DAG is still connected
  // if we remove group and its cached inputs. For example
  //    t0
  //    |
  //   cache
  //   |  |
  //  t1  t2
  // if groups = {{t1, t2}, {t0}}, then removing {t0, cache} from the DAG will
  // make it disconnected.
  std::unordered_set<TensorView*> group2_and_cached_inputs(
      grouped_inputs_outputs[1].begin(), grouped_inputs_outputs[1].end());
  for (auto tv : grouped_inputs_outputs[1]) {
    if (tv->isFusionInput()) {
      auto existing_cache = ir_utils::consumerTvsOf(tv)[0];
      if (ir_utils::consumerTvsOf(existing_cache).size() > 1) {
        auto new_cache = tv->cacheAfter();
        new_cache->setMemoryType(MemoryType::Shared);
        group2_and_cached_inputs.emplace(new_cache);
      } else {
        existing_cache->setMemoryType(MemoryType::Shared);
        group2_and_cached_inputs.emplace(existing_cache);
      }
    }
  }
  // set cached outputs of group 2 to shared memory
  for (auto pair : cached_outputs) {
    auto cached_output = pair.first;
    auto output = pair.second;
    if (group2_and_cached_inputs.count(output) > 0) {
      cached_output->setMemoryType(MemoryType::Shared);
    }
  }

  TensorView* reference1 =
      domain_map.findReferenceFor(grouped_inputs_outputs[0]);
  TensorView* reference2 =
      domain_map.findReferenceFor(grouped_inputs_outputs[1]);

  TORCH_INTERNAL_ASSERT(
      reference1 != nullptr,
      "Could not find a fully broadcasted tensor to reference schedule on the first group.");

  TORCH_INTERNAL_ASSERT(
      reference2 != nullptr,
      "Could not find a fully broadcasted tensor to reference schedule on the second group.");

  auto inner_most_id1 = scheduler_utils::innerMostRootDim(reference1);
  auto inner_most_id2 = scheduler_utils::innerMostRootDim(reference2);

  auto inner_most_pos1_in_ref1 =
      domain_map.getPosMappedTo(reference1, inner_most_id1);
  auto inner_most_pos2_in_ref1 =
      domain_map.getPosMappedTo(reference1, inner_most_id2);

  // make tile
  // [..., I1, .., I2, ...]
  reference1->split(inner_most_pos1_in_ref1, params.tile_size1);
  reference1->reorder({{inner_most_pos1_in_ref1 + 1, -1}});
  reference1->split(inner_most_pos2_in_ref1, params.tile_size2);
  reference1->reorder({{inner_most_pos2_in_ref1 + 1, -1}});
  // [..., I1/tile1, .., I2/tile2, ..., tile1, tile2]

  // Merge remaining dimensions
  int lhs_i = -1;
  for (int i = (int)reference1->nDims() - 2; i > 0; i--) {
    auto axis_i = i - 1;
    if (lhs_i == -1) {
      lhs_i = axis_i;
    } else {
      reference1->merge(axis_i, lhs_i);
      lhs_i = axis_i;
    }
  }
  reference1->split(0, 1);
  // [merged_dim, 1, tile1, tile2]

  // parallelize non-tile dimensions
  reference1->axis(1)->parallelize(ParallelType::Unswitch);
  reference1->axis(0)->parallelize(ParallelType::BIDx);
  // [BIDx, Unswitch, tile1, tile2]

  // Propagate transformations so far to the entire DAG
  TransformPropagator propagator(reference1);
  MaxRootDomainInfoSpanningTree entire_dag(reference1);
  entire_dag.traverse(&propagator);
  scheduler_utils::parallelizeAllLike(reference1);

  // For a transpose scheduling, all we need is to bind threadIdx.x differently
  // for inputs and outputs. This swap of binding could happen at any tensor on
  // the path from input to output, especially, it does not have to be in the
  // transpose tensor. Here, we naively do the binding swap at cached
  // input/output for simplicity. We might need to find a better set of swap
  // tensors in the future to reduce shared memory usage.

  // transform tile for vectorization/unroll
  // See note [vectorization and unroll of input and output]

  // schedule group 2
  int pos = reference2->nDims() - 2;
  // [..., tile1, tile2]
  reference2->merge(pos);
  reference2->split(pos, params.vectorize_factor2);
  reference2->split(pos, kThreadsPerBlock);
  // [..., Unroll, TIDx, Vectorize]

  // Propagate transformations of reference2 to the entire DAG except
  // group 1. We actually only want to propagate to the fusion outputs, but
  // inputs and outputs themselves are disconnected, so we have to borrow the
  // entire DAG and use its spanning tree.
  {
    auto all_tvs_except1 = ir_utils::allTvsExcept(
        fusion,
        {grouped_inputs_outputs[0].begin(), grouped_inputs_outputs[0].end()});
    SetSelector selector({all_tvs_except1.begin(), all_tvs_except1.end()});
    MaxRootDomainInfoSpanningTree entire_dag_except1(reference2, &selector);
    TransformPropagator propagator(reference2);
    entire_dag_except1.traverse(&propagator);
  }

  // parallelize group2 and its cached inputs
  {
    reference2->axis(-1)->parallelize(ParallelType::Vectorize);
    reference2->axis(-2)->parallelize(ParallelType::TIDx);
    reference2->axis(-3)->parallelize(ParallelType::Unroll);

    ComputeAtMap ca_map(fusion);

    scheduler_utils::parallelizeAllLike(
        reference2,
        {group2_and_cached_inputs.begin(), group2_and_cached_inputs.end()},
        {ParallelType::Vectorize, ParallelType::TIDx});

    // Only unrolled the axes that exactly maps to the unrolled axes
    //  on reference as support for permissively mapped axes are not
    //  yet clearly defined.
    std::vector<TensorView*> unrolled_group2_cached_inputs;
    for (auto gin : group2_and_cached_inputs) {
      if (std::any_of(
              gin->domain()->domain().begin(),
              gin->domain()->domain().end(),
              [&ca_map, reference2](IterDomain* id) {
                return ca_map.areMapped(
                    id, reference2->axis(-3), IdMappingMode::EXACT);
              })) {
        unrolled_group2_cached_inputs.push_back(gin);
      }
    }

    scheduler_utils::parallelizeAllLike(
        reference2, unrolled_group2_cached_inputs, {ParallelType::Unroll});
  }

  // schedule group 1
  reference1->reorder({{-2, -1}});
  // [..., tile2, tile1]
  pos = reference1->nDims() - 2;
  reference1->merge(pos);
  reference1->split(pos, params.vectorize_factor1);
  reference1->split(pos, kThreadsPerBlock);
  reference1->axis(-1)->parallelize(ParallelType::Vectorize);
  reference1->axis(-2)->parallelize(ParallelType::TIDx);
  reference1->axis(-3)->parallelize(ParallelType::Unroll);
  // [..., Unroll, TIDx, Vectorize]

  // Propagate transformations, parallelization of the reference1 to the entire
  // DAG except group 2 and its corresponding cached outputs.
  {
    auto all_tvs_except2 =
        ir_utils::allTvsExcept(fusion, group2_and_cached_inputs);
    SetSelector selector({all_tvs_except2.begin(), all_tvs_except2.end()});
    MaxRootDomainInfoSpanningTree entire_dag_except_outputs(
        reference1, &selector);
    TransformPropagator propagator(reference1);
    entire_dag_except_outputs.traverse(&propagator);
    scheduler_utils::parallelizeAllLike(
        reference1, all_tvs_except2, {ParallelType::TIDx});
  }

  // vectorize and unroll group 1's output and cached input
  {
    ComputeAtMap ca_map(fusion);
    std::vector<TensorView*> group1_and_cached_inputs(
        grouped_inputs_outputs[0].begin(), grouped_inputs_outputs[0].end());
    for (auto tv : grouped_inputs_outputs[0]) {
      if (tv->isFusionInput()) {
        group1_and_cached_inputs.emplace_back(ir_utils::consumerTvsOf(tv)[0]);
      }
    }
    scheduler_utils::parallelizeAllLike(
        reference1, group1_and_cached_inputs, {ParallelType::Vectorize});

    // Only unrolled the axes that exactly maps to the unrolled axes
    //  on reference as support for permissively mapped axes are not
    //  yet clearly defined.
    std::vector<TensorView*> unrolled_group1_cached_inputs;
    for (auto gin : group1_and_cached_inputs) {
      if (std::any_of(
              gin->domain()->domain().begin(),
              gin->domain()->domain().end(),
              [&ca_map, reference1](IterDomain* id) {
                return ca_map.areMapped(
                    id, reference1->axis(-3), IdMappingMode::EXACT);
              })) {
        unrolled_group1_cached_inputs.push_back(gin);
      }
    }

    scheduler_utils::parallelizeAllLike(
        reference1, unrolled_group1_cached_inputs, {ParallelType::Unroll});
  }

  // cleanup parallelization from reference1 and reference2 if they are fusion
  // inputs
  for (auto tv : {reference1, reference2}) {
    if (tv->isFusionInput()) {
      for (auto id : tv->domain()->domain()) {
        id->parallelize(ParallelType::Serial);
      }
    }
  }

  // Inline
  InlinePropagator inline_propagator(
      reference1, -1, ComputeAtMode::MostInlined);
  entire_dag.traverse(&inline_propagator);
}

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
