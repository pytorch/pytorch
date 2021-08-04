#include <torch/csrc/jit/codegen/cuda/scheduler/pointwise.h>

#include <torch/csrc/jit/codegen/cuda/executor_utils.h>
#include <torch/csrc/jit/codegen/cuda/instrumentation.h>
#include <torch/csrc/jit/codegen/cuda/ir_utils.h>
#include <torch/csrc/jit/codegen/cuda/lower_utils.h>
#include <torch/csrc/jit/codegen/cuda/scheduler/registry.h>
#include <torch/csrc/jit/codegen/cuda/scheduler/utils.h>
#include <torch/csrc/jit/codegen/cuda/transform_replay.h>
#include <torch/csrc/jit/codegen/cuda/utils.h>

#include <torch/csrc/jit/codegen/cuda/ir_iostream.h>

#include <ATen/cuda/CUDAContext.h>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

namespace {
// constexpr int64_t x_grid_limit = ((int64_t)1 << (int64_t)31) - (int64_t)1;
// Unused at the moment, commenting for clang tidy
constexpr int64_t kThreadX = 128;
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
  TensorView* largest_out = nullptr;
  int max_dims = -1;

  auto in_tvs = ir_utils::filterByType<TensorView>(fusion->inputs());
  auto out_tvs_it = ir_utils::filterByType<TensorView>(fusion->outputs());
  // Will want to access this with direct indexing later, convert now.
  std::vector<TensorView*> out_tvs(out_tvs_it.begin(), out_tvs_it.end());

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

  int64_t n_elems = 1;
  for (auto id : largest_out->getMaybeRFactorDomain()) {
    auto inferred_val =
        runtime_info.expressionEvaluator().evaluate(id->extent());
    TORCH_INTERNAL_ASSERT(
        inferred_val.has_value(),
        "Error inferring size for pointwise scheduler.");
    n_elems *= inferred_val.value();
  }

  // TODO: Set to 1?
  int64_t max_input_dtype_size = 2;
  size_t n_tensors = 0;

  for (auto inp : in_tvs) {
    max_input_dtype_size = std::max(
        max_input_dtype_size,
        (int64_t)dataTypeSize(inp->getDataType().value()));
    n_tensors++;
  }
  n_tensors += std::distance(out_tvs.begin(), out_tvs.end());

  const int64_t device_multiprocessor_count =
      (int64_t)at::cuda::getCurrentDeviceProperties()->multiProcessorCount;

  constexpr int64_t kSixteen = 16; // clang tidy

  auto max_unroll_factor = ceilDiv(
      // Available unrolling based on size of data type
      (int64_t)kSixteen / max_input_dtype_size,
      // Reduce unrolling if we have many inputs, start reduction at 4 inputs
      std::max(
          (scheduler_utils::lastPow2((int64_t)n_tensors) >> 2), (int64_t)1));

  // Don't unroll at the cost of getting a full wave on the GPU
  if (n_elems < device_multiprocessor_count * kThreadX &&
      max_unroll_factor > 1) {
    max_unroll_factor = std::min(
        max_unroll_factor,
        ceilDiv(n_elems, device_multiprocessor_count * kThreadX));
  }

  // If we use RNG don't unroll so we can do correctness testing
  if (fusion->isStochastic() && disableRNGUnrolling()) {
    max_unroll_factor = 1;
  }

  PointwiseParams params;
  params.tag = "Pointwise heuristics";

  // Don't try to vectorize if it's not recommended
  params.inner_factor = 1;

  // Vectorize as much as we can
  size_t vectorize_factor = max_unroll_factor;

  HeuristicCacheAccessor<std::vector<TensorView*>>
      vectorizable_inputs_outputs_data;

  // TODO: move all these boilerplate code into the accessor class
  // (follow up)
  if (data_cache && !data_cache->isRecording()) {
    vectorizable_inputs_outputs_data.writeTemporary(
        data_cache->getVectorizableInputsOutputs());
  } else {
    vectorizable_inputs_outputs_data.writeNew(
        scheduler_utils::getVectorizableInputsOutputs(largest_out));
    if (data_cache && data_cache->isRecording()) {
      data_cache->setVectorizableInputsOutputs(
          vectorizable_inputs_outputs_data.read());
    }
  }

  auto& vectorizable_inputs_outputs = vectorizable_inputs_outputs_data.read();

  for (auto tv : vectorizable_inputs_outputs) {
    const auto tv_vectorize_factor = runtime_info.getVectorizableWidth(tv);
    vectorize_factor = std::min(vectorize_factor, tv_vectorize_factor);
  }

  if (vectorize_factor == 1) {
    params.vectorize = false;
    params.inner_factor = max_unroll_factor;
  } else {
    params.vectorize = true;
    params.inner_factor = vectorize_factor;
  }

  return params;
}

bool schedulePointwise(
    Fusion* fusion,
    const at::ArrayRef<c10::IValue>& runtime_inputs) {
  FUSER_PERF_SCOPE("scheduleFusion");
  auto params = getPointwiseHeuristics(fusion, runtime_inputs);
  if (!params.has_value()) {
    return false;
  }
  schedulePointwise(fusion, params.value());
  return true;
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
      !fusion->hasReduction(), "This scheduler only handles pointwise ops.");

  // For intermediate outputs, apply cache_fork
  auto outs = fusion->outputs();
  for (const auto output : outs) {
    if (!output->uses().empty() && output->definition() != nullptr) {
      if (output->getValType().value() == ValType::TensorView) {
        output->as<TensorView>()->cache_fork();
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

  TensorView* reference_tv = nullptr;
  for (auto out : output_tvs) {
    if (out->definition() == nullptr) {
      continue;
    }
    if (nRootDims(out) == max_dims) {
      reference_tv = out;
      break;
    }
  }

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
  auto vectorizable_dims =
      scheduler_utils::FindAllMappedDims::from(reference_tv, inner_most_id);

  // Caches of inputs
  std::vector<TensorView*> cached_inputs;

  // Output, cache_before of output
  std::vector<std::pair<TensorView*, TensorView*>> cached_outputs;

  // Track what should be vectorized versus unrolled
  std::unordered_set<TensorView*> vectorized_tensor;

  // Figure out which inputs to cache for unrolling or vectorization
  for (auto inp : input_tvs) {
    if (inp->uses().empty()) {
      continue;
    }
    // Need to check before caching.
    bool vectorize = params.vectorize &&
        scheduler_utils::shouldVectorize(inp, vectorizable_dims);
    cached_inputs.emplace_back(inp->cache_after());
    if (vectorize) {
      vectorized_tensor.emplace(cached_inputs.back());
    }
  }

  // Figure out which outputs to cache for unrolling or vectorization
  for (auto out : output_tvs) {
    if (out->definition() == nullptr) {
      continue;
    }
    // Need to check before caching.
    bool vectorize = params.vectorize &&
        scheduler_utils::shouldVectorize(out, vectorizable_dims);
    cached_outputs.emplace_back(std::make_pair(out, out->cache_before()));
    if (vectorize) {
      vectorized_tensor.emplace(out);
    }
  }

  auto all_tvs = ir_utils::allTvs(fusion);

  scheduler_utils::mergeNonReduction(reference_tv);

  if (params.vectorize) {
    // Vectorize
    reference_tv->split(0, params.inner_factor);
    // Unswitch
    reference_tv->split(0, 1);
    // Threads
    reference_tv->split(0, kThreadX);

    reference_tv->axis(0)->parallelize(ParallelType::BIDx);
    reference_tv->axis(1)->parallelize(ParallelType::TIDx);
    reference_tv->axis(2)->parallelize(ParallelType::Unswitch);
    // Aggressively mark with vectorized and cleanup later. That way we don't
    // have to manually specify parallelization outside the reference.
    reference_tv->axis(-1)->parallelize(ParallelType::Vectorize);

    //[BIDx, TIDx, Unswitch, Vectorization]
    // To make consistent with unrolling:
    reference_tv->reorder({{1, 3}, {2, 1}, {3, 2}});
    //[BIDx, Unswitch, Vectorization, TIDx]
  } else {
    // Threads
    reference_tv->split(0, kThreadX);
    // Unroll
    reference_tv->split(0, params.inner_factor);
    // Unswitch
    reference_tv->split(0, 1);

    // [BIDx, Unswitch, Unroll, TIDx]
    reference_tv->axis(0)->parallelize(ParallelType::BIDx);
    reference_tv->axis(1)->parallelize(ParallelType::Unswitch);
    reference_tv->axis(3)->parallelize(ParallelType::TIDx);
  }

  TransformPropagator::from(reference_tv);
  scheduler_utils::parallelizeAllLike(reference_tv, all_tvs);

  if (params.vectorize) {
    // Clear vectorize on tensors that shouldn't have it
    for (auto tv : all_tvs) {
      if (!vectorized_tensor.count(tv)) {
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
