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

// Largest Power of 2 less-than n
constexpr int64_t lastPow2(int64_t n) {
  TORCH_INTERNAL_ASSERT(n >= 0);
  n |= (n >> 1);
  n |= (n >> 2);
  n |= (n >> 4);
  n |= (n >> 8); // NOLINT(cppcoreguidelines-avoid-magic-numbers)
  n |= (n >> 16); // NOLINT(cppcoreguidelines-avoid-magic-numbers)
  n |= (n >> 32); // NOLINT(cppcoreguidelines-avoid-magic-numbers)
  return std::max((int64_t)1, n - (n >> 1));
}
} // namespace

c10::optional<PointwiseParams> getPointwiseHeuristics(
    Fusion* fusion,
    const at::ArrayRef<c10::IValue>& runtime_inputs) {
  SchedulerRuntimeInfo runtime_info(fusion, runtime_inputs, true);
  return getPointwiseHeuristics(fusion, runtime_info);
}

namespace {
// Want to make sure this is consistent across heuristics and scheduling.
// Based on fusion information only. Does this TV have all dimensions of the
// fusion. Does it have an iter domain for its inner most dimension. For
// heuristics this information should be augmented by actual input information.
// i.e. true from this function is required but not sufficient
bool shouldVectorize(TensorView* tv, int64_t max_dims) {
  const auto& root_dom =
      TensorDomain::noReductions(tv->getMaybeRFactorDomain());

  // Don't vectorize 0-dim tensors
  if (root_dom.size() == 0) {
    return false;
  }

  // Don't vectorize tensors that don't have all dimensions in the fusion
  if (root_dom.size() != (size_t)max_dims) {
    return false;
  }

  // Don't vectorize if inner most dimension is a broadcast
  if (root_dom[root_dom.size() - 1]->isBroadcast()) {
    return false;
  }

  const auto& contiguity = tv->domain()->contiguity();
  // Don't vectorize if inner most dimension is not contiguous
  if (!contiguity[contiguity.size() - 1]) {
    return false;
  }

  return true;
}

} // namespace

c10::optional<PointwiseParams> getPointwiseHeuristics(
    Fusion* fusion,
    SchedulerRuntimeInfo& runtime_info) {
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
      std::max((lastPow2((int64_t)n_tensors) >> 2), (int64_t)1));

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

  for (auto tv_inp : ir_utils::filterByType<TensorView>(fusion->inputs())) {
    if (shouldVectorize(tv_inp, max_dims)) {
      const auto inp_vectorize_factor =
          runtime_info.getVectorizableWidth(tv_inp);
      vectorize_factor = std::min(vectorize_factor, inp_vectorize_factor);
    }
  }

  for (auto output_tv : out_tvs) {
    if (shouldVectorize(output_tv, max_dims)) {
      const auto out_vectorize_factor =
          runtime_info.getVectorizableWidth(output_tv);
      vectorize_factor = std::min(vectorize_factor, out_vectorize_factor);
    }
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
  for (auto tv : scheduler_utils::allTvs(fusion)) {
    if (tv->isFusionInput() || tv->isFusionOutput()) {
      tv->setMemoryType(MemoryType::Global);
    } else {
      tv->setMemoryType(MemoryType::Local);
    }
  }

  // maybe has_reduction for scheduling should be done on a per output tensor
  // basis.
  TORCH_INTERNAL_ASSERT(
      !fusion->hasReduction(), "This scheduler only handles pointwise ops.");

  // For intermediate outputs, apply cache_fork
  auto outs = fusion->outputs();
  for (const auto output : outs) {
    if (!output->uses().empty()) {
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

  // Caches of inputs
  std::vector<TensorView*> cached_inputs;
  // Inputs that aren't cacched
  std::vector<TensorView*> not_cached_inputs;

  // Output, cache_before of output
  std::vector<std::pair<TensorView*, TensorView*>> cached_outputs;
  // Outputs that aren't cached
  std::vector<TensorView*> not_cached_outputs;

  // Figure out which inputs to cache for unrolling or vectorization
  for (auto inp : input_tvs) {
    // If zero dim tensor, don't process it
    if (std::any_of(
            inp->getMaybeRFactorDomain().begin(),
            inp->getMaybeRFactorDomain().end(),
            [](IterDomain* iter_domain) {
              return iter_domain->extent()->isZeroInt();
            })) {
      continue;
    }

    bool cache_input = params.inner_factor > 1;
    cache_input = cache_input && nRootDims(inp) == max_dims;
    if (params.vectorize) {
      cache_input = cache_input && shouldVectorize(inp, max_dims);
    }

    if (cache_input) {
      cached_inputs.emplace_back(inp->cache_after());
    } else {
      not_cached_inputs.emplace_back(inp);
    }
  }

  // Figure out which outputs to cache for unrolling or vectorization
  for (auto out : output_tvs) {
    // If zero dim tensor, don't process it
    if (std::any_of(
            out->getRootDomain().begin(),
            out->getRootDomain().end(),
            [](IterDomain* iter_domain) {
              return iter_domain->extent()->isZeroInt();
            })) {
      continue;
    }

    bool cache_output = params.inner_factor > 1;
    cache_output = cache_output && nRootDims(out) == max_dims;

    if (params.vectorize) {
      cache_output = cache_output && shouldVectorize(out, max_dims);
    }

    if (cache_output) {
      cached_outputs.emplace_back(std::make_pair(out, out->cache_before()));
    } else {
      not_cached_outputs.emplace_back(out);
    }
  }

  TensorView* reference_tv = nullptr;
  for (auto out : output_tvs) {
    if (nRootDims(out) == max_dims) {
      reference_tv = out;
      break;
    }
  }

  TORCH_INTERNAL_ASSERT(
      reference_tv != nullptr,
      "Could not find a fully broadcasted output to reference schedule on.");

  auto all_tvs = scheduler_utils::allTvs(fusion);

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

  // Vectorize or unroll inputs
  for (auto cache_tv : cached_inputs) {
    if (params.vectorize && params.inner_factor > 1) {
      cache_tv->axis(2)->parallelize(ParallelType::Vectorize);
    } else if (params.inner_factor > 1) {
      cache_tv->axis(2)->parallelize(ParallelType::Unroll);
    }
  }

  // Vectorize or unroll outputs
  for (auto cache_tv : cached_outputs) {
    if (params.vectorize && params.inner_factor > 1) {
      cache_tv.first->axis(2)->parallelize(ParallelType::Vectorize);
    } else if (params.inner_factor > 1) {
      cache_tv.first->axis(2)->parallelize(ParallelType::Unroll);
    }
  }

  // Start at outputs and work our way back
  //[BIDx, Unswitch, Vectorization, TIDx]
  for (auto entry : cached_outputs) {
    entry.second->computeWith(entry.first, 2, ComputeAtMode::BestEffort);
  }

  std::vector<TensorView*> consumers_of_cached_inputs;
  // Cache of input, and one of its consumers
  std::vector<std::pair<TensorView*, TensorView*>> input_cache_and_consumer;
  {
    std::unordered_set<TensorView*> added;
    for (auto cached_input : cached_inputs) {
      auto consumer_tvs = scheduler_utils::consumerTvsOf(cached_input);
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

  // Producers for inlined computeAt
  std::vector<TensorView*> compute_from = not_cached_inputs;
  compute_from.insert(
      compute_from.end(),
      consumers_of_cached_inputs.begin(),
      consumers_of_cached_inputs.end());

  // Consumers for inlined computeAt
  std::vector<TensorView*> compute_to = not_cached_outputs;
  for (auto entry : cached_outputs) {
    compute_to.emplace_back(entry.second);
  }

  // [BIDx, Unswitch, Unroll, TIDx]
  // Can't use negative numbers for specification of axes because trivial
  // reductions can get pushed inner most, see:
  // TestCudaFuser.test_trivial_reduction
  // Inline inside computations
  scheduler_utils::computeAtBetween(
      compute_from, compute_to, -1, ComputeAtMode::MostInlined);

  for (auto entry : input_cache_and_consumer) {
    entry.first->computeAt(entry.second, 2, ComputeAtMode::BestEffort);
  }

  // Re parallelize just for an abundance of safety.
  // TODO: Look through computeAt to make sure we maintain parallel type
  // properly
  for (auto id : reference_tv->domain()->domain()) {
    if (id->getParallelType() == ParallelType::Vectorize) {
      id->parallelize(ParallelType::Serial);
    }
  }
  // Make sure parallelization is all still correct after computeAt
  scheduler_utils::parallelizeAllLike(reference_tv, all_tvs);

  // Vectorize or unroll inputs
  for (auto cache_tv : cached_inputs) {
    if (params.vectorize && params.inner_factor > 1) {
      cache_tv->axis(2)->parallelize(ParallelType::Vectorize);
    } else if (params.inner_factor > 1) {
      cache_tv->axis(2)->parallelize(ParallelType::Unroll);
    }
  }

  // Vectorize or unroll outputs
  for (auto cache_tv : cached_outputs) {
    if (params.vectorize && params.inner_factor > 1) {
      cache_tv.first->axis(2)->parallelize(ParallelType::Vectorize);
    } else if (params.inner_factor > 1) {
      cache_tv.first->axis(2)->parallelize(ParallelType::Unroll);
    }
  }
}

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
