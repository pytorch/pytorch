#pragma once

#include <torch/csrc/jit/codegen/cuda/executor.h>
#include <torch/csrc/jit/codegen/cuda/fusion.h>
#include <torch/csrc/jit/codegen/cuda/ir_all_nodes.h>
#include <torch/csrc/jit/codegen/cuda/ir_utils.h>
#include <torch/csrc/jit/codegen/cuda/kernel_cache.h>
#include <torch/csrc/jit/codegen/cuda/lower2device.h>
#include <torch/csrc/jit/codegen/cuda/ops/all_ops.h>
#include <torch/csrc/jit/codegen/cuda/scheduler/all_schedulers.h>

#include <benchmark/benchmark.h>

#include <ATen/cuda/CUDAContext.h>
#include <torch/torch.h>

#include <cuda_runtime.h>

using namespace torch::jit::fuser::cuda;

// Make a tensor that is known to be non-contiguous of dimensionality=ndims,
// but unknown sizes
TensorView* makeSymbolicTensor(size_t ndims, DataType dtype = DataType::Float);

// Make a tensor that is known to be fully contiguous of dimensionality=ndims,
// but unknown sizes. Taken from test_gpu.cpp
TensorView* makeContigTensor(size_t ndims, DataType dtype = DataType::Float);

// Make a non-contiguous tensor of compile-time known sizes
TensorView* makeConcreteTensor(
    std::vector<int64_t> shape,
    DataType dtype = DataType::Float);

// Make a contiguous tensor of compile-time known sizes
TensorView* makeContigConcreteTensor(
    std::vector<int64_t> shape,
    DataType dtype = DataType::Float);

std::string toString(const ReductionParams& rparams);
std::string toString(const PointwiseParams& params);
std::string toString(const TransposeParams& params);
std::string toString(const std::shared_ptr<HeuristicParams>& params);
std::string toString(LaunchParams lparams);

// Run benchmark iterations with provided inputs. If not segmented, report
// kernel time from the runtime, as well as heuristic parameters. If segmented
// use timers. Make sure to clear L2 between iterations.
void runBenchmarkIterations(
    benchmark::State& benchmark_state,
    FusionExecutorCache* fusion_executor_cache,
    std::vector<c10::IValue>& aten_inputs);

void clearL2Cache();

class CudaKernelTimer {
 public:
  CudaKernelTimer() {
    // Setup
    C10_CUDA_CHECK(cudaEventCreate(&start_event));
    C10_CUDA_CHECK(cudaEventCreate(&finish_event));
    C10_CUDA_CHECK(cudaEventRecord(start_event));
  }

  ~CudaKernelTimer() {
    C10_CUDA_IGNORE_ERROR(cudaEventDestroy(start_event));
    C10_CUDA_IGNORE_ERROR(cudaEventDestroy(finish_event));
  }

  void restart() {
    C10_CUDA_CHECK(cudaEventRecord(start_event));
  }

  float elapsed() {
    // Record
    C10_CUDA_CHECK(cudaEventRecord(finish_event));
    C10_CUDA_CHECK(cudaEventSynchronize(start_event));
    C10_CUDA_CHECK(cudaEventSynchronize(finish_event));
    C10_CUDA_CHECK(
        cudaEventElapsedTime(&kernel_time_ms_, start_event, finish_event));
    return kernel_time_ms_;
  }

 private:
  // Create
  float kernel_time_ms_ = 0;
  cudaEvent_t start_event = {};
  cudaEvent_t finish_event = {};
};

namespace executorCache {
using ExecutorPtr = std::unique_ptr<FusionExecutorCache>;
using ExecutorMap = std::unordered_map<std::string, ExecutorPtr>;
ExecutorMap& getGlobalMap();
} // namespace executorCache

//! Utility to manage FusionExecutorCache instances for
//!  all defined benchmarks
class BenchmarkGraph : public benchmark::Fixture {
 public:
  using SetupFusionFunction = std::function<void(Fusion*)>;
  using SetupFusionMap = std::unordered_map<std::string, SetupFusionFunction>;

  virtual std::string graphName() = 0;
  virtual SetupFusionFunction setupFusion() = 0;

  FusionExecutorCache* getExecutorCache() {
    auto& executor_ = getExecutorCacheMap()[graphName()];
    TORCH_INTERNAL_ASSERT(executor_);
    return executor_.get();
  }

  void SetUp(const ::benchmark::State& state) {
    auto& executor_ = getExecutorCacheMap()[graphName()];
    // Makes sure same graph hasn't been compiled before
    if (!executor_) {
      auto fusion_ptr = std::make_unique<Fusion>();
      FusionGuard(fusion_ptr.get());
      setupFusion()(fusion_ptr.get());
      getExecutorCacheMap()[graphName()] =
          std::make_unique<FusionExecutorCache>(std::move(fusion_ptr));
    }
  }

  void TearDown(const ::benchmark::State& state) {}

 protected:
  static executorCache::ExecutorMap& getExecutorCacheMap() {
    return executorCache::getGlobalMap();
  }
};

#define NVFUSER_TO_STRING_HELPER(n) std::string(#n)
#define NVFUSER_TO_STRING(n) NVFUSER_TO_STRING_HELPER(n)

//! NVFUSER_BENCHMARK_RUN utility usage:
//!  This utility helps create and manage FusionExecutorCaches and tries to use
//!  the caching
//! mechanism in NVFuser to avoid re-compilation.
//!
//!  There are two macros in this utility: NVFUSER_BENCHMARK_DEFINE, and
//!  NVFUSER_BENCHMARK_RUN,
//! and user needs to supply two functions SETUP_FUSION and RUN_FUSION, with
//! following signatures:
//!
//!  SETUP_FUSION(Fusion* , args...);
//!  RUN_FUSION(benchmark::State&, FusionExecutorCache* , args...);
//!
//!  where args... are additional arguments, and they need to be the same for
//!  SETUP_FUSION and RUN_FUSION.
//!
//!  SETUP_FUSION is called once in each definition of benchmark to build the
//!  fusionIR graph
//!
//!  RUN_FUSION is just like the normal benchmark instance, except that a
//!  FusionExecutorCache
//!   will be provided for scheduling, running and timing the fusion runs. It is
//!   called once in each benchmark instance. For example:
//!   NVFUSER_BENCHMARK_RUN(my_benchmark)
//!    ->RangeMultiplier(2)
//!    ->Ranges({{1, 4})
//!  Calls RUN_FUSION 3 times.
//!
//!  To register a benchmark, the API is:
//!
//!  NVFUSER_BENCHMARK_DEFINE(my_benchmark,SETUP_FUSION,RUN_FUSION,args...);
//!
//!    where my_benchmark is any unique name given for this benchmark,
//!      SETUP_FUSION, RUN_FUSION as described above,
//!      args... is the arg list supplied to both setup_fusion and run_fusion
//!
//!  each NVFUSER_BENCHMARK_DEFINE registers a benchmark with a single
//!  FusionExecutorCache, i.e. a single fusion graph, and multiple benchmark
//!  data points can be registered like:
//!
//!  NVFUSER_BENCHMARK_RUN(my_benchmark)
//!    ->Ranges({{1,2}});
//!
//!  NVFUSER_BENCHMARK_RUN(my_benchmark)
//!    ->Ranges({{3,4}});
//!
//!  All datapoints will use the same FusionExecutorCache so recompilation is
//!  avoided as much as possible.

#define NVFUSER_BENCHMARK_DEFINE(                                       \
    BENCHMARK_NAME, SETUP_FUSION, RUN_FUSION, ...)                      \
  class BENCHMARK_NAME##___GRAPH : public BenchmarkGraph {              \
   public:                                                              \
    std::string graphName() {                                           \
      return NVFUSER_TO_STRING(BENCHMARK_NAME##___GRAPH);               \
    }                                                                   \
    SetupFusionFunction setupFusion() {                                 \
      return [](Fusion* fusion) { SETUP_FUSION(fusion, __VA_ARGS__); }; \
    }                                                                   \
  };                                                                    \
  BENCHMARK_DEFINE_F(BENCHMARK_NAME##___GRAPH, BENCHMARK_NAME)          \
  (benchmark::State & benchmark_state) {                                \
    RUN_FUSION(                                                         \
        benchmark_state,                                                \
        BENCHMARK_NAME##___GRAPH::getExecutorCache(),                   \
        __VA_ARGS__);                                                   \
  }

#define NVFUSER_BENCHMARK_RUN(BENCHMARK_NAME) \
  BENCHMARK_REGISTER_F(BENCHMARK_NAME##___GRAPH, BENCHMARK_NAME)
