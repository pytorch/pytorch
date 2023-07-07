#define TORCH_ASSERT_ONLY_METHOD_OPERATORS

#include <ATen/cuda/CUDAConfig.h>  // for the definition of AT_CUDNN_ENABLED

#if AT_CUDNN_ENABLED()

#include <ATen/native/cudnn/Macros.h>

#if HAS_CUDNN_V8()

#include <ATen/cudnn/cudnn-wrapper.h>

#include <c10/macros/Macros.h>

C10_DIAGNOSTIC_PUSH_AND_IGNORED_IF_DEFINED("-Wsuggest-override")
#include <cudnn_frontend.h>
C10_DIAGNOSTIC_POP()

#include <cudnn_frontend_find_plan.h>
#include <cudnn_frontend_get_plan.h>
#include <ATen/core/Tensor.h>
#include <ATen/TensorUtils.h>
#include <ATen/cuda/Exceptions.h>
#include <ATen/native/ConvUtils.h>
#include <ATen/native/cudnn/ConvShared.h>
#include <ATen/native/utils/ParamsHash.h>
#include <ATen/cudnn/Handle.h>
#include <ATen/TensorUtils.h>

#include <c10/util/env.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDACachingAllocator.h>

#include <unordered_map>
#include <list>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#else
#include <ATen/ops/empty.h>
#endif

#ifdef __linux__
#include <dlfcn.h>
#endif

namespace at { namespace native {

namespace {

// TODO: remove duplicate code in Conv_v7.cpp
constexpr int64_t operator "" _TiB(unsigned long long n) {
  return size_t(n) << 40;
}

uint8_t getAlignment(const Tensor &t) {
  // alignment are in bytes
  uint8_t alignment = 1;
  uintptr_t address = reinterpret_cast<uintptr_t>(t.data_ptr());
  for (; alignment < 32; alignment *= 2) {
    if (address % (alignment * 2)) {
      return alignment;
    }
  }
  return alignment;
}

cudnn_frontend::Tensor getTensorDescriptorWithTypeVirtual(const Tensor &t, const int64_t id, const uint8_t alignment, const cudnnDataType_t dataType, const at::MemoryFormat memory_format, const bool _virtual) {
#if defined(__linux__) && !defined(FBCODE_CAFFE2) && CUDNN_MAJOR == 8 && CUDNN_MINOR > 5
  // Workaround for cudnn error handling deficiency, that results in a crash on Ubuntu-22+
  // if `libnvrtc.so` is not found on the system, which strictly speaking is not necessary
  // for usecases below
  // See https://github.com/pytorch/pytorch/issues/97041
  static C10_UNUSED auto cudnn_cnn_infer_handler = [] {
    void *handle = dlopen("libcudnn_cnn_infer.so.8", RTLD_LAZY);
    char *err = dlerror();
    if (!handle) {
      TORCH_WARN("Attempt to open cnn_infer failed: handle=", handle, " error: ", err);
    } else if (err) {
      TORCH_WARN("Applied workaround for CuDNN issue, install nvrtc.so");
    }
    return handle;
  }();
#endif
  auto sizes = t.sizes();
  auto strides = t.strides();
  bool channels_last = memory_format == at::MemoryFormat::ChannelsLast ||
    memory_format == at::MemoryFormat::ChannelsLast3d;
  fixSizeOneDimStride<int64_t>(sizes.size(), &sizes[0], (int64_t *) &strides[0], channels_last);
  auto r = cudnn_frontend::TensorBuilder()
    .setDim(sizes.size(), sizes.data())
    .setStrides(strides.size(), strides.data())
    .setId(id)
    .setAlignment(alignment)
    .setDataType(dataType)
    .setVirtual(_virtual)
    .build();
  return r;
}

cudnn_frontend::Tensor getTensorDescriptor(const Tensor &t, const int64_t id, const uint8_t alignment, const at::MemoryFormat memory_format) {
  return getTensorDescriptorWithTypeVirtual(t, id, alignment, getCudnnDataType(t), memory_format, false);
}

cudnn_frontend::ConvDesc_v8 getConvDescriptor(cudnnDataType_t dataType, IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, const at::ScalarType scalar_type) {
  uint64_t convDim = stride.size();
  if (scalar_type == kBFloat16 || scalar_type == kHalf) {
    dataType = CUDNN_DATA_FLOAT;
  }
  return cudnn_frontend::ConvDescBuilder()
    .setDataType(dataType)
    .setMathMode(CUDNN_CROSS_CORRELATION)
    .setNDims(convDim)
    .setStrides(convDim, stride.data())
    .setPrePadding(convDim, padding.data())
    .setPostPadding(convDim, padding.data())
    .setDilation(convDim, dilation.data())
    .build();
}

void filterEngineConfigs(
  cudnn_frontend::EngineConfigList &from,
  cudnn_frontend::EngineConfigList &to,
  bool deterministic, bool allow_tf32, c10::ScalarType scalar_type)
{
  auto filter = [=](cudnnBackendDescriptor_t c) {
    if (deterministic) {
      if (cudnn_frontend::hasNumericalNote<CUDNN_NUMERICAL_NOTE_NONDETERMINISTIC>(c)) {return true;}
    }
    if (cudnn_frontend::hasNumericalNote<CUDNN_NUMERICAL_NOTE_DOWN_CONVERT_INPUTS>(c)) {return true;}
    if (scalar_type == kFloat) {
      // TODO: check under which conditions this is OK
      if (!allow_tf32 && cudnn_frontend::hasNumericalNote<CUDNN_NUMERICAL_NOTE_TENSOR_CORE>(c)) {return true;}
    }
    return false;
  };
  cudnn_frontend::filter(from, to, filter);
}

struct CacheKey {
  ConvolutionParams params;
  cudnnBackendDescriptorType_t operation;
  uint8_t x_alignment;
  uint8_t w_alignment;
  uint8_t y_alignment;
};

struct CacheKeyFused {
  ConvolutionParams params;
  // No op here because it is assumed to be a forward conv op
  uint8_t x_alignment;
  uint8_t w_alignment;
  uint8_t y_alignment;
  uint8_t z_alignment;
  uint8_t b_alignment;
  // TODO: does it make sense to have this in the key? but alpha is a graph-level param...
  float alpha;
};

struct CacheKeyWrapper : ParamsWrapper<CacheKey> {
  CacheKeyWrapper(const cudnnBackendDescriptorType_t operation, const Tensor& y, const Tensor& x, const Tensor& w, const IntArrayRef padding, const IntArrayRef stride, const IntArrayRef dilation, int64_t groups, bool deterministic, bool allow_tf32) {
    at::MemoryFormat memory_format = cudnn_conv_suggest_memory_format(x, w);
    setConvolutionParams(&(this->pod.params), x, w, padding, stride, dilation, groups, deterministic, allow_tf32, memory_format);
    this->pod.operation = operation;
    this->pod.x_alignment = getAlignment(x);
    this->pod.y_alignment = getAlignment(y);
    this->pod.w_alignment = getAlignment(w);
  }
};

struct CacheKeyFusedWrapper : ParamsWrapper<CacheKeyFused> {
  CacheKeyFusedWrapper(const Tensor& y, const Tensor& x, const Tensor& w, const Tensor& z, const Tensor& b, const float alpha, const IntArrayRef padding, const IntArrayRef stride, const IntArrayRef dilation, int64_t groups, bool deterministic, bool allow_tf32) {
    at::MemoryFormat memory_format = cudnn_conv_suggest_memory_format(x, w);
    setConvolutionParams(&(this->pod).params, x, w, padding, stride, dilation, groups, deterministic, allow_tf32, memory_format);
    this->pod.x_alignment = getAlignment(x);
    this->pod.y_alignment = getAlignment(y);
    this->pod.w_alignment = getAlignment(w);
    this->pod.z_alignment = getAlignment(z);
    this->pod.b_alignment = getAlignment(b);
    this->pod.alpha = alpha;
  }
};

static int _parseChosenLRUCacheLimit() {
  const char * val = getenv("TORCH_CUDNN_V8_API_LRU_CACHE_LIMIT");
  int limit = 10000; // roughly corresponds to 2GiB assuming 200KiB per ExecutionPlan
  if (val) {
    try {
      limit = std::stoi(val);
    } catch(std::invalid_argument const& e) {
      TORCH_WARN("invalid TORCH_CUDNN_V8_API_LRU_CACHE_LIMIT,",
                 " using default LRU cache limit of ", limit, " entries.");
    } catch(std::out_of_range const& e) {
      TORCH_WARN("invalid TORCH_CUDNN_V8_API_LRU_CACHE_LIMIT,",
                 " using default LRU ache limit of ", limit, " entries.");
    }
  }
  return limit;
}

static int getLRUCacheLimit() {
  static int limit = _parseChosenLRUCacheLimit();
  return limit;
}

template <typename T, typename KeyType>
struct BenchmarkCache {
std::list<KeyType> engine_cache_order;
std::unordered_map<KeyType, std::pair<cudnn_frontend::ExecutionPlan, typename std::list<KeyType>::iterator >, ParamsWrapperHash<KeyType>> engine_cache;

// no mutexes here as caches are now thread local for v8, can also return a pointer
// to the Execution Plan if we know it will not be invalidated by another thread
cudnn_frontend::ExecutionPlan* find(const KeyType& key) {
  int lru_cache_limit = getLRUCacheLimit();
  if (lru_cache_limit < 0) {
    return nullptr;
  }
  auto it = engine_cache.find(key);
  if (it == engine_cache.end()) {
    return nullptr;
  }
  if (lru_cache_limit) {
    TORCH_INTERNAL_ASSERT(*(it->second.second) == key, "CUDNN V8 LRU Cache Corrupted (found key mismatches list). Please report a bug to PyTorch.");
    auto engine_cache_order_size = engine_cache_order.size();
    auto engine_cache_size = engine_cache.size();
    TORCH_INTERNAL_ASSERT(engine_cache_order_size == engine_cache_size, "CUDNN V8 LRU Cache Corrupted (found list vs. map size mismatch). Please report a bug to PyTorch.");
    // update most recently accessed
    auto plan = it->second.first;
    engine_cache_order.erase(it->second.second);
    engine_cache_order.push_back(key);
    engine_cache.erase(key);
    engine_cache.emplace(key, std::make_pair(plan, --engine_cache_order.end()));
    it = engine_cache.find(key);
    TORCH_INTERNAL_ASSERT(it->first == *(it->second.second), "CUDNN V8 LRU Cache Corrupted (refresh list vs. map key mismatch). Please report a bug to PyTorch.");
    TORCH_INTERNAL_ASSERT((long) engine_cache_order.size() <= lru_cache_limit, "CUDNN V8 LRU Cache Corrupted (refresh size exceeds limit: ", lru_cache_limit, " please report a bug to PyTorch.");
    TORCH_INTERNAL_ASSERT(engine_cache_order.size() == engine_cache_order_size, "CUDNN V8 LRU Cache Corrupted (list size unexpectedly changed). Please report a bug to PyTorch.");
    TORCH_INTERNAL_ASSERT(engine_cache.size() == engine_cache.size(), "CUDNN V8 LRU Cache Corrupted (cache size unexpectedly changed). Please report a bug to PyTorch.");
  }
  return &(it->second.first);
}

void update(const KeyType& key, T& results) {
  int lru_cache_limit = getLRUCacheLimit();
  if (lru_cache_limit < 0) {
    return;
  } else if (lru_cache_limit) {
    auto it = engine_cache.find(key);
    if (it == engine_cache.end()) {
      auto engine_cache_order_size = engine_cache_order.size();
      auto engine_cache_size = engine_cache.size();
      TORCH_INTERNAL_ASSERT(engine_cache_order_size == engine_cache_size, "CUDNN V8 LRU Cache Corrupted (list vs. map size mismatch). Please report a bug to PyTorch.");
      if ((long) engine_cache_order_size >= lru_cache_limit) {
        // need to perform eviction
        TORCH_INTERNAL_ASSERT(engine_cache.find(engine_cache_order.front()) != engine_cache.end(), "CUDNN V8 LRU Cache Corrupted (eviction key not in map). Please report a bug to PyTorch.");
        engine_cache.erase(engine_cache_order.front());
        engine_cache_order.pop_front();
      }
    } else {
      TORCH_INTERNAL_ASSERT(*(it->second.second) == key, "CUDNN V8 LRU Cache Corrupted (list iterator key mismatch). Please report a bug to PyTorch.");
      engine_cache_order.erase(it->second.second);
    }
    engine_cache_order.push_back(key);
    engine_cache.erase(key);
    engine_cache.emplace(key, std::make_pair(results, --engine_cache_order.end()));
    TORCH_INTERNAL_ASSERT(engine_cache.find(key)->first == *(engine_cache.find(key)->second.second), "CUDNN V8 LRU Cache Corrupted (updated list vs. map key mismatch). Please report a bug to PyTorch.");
    TORCH_INTERNAL_ASSERT((long) engine_cache_order.size() <= lru_cache_limit, "CUDNN V8 LRU Cache Corrupted (updated size exceeds limit: ", lru_cache_limit, " please report a bug to PyTorch.");
  } else {
    engine_cache.erase(key);
    engine_cache.emplace(key, std::make_pair(results, engine_cache_order.end())); // dummy iterator
  }
}

};

// @eqy: use thread local caches as cuDNN Execution Plans are not guaranteed to be thread safe across all engines
// see Limitations in https://docs.nvidia.com/deeplearning/cudnn/release-notes/index.html
thread_local BenchmarkCache<cudnn_frontend::ExecutionPlan, CacheKeyWrapper> benchmark_cache;
thread_local BenchmarkCache<cudnn_frontend::ExecutionPlan, CacheKeyFusedWrapper> benchmark_cache_fused;

} // namespace

void run_conv_plan(cudnnHandle_t handle, const Tensor& x, const Tensor& y, const Tensor& w, const cudnn_frontend::ExecutionPlan& plan) {
  c10::DeviceGuard g(x.options().device());
  auto workspace_size = plan.getWorkspaceSize();
  auto workspace_ptr = c10::cuda::CUDACachingAllocator::get()->allocate(workspace_size);
  void *data_ptrs[] = {x.data_ptr(), y.data_ptr(), w.data_ptr()};
  int64_t uids[] = {'x', 'y', 'w'};
  auto variantPack = cudnn_frontend::VariantPackBuilder()
      .setWorkspacePointer(workspace_size ? workspace_ptr.get() : nullptr)
      .setDataPointers(3, data_ptrs)
      .setUids(3, uids)
      .build();
  AT_CUDNN_CHECK(cudnnBackendExecute(handle, plan.get_raw_desc(), variantPack.get_raw_desc()));
}

void run_conv_plan_fused(cudnnHandle_t handle, const Tensor& x, const Tensor& y, const Tensor& w, const Tensor& z, const Tensor& b, const cudnn_frontend::ExecutionPlan& plan) {
  c10::DeviceGuard g(x.options().device());
  auto workspace_size = plan.getWorkspaceSize();
  auto workspace_ptr = c10::cuda::CUDACachingAllocator::get()->allocate(workspace_size);
  void *data_ptrs[] = {x.data_ptr(), y.data_ptr(), w.data_ptr(), z.data_ptr(), b.data_ptr()};
  int64_t uids[] = {'x', 'y', 'w', 'z', 'b'};
  auto variantPack = cudnn_frontend::VariantPackBuilder()
      .setWorkspacePointer(workspace_size ? workspace_ptr.get() : nullptr)
      .setDataPointers(5, data_ptrs)
      .setUids(5, uids)
      .build();
  AT_CUDNN_CHECK(cudnnBackendExecute(handle, plan.get_raw_desc(), variantPack.get_raw_desc()));
}

auto build_opgraph(const cudnnHandle_t handle, const cudnnBackendDescriptorType_t desc, const Tensor& x, const Tensor& y, const Tensor& w, const CacheKeyWrapper& key, const IntArrayRef padding, const IntArrayRef stride, const IntArrayRef dilation) {
  auto op = cudnn_frontend::OperationBuilder(desc)
      .setxDesc(getTensorDescriptor(x, 'x', key.pod.x_alignment, key.pod.params.memory_format))
      .setyDesc(getTensorDescriptor(y, 'y', key.pod.y_alignment, key.pod.params.memory_format))
      .setwDesc(getTensorDescriptor(w, 'w', key.pod.w_alignment, key.pod.params.memory_format))
      .setcDesc(getConvDescriptor(key.pod.params.dataType, padding, stride, dilation, x.scalar_type()))
      .build();
  std::array<cudnn_frontend::Operation const *, 1> ops = {&op};
  auto opGraph = cudnn_frontend::OperationGraphBuilder()
      .setHandle(handle)
      .setOperationGraph(ops.size(), ops.data())
      .build();
  return opGraph;
}

auto build_opgraph_fused(const cudnnHandle_t handle, const Tensor & x, const Tensor& y, const Tensor& w, const Tensor& z, const Tensor& b, const float alpha, const CacheKeyFusedWrapper& key, const IntArrayRef padding, const IntArrayRef stride, const IntArrayRef dilation) {
  // need computation to be done in FLOAT type regardless of reduced precision input
  const auto precision = CUDNN_DATA_FLOAT;
  auto addDesc = cudnn_frontend::PointWiseDescBuilder()
                           .setMode(CUDNN_POINTWISE_ADD)
                           .setMathPrecision(precision)
                           .build();
  auto addBiasDesc = cudnn_frontend::PointWiseDescBuilder()
                            .setMode(CUDNN_POINTWISE_ADD)
                            .setMathPrecision(precision)
                            .build();
  auto actDesc = cudnn_frontend::PointWiseDescBuilder()
                           .setMode(CUDNN_POINTWISE_RELU_FWD)
                           .setMathPrecision(precision)
                           .build();
  auto convDesc = getConvDescriptor(key.pod.params.dataType, padding, stride, dilation, x.scalar_type());
  const float alpha1 = 1.0;
  const float alpha2 = alpha;
  auto conv_op = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR)
                   .setxDesc(getTensorDescriptor(x, 'x', key.pod.x_alignment, key.pod.params.memory_format))
                   // virtual output of conv
                   .setyDesc(getTensorDescriptorWithTypeVirtual(y, 'C', key.pod.y_alignment, precision, key.pod.params.memory_format, true))
                   .setwDesc(getTensorDescriptor(w, 'w', key.pod.w_alignment, key.pod.params.memory_format))
                   .setAlpha(alpha1)
                   .setcDesc(convDesc)
                   .build();
  auto add_op = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
                           .setxDesc(conv_op.getOutputTensor())
                           .setbDesc(getTensorDescriptor(z, 'z', key.pod.z_alignment, key.pod.params.memory_format))
                           // another virtual output (of add)
                           .setyDesc(getTensorDescriptorWithTypeVirtual(y, 'A', key.pod.y_alignment, precision, key.pod.params.memory_format, true))
                           .setpwDesc(addDesc)
                           .setAlpha(alpha1)
                           .setAlpha2(alpha2)
                           .build();
  auto add_bias_op = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
                           .setxDesc(add_op.getOutputTensor())
                           .setbDesc(getTensorDescriptor(b, 'b', key.pod.b_alignment, key.pod.params.memory_format))
                           // another virtual output (of add bias)
                           .setyDesc(getTensorDescriptorWithTypeVirtual(y, 'B', key.pod.y_alignment, precision, key.pod.params.memory_format, true))
                           .setpwDesc(addBiasDesc)
                           .build();
  auto act_op = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
                          .setxDesc(add_bias_op.getOutputTensor())
                          // final output is in original datatype
                          .setyDesc(getTensorDescriptor(y, 'y', key.pod.y_alignment, key.pod.params.memory_format))
                          .setpwDesc(actDesc)
                          .build();
  std::array<cudnn_frontend::Operation const*, 4> ops = {&conv_op, &add_op, &add_bias_op, &act_op};
  auto opGraph = cudnn_frontend::OperationGraphBuilder()
                     .setHandle(handle)
                     .setOperationGraph(ops.size(), ops.data())
                     .build();
  return opGraph;
}

auto get_generator_sources(const cudnnBackendDescriptorType_t& desc, const Tensor& x, const bool deterministic, const bool allow_tf32, const cudnnBackendHeurMode_t heur_mode, const bool heuristic, const bool fallback) {
  // Method for engine config generator based on heuristics
  const auto heurgen_method = [/*&desc,*/ &x, deterministic, allow_tf32, heur_mode](cudnn_frontend::OperationGraph &opGraph) -> cudnn_frontend::EngineConfigList {
      auto heuristics = cudnn_frontend::EngineHeuristicsBuilder()
                            .setOperationGraph(opGraph)
                            .setHeurMode(heur_mode)
                            .build();
      auto &engine_configs = heuristics.getEngineConfig(heuristics.getEngineConfigCount());
      cudnn_frontend::EngineConfigList filtered_configs;
      filterEngineConfigs(engine_configs, filtered_configs, deterministic, allow_tf32, x.scalar_type());
      return filtered_configs;
  };
  // Method for engine config generator based on fallback list
  const auto fallback_method = [&desc, &x, deterministic, allow_tf32](cudnn_frontend::OperationGraph &opGraph) -> cudnn_frontend::EngineConfigList {
    auto fallback = cudnn_frontend::EngineFallbackListBuilder()
                        .setOperationGraph(opGraph)
                        .setOperation(desc)
                        .build();
    auto &fallback_list = fallback.getFallbackList();
    cudnn_frontend::EngineConfigList filtered_configs;
    filterEngineConfigs(fallback_list, filtered_configs, deterministic, allow_tf32, x.scalar_type());
    return filtered_configs;
  };
  if (heuristic && fallback) {
    std::vector<cudnn_frontend::GeneratorSource> sources = {heurgen_method, fallback_method};
    return sources;
  } else if (heuristic) {
    std::vector<cudnn_frontend::GeneratorSource> sources = {heurgen_method};
    return sources;
  } else {
    std::vector<cudnn_frontend::GeneratorSource> sources = {fallback_method};
    return sources;
  }
}

int64_t get_available_workspace() {
  int device;
  C10_CUDA_CHECK(c10::cuda::GetDevice(&device));
  size_t max_block_size = 0;
  c10::cuda::CUDACachingAllocator::cacheInfo(device, &max_block_size);
  return static_cast<int64_t>(max_block_size);
}

static nlohmann::json errata_json_handle;

bool plan_errata_exception(const cudnnHandle_t handle, const std::string & executionPlanTag) {
  static bool has_json = cudnn_frontend::load_from_config(errata_json_handle, "");
  if (!has_json) {
    return false;
  } else {
    return cudnn_frontend::check_errata(errata_json_handle, executionPlanTag, handle, [](){return true;});
  }
}

void generate_and_filter_plans(const cudnnHandle_t handle, cudnn_frontend::OperationGraph& opGraph, cudnn_frontend::EngineConfigGenerator& generator, const Tensor& x, cudnn_frontend::executionPlans_t& valid_plans, at::DataPtr& workspace_ptr) {
  auto initial_predicate_function = [&](cudnn_frontend::ExecutionPlan const& plan) -> bool {
    return plan_errata_exception(handle, plan.getTag());
  };
  auto plans = generator.cudnnGetPlan(handle, opGraph, initial_predicate_function);
  int64_t max_block_size = get_available_workspace();
  int64_t max_workspace_size = 0;
  std::for_each(plans.begin(), plans.end(), [&] (cudnn_frontend::ExecutionPlan& plan) {
    int64_t curr_workspace_size = plan.getWorkspaceSize();
    if (curr_workspace_size <= max_block_size) {
      if (curr_workspace_size > max_workspace_size) {
        max_workspace_size = plan.getWorkspaceSize();
      }
      valid_plans.emplace_back(std::move(plan));
    }
  });
  TORCH_CHECK_WITH(OutOfMemoryError, max_workspace_size < 1_TiB, "Not enough memory for workspace!");
  bool remove_invalid = false;
  while (max_workspace_size) {
    try {
      workspace_ptr = c10::cuda::CUDACachingAllocator::get()->allocate(max_workspace_size);
      break;
    } catch (c10::OutOfMemoryError &e) {
      max_workspace_size /= 2;
      (void)cudaGetLastError(); // clear CUDA error
      remove_invalid = true;
    }
  }
  if (remove_invalid) {
    cudnn_frontend::executionPlans_t new_valid_plans;
    for (auto &plan : valid_plans) {
      if (plan.getWorkspaceSize() <= max_workspace_size) {
        new_valid_plans.emplace_back(std::move(plan));
      }
    }
    valid_plans = std::move(new_valid_plans);
  }
}

auto get_plans_from_find(const cudnnHandle_t handle, const cudnnBackendDescriptorType_t desc, const Tensor& x, const Tensor& y, const Tensor& w, const CacheKeyWrapper& key, const IntArrayRef padding, const IntArrayRef stride, const IntArrayRef dilation, const bool deterministic, const bool allow_tf32) {
  auto opGraph = build_opgraph(handle, desc, x, y, w, key, padding, stride, dilation);
  void *data_ptrs[] = {x.data_ptr(), y.data_ptr(), w.data_ptr()};
  int64_t uids[] = {'x', 'y', 'w'};
  // We don't care about getting the best ordering of algos if we're roing to run all of them
  auto sources = get_generator_sources(desc, x, deterministic, allow_tf32, CUDNN_HEUR_MODE_INSTANT, true, true);
  cudnn_frontend::EngineConfigGenerator generator(sources.size(), sources.data());
  cudnn_frontend::executionPlans_t valid_plans;
  c10::DeviceGuard g(x.options().device());
  at::DataPtr workspace_ptr;
  generate_and_filter_plans(handle, opGraph, generator, x, valid_plans, workspace_ptr);
  auto variantPack = cudnn_frontend::VariantPackBuilder()
      .setDataPointers(3, data_ptrs)
      .setUids(3, uids)
      .setWorkspacePointer(workspace_ptr ? workspace_ptr.get() : nullptr)
      .build();

  auto benchmark_limit = at::globalContext().benchmarkLimitCuDNN();
  benchmark_limit = benchmark_limit ? benchmark_limit : 10000;
  auto plans = cudnn_frontend::time_sorted_plan<cudnn_frontend::CudnnFindSamplingTechnique::CUDNN_FIND_SAMPLE_ONCE>(handle, std::move(valid_plans), variantPack, benchmark_limit);

  cudnn_frontend::executionPlans_t sorted_plans;
  for (auto& plan : plans) {
    sorted_plans.emplace_back(std::move(plan));
  }
  return sorted_plans;
}

auto get_plans_from_find_fused(const cudnnHandle_t handle,
                               const Tensor& x, const Tensor& y, const Tensor& w, const Tensor& z, const Tensor& b,
                               const float alpha, const CacheKeyFusedWrapper& key,
                               const IntArrayRef padding, const IntArrayRef stride, const IntArrayRef dilation,
                               const bool deterministic, const bool allow_tf32) {
  auto opGraph = build_opgraph_fused(handle, x, y, w, z, b, alpha, key, padding, stride, dilation);
  void *data_ptrs[] = {x.data_ptr(), y.data_ptr(), w.data_ptr(), z.data_ptr(), b.data_ptr()};
  int64_t uids[] = {'x', 'y', 'w', 'z', 'b'};

  auto sources = get_generator_sources(CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR, x, deterministic, allow_tf32, CUDNN_HEUR_MODE_INSTANT, true, true);
  cudnn_frontend::EngineConfigGenerator generator(sources.size(), sources.data());
  cudnn_frontend::executionPlans_t valid_plans;
  c10::DeviceGuard g(x.options().device());
  at::DataPtr workspace_ptr;
  generate_and_filter_plans(handle, opGraph, generator, x, valid_plans, workspace_ptr);
  auto variantPack = cudnn_frontend::VariantPackBuilder()
      .setDataPointers(5, data_ptrs)
      .setUids(5, uids)
      .setWorkspacePointer(workspace_ptr ? workspace_ptr.get() : nullptr)
      .build();

  auto benchmark_limit = at::globalContext().benchmarkLimitCuDNN();
  benchmark_limit = benchmark_limit ? benchmark_limit : 10000;
  auto plans = cudnn_frontend::time_sorted_plan<cudnn_frontend::CudnnFindSamplingTechnique::CUDNN_FIND_SAMPLE_ONCE>(handle, std::move(valid_plans), variantPack, benchmark_limit);

  cudnn_frontend::executionPlans_t sorted_plans;
  for (auto& plan : plans) {
    sorted_plans.emplace_back(std::move(plan));
  }
  return sorted_plans;
}


// We only get configs from this stage to avoid building unnecessary plans that are never executed
auto get_configs_from_heuristics(const cudnnHandle_t handle, const cudnnBackendDescriptorType_t desc, std::string& opgraph_tag, const Tensor& x, const Tensor& y, const Tensor& w, const CacheKeyWrapper& key, const IntArrayRef padding, const IntArrayRef stride, const IntArrayRef dilation, const bool deterministic, const bool allow_tf32, const bool fallback) {
  auto opGraph = build_opgraph(handle, desc, x, y, w, key, padding, stride, dilation);
  opgraph_tag = opGraph.getTag();
  auto heuristic_mode = at::native::cudnnv8_use_heur_mode_b() ? CUDNN_HEUR_MODE_B : CUDNN_HEUR_MODE_INSTANT;
  auto sources = get_generator_sources(desc, x, deterministic, allow_tf32, heuristic_mode, !fallback, fallback);

  cudnn_frontend::EngineConfigGenerator generator(sources.size(), sources.data());
  auto configs = generator.generate_engine_config(opGraph);
  return configs;
}

auto get_configs_from_heuristics_fused(const cudnnHandle_t handle, std::string& opgraph_tag, const Tensor& x, const Tensor& y, const Tensor& w, const Tensor& z, const Tensor& b, const float alpha, const CacheKeyFusedWrapper& key, const IntArrayRef padding, const IntArrayRef stride, const IntArrayRef dilation, const bool deterministic, const bool allow_tf32, const bool fallback) {
  auto opGraph = build_opgraph_fused(handle, x, y, w, z, b, alpha, key, padding, stride, dilation);
  opgraph_tag = opGraph.getTag();
  auto heuristic_mode = at::native::cudnnv8_use_heur_mode_b() ? CUDNN_HEUR_MODE_B : CUDNN_HEUR_MODE_INSTANT;
  auto sources = get_generator_sources(CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR, x, deterministic, allow_tf32, heuristic_mode, !fallback, fallback);

  cudnn_frontend::EngineConfigGenerator generator(sources.size(), sources.data());
  auto configs = generator.generate_engine_config(opGraph);
  return configs;
}

void try_plans(cudnn_frontend::executionPlans_t& plans, const CacheKeyWrapper& key, const cudnnHandle_t handle, const Tensor& x, const Tensor& y, const Tensor& w) {
  for (auto & plan : plans) {
    try {
      run_conv_plan(handle, x, y, w, plan);
      benchmark_cache.update(key, plan);
      return;
    } catch (cudnn_frontend::cudnnException &e) {} catch (CuDNNError &e) {}
      catch (c10::OutOfMemoryError &e) {
        (void)cudaGetLastError(); // clear CUDA error
    }
  }
  TORCH_CHECK(false, "FIND was unable to find an engine to execute this computation");
}

void try_plans_fused(cudnn_frontend::executionPlans_t& plans, const CacheKeyFusedWrapper& key, const cudnnHandle_t handle, const Tensor& x, const Tensor& y, const Tensor& w, const Tensor& z, const Tensor& b) {
  for (auto & plan : plans) {
    try {
      run_conv_plan_fused(handle, x, y, w, z, b, plan);
      benchmark_cache_fused.update(key, plan);
      return;
    } catch (cudnn_frontend::cudnnException &e) {} catch (CuDNNError &e) {}
      catch (c10::OutOfMemoryError &e) {
        (void)cudaGetLastError(); // clear CUDA error
    }
  }
  TORCH_CHECK(false, "FIND was unable to find an engine to execute this computation");
}

bool try_configs(cudnn_frontend::EngineConfigList& configs, const std::string& opgraph_tag, const CacheKeyWrapper& key, const cudnnHandle_t handle, const Tensor& x, const Tensor& y, const Tensor& w) {
  for (auto & config : configs) {
    try {
      auto plan = cudnn_frontend::ExecutionPlanBuilder()
                    .setHandle(handle)
                    .setEngineConfig(config, opgraph_tag)
                    .build();
      if (plan_errata_exception(handle, plan.getTag())) {
        continue;
      }
      run_conv_plan(handle, x, y, w, plan);
      benchmark_cache.update(key, plan);
      return true;
    } catch (cudnn_frontend::cudnnException &e) {} catch(CuDNNError &e) {}
      catch (c10::OutOfMemoryError &e) {
        (void)cudaGetLastError(); // clear CUDA error
    }
  }
  return false;
}

bool try_configs_fused(cudnn_frontend::EngineConfigList& configs, const std::string& opgraph_tag, const CacheKeyFusedWrapper& key, const cudnnHandle_t handle, const Tensor& x, const Tensor& y, const Tensor& w, const Tensor& z, const Tensor& b) {
  for (auto & config : configs) {
    try {
      auto plan = cudnn_frontend::ExecutionPlanBuilder()
                    .setHandle(handle)
                    .setEngineConfig(config, opgraph_tag)
                    .build();
      if (plan_errata_exception(handle, plan.getTag())) {
        continue;
      }
      run_conv_plan_fused(handle, x, y, w, z, b, plan);
      benchmark_cache_fused.update(key, plan);
      return true;
    } catch (cudnn_frontend::cudnnException &e) {} catch(CuDNNError &e) {}
      catch (c10::OutOfMemoryError &e) {
        (void)cudaGetLastError(); // clear CUDA error
    }
  }
  return false;
}

void run_single_conv(const cudnnBackendDescriptorType_t operation,
  const Tensor& x, const Tensor& y, const Tensor& w,
  const IntArrayRef padding, const IntArrayRef stride, const IntArrayRef dilation, const int64_t groups,
  const bool benchmark, const bool deterministic, const bool allow_tf32) {
  cudnnHandle_t handle = getCudnnHandle();
  CacheKeyWrapper key(operation, y, x, w, padding, stride, dilation, groups, deterministic, allow_tf32);
  // TODO: is this thread safe if cache is updated? is pointer stale?
  auto search = benchmark_cache.find(key);
  if (search) {
    try {
      run_conv_plan(handle, x, y, w, *search);
      return;
    } catch(c10::OutOfMemoryError &e) {
      (void)cudaGetLastError(); // clear CUDA error
    }
  }
  if (!benchmark) {
    std::string opgraph_tag; // extra data needed for errata filter
    // heuristic configs
    cudnn_frontend::EngineConfigList configs = get_configs_from_heuristics(handle, operation,
                                                                           opgraph_tag,
                                                                           x, y, w, key,
                                                                           padding, stride, dilation,
                                                                           deterministic, allow_tf32, false);
    if (try_configs(configs, opgraph_tag, key, handle, x, y, w)) { return; }
    // fallback configs
    configs = get_configs_from_heuristics(handle, operation,
                                          opgraph_tag,
                                          x, y, w, key,
                                          padding, stride, dilation,
                                          deterministic, allow_tf32, true);
    if (try_configs(configs, opgraph_tag, key, handle, x, y, w)) { return; }
    TORCH_CHECK(false, "GET was unable to find an engine to execute this computation");
  } else {
    cudnn_frontend::executionPlans_t plans = get_plans_from_find(handle, operation,
                                                                 x, y, w, key,
                                                                 padding, stride, dilation,
                                                                 deterministic, allow_tf32);
    // Replicate v7 behavior: clear cached blocks as benchmark incurs
    // significant memory consumptiont that is not needed after this step
    if (at::native::_cudnn_get_conv_benchmark_empty_cache()) {
      c10::cuda::CUDACachingAllocator::emptyCache();
    }
    try_plans(plans, key, handle, x, y, w);
  }
}

void run_fused_conv(const Tensor& x, const Tensor& y, const Tensor& w, const Tensor& z, const Tensor& b,
  float alpha, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation,
  int64_t groups, const bool benchmark, const bool deterministic, const bool allow_tf32) {
  cudnnHandle_t handle = getCudnnHandle();

  CacheKeyFusedWrapper key(y, x, w, z, b, alpha, padding, stride, dilation, groups, deterministic, allow_tf32);
  auto search = benchmark_cache_fused.find(key);
  if (search) {
    try {
      run_conv_plan_fused(handle, x, y, w, z, b, *search);
      return;
    } catch(c10::OutOfMemoryError &e) {
      (void)cudaGetLastError(); // clear CUDA error
    }
  }
  if (!benchmark) {
    std::string opgraph_tag; // extra data needed for errata filter
    // heuristic configs
    cudnn_frontend::EngineConfigList configs = get_configs_from_heuristics_fused(handle,
                                                                                 opgraph_tag,
                                                                                 x, y, w, z, b, alpha, key,
                                                                                 padding, stride, dilation,
                                                                                 deterministic, allow_tf32, false);
    if (try_configs_fused(configs, opgraph_tag, key, handle, x, y, w, z, b)) { return; }
    // fallback configs
    configs = get_configs_from_heuristics_fused(handle,
                                                opgraph_tag,
                                                x, y, w, z, b, alpha, key,
                                                padding, stride, dilation,
                                                deterministic, allow_tf32, true);
    if (try_configs_fused(configs, opgraph_tag, key, handle, x, y, w, z, b)) { return; }
    TORCH_CHECK(false, "GET was unable to find an engine to execute this computation");
  } else {
    cudnn_frontend::executionPlans_t plans = get_plans_from_find_fused(handle,
                                                                       x, y, w, z, b, alpha, key,
                                                                       padding, stride, dilation,
                                                                       deterministic, allow_tf32);
    try_plans_fused(plans, key, handle, x, y, w, z, b);
  }
}

void raw_cudnn_convolution_forward_out(
    const Tensor& output, const Tensor& input, const Tensor& weight,
    const IntArrayRef padding, const IntArrayRef stride, const IntArrayRef dilation, const int64_t groups,
    const bool benchmark, const bool deterministic, const bool allow_tf32)
{
  if (output.numel() == 0) { return; }
  if (at::native::cudnnv8_enabled_check_debug()) {
    run_single_conv(CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR,
      input, output, weight, padding, stride, dilation, groups,
      benchmark, deterministic, allow_tf32);
  } else {
    raw_cudnn_convolution_forward_out_v7(
      output, input, weight,
      padding, stride, dilation, groups,
      benchmark, deterministic, allow_tf32);
  }
}

void raw_cudnn_convolution_backward_input_out(
    const at::Tensor& grad_input,
    const at::Tensor& grad_output,
    const at::Tensor& weight,
    const IntArrayRef padding, const IntArrayRef stride, const IntArrayRef dilation, const int64_t groups,
    const bool benchmark, const bool deterministic, const bool allow_tf32) {
  if (grad_input.numel() == 0) { return; }
  if (at::native::cudnnv8_enabled_check_debug()) {
    run_single_conv(CUDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_DATA_DESCRIPTOR,
      grad_input, grad_output, weight, padding, stride, dilation, groups,
      benchmark, deterministic, allow_tf32);
  } else {
    raw_cudnn_convolution_backward_input_out_v7(
      grad_input,
      grad_output,
      weight,
      padding, stride, dilation, groups,
      benchmark, deterministic, allow_tf32);
  }
}

void raw_cudnn_convolution_backward_weight_out(
    const Tensor& grad_weight, const Tensor& grad_output, const Tensor& input,
    const IntArrayRef padding, const IntArrayRef stride, const IntArrayRef dilation, const int64_t groups,
    const bool benchmark, const bool deterministic, const bool allow_tf32) {
  if (grad_weight.numel() == 0) { return; }
  if (at::native::cudnnv8_enabled_check_debug()) {
    run_single_conv(CUDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_FILTER_DESCRIPTOR,
      input, grad_output, grad_weight, padding, stride, dilation, groups,
      benchmark, deterministic, allow_tf32);
  } else {
    raw_cudnn_convolution_backward_weight_out_v7(
      grad_weight, grad_output, input,
      padding, stride, dilation, groups,
      benchmark, deterministic, allow_tf32);
  }
}

void raw_cudnn_convolution_add_relu_out(
    const Tensor& output,
    const Tensor& input,
    const Tensor& weight,
    const Tensor& z,
    float alpha,
    const Tensor& bias,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    int64_t groups,
    bool benchmark,
    bool deterministic,
    bool allow_tf32) {
  if (output.numel() == 0) { return; }
  if (at::native::cudnnv8_enabled_check_debug()) {
    auto bias_ = input.ndimension() == 4 ? bias.view({1, bias.numel(), 1, 1}) : bias.view({1, bias.numel(), 1, 1, 1});
    run_fused_conv(input, output, weight, z, bias_,
      alpha, stride, padding, dilation,
      groups, benchmark, deterministic, allow_tf32);
  } else {
    raw_cudnn_convolution_add_relu_out_v7(output, input, weight, z,
                                          alpha, bias, stride, padding, dilation,
                                          groups, benchmark, deterministic, allow_tf32);
  }
}

}} // at::native

#endif  // HAS_CUDNN_V8
#endif  // AT_CUDNN_ENABLED
