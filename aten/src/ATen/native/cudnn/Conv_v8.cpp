#include <ATen/cuda/CUDAConfig.h>  // for the definition of AT_CUDNN_ENABLED

#if AT_CUDNN_ENABLED()

#include <ATen/native/cudnn/Macros.h>

#if HAS_CUDNN_V8()

#include <ATen/cudnn/cudnn-wrapper.h>
#include <cudnn_frontend.h>
#include <cudnn_frontend_find_plan.h>
#include <cudnn_frontend_get_plan.h>
#include <ATen/ATen.h>
#include <ATen/TensorUtils.h>
#include <ATen/cuda/Exceptions.h>
#include <ATen/native/ConvUtils.h>
#include <ATen/native/cudnn/ConvShared.h>
#include <ATen/native/utils/ParamsHash.h>
#include <ATen/cudnn/Handle.h>
#include <ATen/TensorUtils.h>

#include <THC/THC.h>

#include <mutex>
#include <unordered_map>

namespace at { namespace native {

namespace {

// TODO: remove duplicate code in Conv_v7.cpp
constexpr size_t operator "" _TiB(unsigned long long n) {
  return size_t(n) * 1024 * 1024 * 1024 * 1024;
}

uint8_t getAlignment(const Tensor &t) {
  // alignment are in bytes
  uint8_t alignment = 1;
  uintptr_t address = reinterpret_cast<uintptr_t>(t.data_ptr());
  while (address % alignment == 0 && alignment < 64) alignment *= 2;
  return alignment;
}

cudnn_frontend::Tensor getTensorDescriptor(const Tensor &t, const int64_t id, const uint8_t alignment) {
  auto shape = t.sizes();
  auto strides = t.strides();
  return cudnn_frontend::TensorBuilder()
    .setDim(shape.size(), shape.data())
    .setStrides(strides.size(), strides.data())
    .setId(id)
    .setAlignment(alignment)
    .setDataType(getCudnnDataType(t))
    .build();
}

cudnn_frontend::ConvDesc_v8 getConvDescriptor(cudnnDataType_t dataType, IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, const at::ScalarType scalar_type) {
  uint64_t convDim = stride.size();
  cudnn_frontend::ConvDescBuilder_v8& builder = cudnn_frontend::ConvDescBuilder()
    .setDataType(dataType)
    .setMathMode(CUDNN_CROSS_CORRELATION)
    .setNDims(convDim)
    .setStrides(convDim, stride.data())
    .setPrePadding(convDim, padding.data())
    .setPostPadding(convDim, padding.data())
    .setDilation(convDim, dilation.data());
  if (scalar_type == kBFloat16 || scalar_type == kHalf) {
    builder.setDataType(CUDNN_DATA_FLOAT);
  }
  return builder.build();
}

void filterEngineConfigs(
  cudnn_frontend::EngineConfigList &from,
  cudnn_frontend::EngineConfigList &to,
  bool deterministic, bool allow_tf32, c10::ScalarType scalar_type)
{
  auto filter = [=](cudnnBackendDescriptor_t c) {
    if (deterministic) {
      if (cudnn_frontend::hasNumericalNote<CUDNN_NUMERICAL_NOTE_NONDETERMINISTIC>(c)) return true;
    }
    if (scalar_type == kFloat && !allow_tf32) {
      if (cudnn_frontend::hasNumericalNote<CUDNN_NUMERICAL_NOTE_DOWN_CONVERT_INPUTS>(c)) return true;
      if (cudnn_frontend::hasNumericalNote<CUDNN_NUMERICAL_NOTE_TENSOR_CORE>(c)) return true;
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

template <typename T>
struct BenchmarkCache {
std::mutex mutex;
std::unordered_map<CacheKey, cudnn_frontend::ExecutionPlan, ParamsHash<CacheKey>, ParamsEqual<CacheKey>> engine_cache;

cudnn_frontend::ExecutionPlan* find(const CacheKey& key) {
  std::lock_guard<std::mutex> guard(mutex);
  auto it = engine_cache.find(key);
  if (it == engine_cache.end()) {
    return NULL;
  }
  // TODO: probably want ExecutionPlan copy constructor or better way to return
  return &(it->second);
}

void emplace(const CacheKey& key, T& results) {
  std::lock_guard<std::mutex> guard(mutex);
  engine_cache.emplace(key, std::move(results));
}

};

BenchmarkCache<cudnn_frontend::ExecutionPlan> benchmark_cache;

} // namespace
void get_cachekey(CacheKey& key, const cudnnBackendDescriptorType_t operation, const Tensor& y, const Tensor& x, const Tensor& w, const IntArrayRef padding, const IntArrayRef stride, const IntArrayRef dilation, int64_t groups, bool deterministic, bool allow_tf32) {
   memset(&key, 0, sizeof(key));
   setConvolutionParams(&key.params, x, w, padding, stride, dilation, groups, deterministic, allow_tf32);
   key.operation = operation;
   key.x_alignment = getAlignment(x);
   key.y_alignment = getAlignment(y);
   key.w_alignment = getAlignment(w);
}

void run_conv_plan(cudnnHandle_t handle, const Tensor& x, const Tensor& y, const Tensor& w, const cudnn_frontend::ExecutionPlan& plan) {
    auto workspace_size = plan.getWorkspaceSize();
    Tensor workspace;
    workspace = at::empty({workspace_size}, x.options().dtype(kByte));
    void *data_ptrs[] = {x.data_ptr(), y.data_ptr(), w.data_ptr()};
    int64_t uids[] = {'x', 'y', 'w'};
    auto variantPack = cudnn_frontend::VariantPackBuilder()
        .setWorkspacePointer(workspace.data_ptr())
        .setDataPointers(3, data_ptrs)
        .setUids(3, uids)
        .build();
    AT_CUDNN_CHECK(cudnnBackendExecute(handle, plan.get_raw_desc(), variantPack.get_raw_desc()));
}

auto build_opgraph(const cudnnHandle_t handle, const cudnnBackendDescriptorType_t desc, const Tensor& x, const Tensor& y, const Tensor& w, const CacheKey& key, const IntArrayRef padding, const IntArrayRef stride, const IntArrayRef dilation) {
  auto op = cudnn_frontend::OperationBuilder(desc)
      .setxDesc(getTensorDescriptor(x, 'x', key.x_alignment))
      .setyDesc(getTensorDescriptor(y, 'y', key.y_alignment))
      .setwDesc(getTensorDescriptor(w, 'w', key.w_alignment))
      .setcDesc(getConvDescriptor(key.params.dataType, padding, stride, dilation, x.scalar_type()))
      .build();
  std::array<cudnn_frontend::Operation const *, 1> ops = {&op};
  auto opGraph = cudnn_frontend::OperationGraphBuilder()
      .setHandle(handle)
      .setOperationGraph(1, ops.data())
      .build();
  return opGraph;
}

const auto get_generator_sources(const cudnnBackendDescriptorType_t& desc, const Tensor& x, const bool deterministic, const bool allow_tf32) {
   // Method for engine config generator based on heuristics
  auto heurgen_method = [&desc, &x, deterministic, allow_tf32](cudnn_frontend::OperationGraph &opGraph) -> cudnn_frontend::EngineConfigList {
      auto heuristics = cudnn_frontend::EngineHeuristicsBuilder()
                            .setOperationGraph(opGraph)
                            .setHeurMode(CUDNN_HEUR_MODE_INSTANT)
                            .build();
      auto &engine_configs = heuristics.getEngineConfig(heuristics.getEngineConfigCount());
      cudnn_frontend::EngineConfigList filtered_configs;
      filterEngineConfigs(engine_configs, filtered_configs, deterministic, allow_tf32, x.scalar_type());
      return filtered_configs;
  };
  // Method for engine config generator based on fallback list
  auto fallback_method = [&desc, &x, deterministic, allow_tf32](cudnn_frontend::OperationGraph &opGraph) -> cudnn_frontend::EngineConfigList {
    auto fallback = cudnn_frontend::EngineFallbackListBuilder()
                        .setOperationGraph(opGraph)
                        .setOperation(desc)
                        .build();
    auto &fallback_list = fallback.getFallbackList();
    cudnn_frontend::EngineConfigList filtered_configs;
    filterEngineConfigs(fallback_list, filtered_configs, deterministic, allow_tf32, x.scalar_type());
    return filtered_configs;
  };
  std::array<cudnn_frontend::GeneratorSource const, 2> sources = {heurgen_method, fallback_method};
  return sources;
}

const auto get_fallback_method(const cudnn_frontend::OperationGraph &opGraph, const cudnnBackendDescriptorType_t& desc, const Tensor& x, const bool deterministic, const bool allow_tf32) {
  // Method for engine config generator based on fallback list
  auto fallback_method = [&](cudnn_frontend::OperationGraph &opGraph) -> cudnn_frontend::EngineConfigList {
    auto fallback = cudnn_frontend::EngineFallbackListBuilder()
                        .setOperationGraph(opGraph)
                        .setOperation(desc)
                        .build();
    auto &fallback_list = fallback.getFallbackList();
    cudnn_frontend::EngineConfigList filtered_configs;
    filterEngineConfigs(fallback_list, filtered_configs, deterministic, allow_tf32, x.scalar_type());
    return filtered_configs;
  };
  return fallback_method;
}

auto get_plans_from_find(const cudnnHandle_t handle, const cudnnBackendDescriptorType_t desc, const Tensor& x, const Tensor& y, const Tensor& w, const CacheKey& key, const IntArrayRef padding, const IntArrayRef stride, const IntArrayRef dilation, const bool deterministic, const bool allow_tf32) {
  auto opGraph = build_opgraph(handle, desc, x, y, w, key, padding, stride, dilation);
  void *data_ptrs[] = {x.data_ptr(), y.data_ptr(), w.data_ptr()};
  int64_t uids[] = {'x', 'y', 'w'};

  auto sources = get_generator_sources(desc, x, deterministic, allow_tf32);
  auto initial_predicate_function = [&](cudnn_frontend::ExecutionPlan const& plan) -> bool {
    return false;
  };
  cudnn_frontend::EngineConfigGenerator generator(sources.size(), sources.data());
  size_t max_workspace_size = 0u;
  auto plans = generator.cudnnGetPlan(handle, std::move(opGraph), initial_predicate_function);
  int device;
  THCudaCheck(cudaGetDevice(&device));
  size_t max_block_size = 0;
  size_t tmp_bytes = 0;  // Only used for filling pointer parameters that aren't used later
  c10::cuda::CUDACachingAllocator::cacheInfo(device, &tmp_bytes, &max_block_size);
  cudnn_frontend::executionPlans_t valid_plans;

  std::for_each(plans.begin(), plans.end(), [&] (cudnn_frontend::ExecutionPlan& plan) {
    size_t curr_workspace_size = plan.getWorkspaceSize();
    if (curr_workspace_size <= max_block_size) {
      if (curr_workspace_size > max_workspace_size) {
        max_workspace_size = plan.getWorkspaceSize();
      }
      valid_plans.emplace_back(std::move(plan));
    }
  });
  TORCH_CHECK_WITH(CUDAOutOfMemoryError, max_workspace_size < 1_TiB, "Not enough memory for workspace!");
  Tensor workspace;
  workspace = at::empty({max_workspace_size}, x.options().dtype(kByte));
  auto variantPack  = cudnn_frontend::VariantPackBuilder()
      .setDataPointers(3, data_ptrs)
      .setWorkspacePointer(workspace.data_ptr())
      .setUids(3, uids)
      .build();

  auto options = cudnn_frontend::time_sorted_plan<cudnn_frontend::CudnnFindSamplingTechnique::CUDNN_FIND_SAMPLE_TILL_STABLE>(handle, std::move(valid_plans), variantPack);

  cudnn_frontend::executionPlans_t sorted_plans;
  for (auto& option : options) {
    sorted_plans.emplace_back(std::move(option.plan));
  }
  return sorted_plans;
}

auto get_plans_from_heuristics(const cudnnHandle_t handle, const cudnnBackendDescriptorType_t desc, const Tensor& x, const Tensor& y, const Tensor& w, const CacheKey& key, const IntArrayRef padding, const IntArrayRef stride, const IntArrayRef dilation, const bool deterministic, const bool allow_tf32) {
  auto workspace_size = 1LL << 30;
  Tensor workspace;
  workspace = at::empty({workspace_size}, x.options().dtype(kByte));
  auto opGraph = build_opgraph(handle, desc, x, y, w, key, padding, stride, dilation);
  void *data_ptrs[] = {x.data_ptr(), y.data_ptr(), w.data_ptr()};
  int64_t uids[] = {'x', 'y', 'w'};
  auto variantPack  = cudnn_frontend::VariantPackBuilder()
      .setDataPointers(3, data_ptrs)
      .setWorkspacePointer(workspace.data_ptr())
      .setUids(3, uids)
      .build();
  auto predicate_function = [&](cudnn_frontend::ExecutionPlan const& plan) -> bool {
    return plan.getWorkspaceSize() > workspace_size;
  };

  auto sources = get_generator_sources(desc, x, deterministic, allow_tf32);

  cudnn_frontend::EngineConfigGenerator generator(sources.size(), sources.data());
  auto plans = generator.cudnnGetPlan(handle, std::move(opGraph), predicate_function);
  return plans;
}

void try_plans(cudnn_frontend::executionPlans_t& plans, const CacheKey& key, const cudnnHandle_t handle, const Tensor& x, const Tensor& y, const Tensor& w) {
  for (auto & plan : plans) {
    try {
      run_conv_plan(handle, x, y, w, plan);
      benchmark_cache.emplace(key, plan);
      // engine_cache.emplace(key, std::move(plan));
      return;
    } catch (cudnn_frontend::cudnnException &e) {} catch(CuDNNError &e) {}
  }
  TORCH_CHECK(false, "Unable to find an engine to execute this computation");
}

void run_single_conv(const cudnnBackendDescriptorType_t operation,
  const Tensor& x, const Tensor& y, const Tensor& w,
  const IntArrayRef padding, const IntArrayRef stride, const IntArrayRef dilation, const int64_t groups,
  const bool benchmark, const bool deterministic, const bool allow_tf32) {
  cudnnHandle_t handle = getCudnnHandle();

  CacheKey key;
  get_cachekey(key, operation, y, x, w, padding, stride, dilation, groups, deterministic, allow_tf32);
  auto search = benchmark_cache.find(key);
  if (search) {
    run_conv_plan(handle, x, y, w, *search);
    return;
  }

  cudnn_frontend::executionPlans_t plans;
  if (!benchmark) {
    plans = get_plans_from_heuristics(handle, operation, x, y, w, key, padding, stride, dilation, deterministic, allow_tf32);
  } else {
    plans = get_plans_from_find(handle, operation, x, y, w, key, padding, stride, dilation, deterministic, allow_tf32);
  }
  try_plans(plans, key, handle, x, y, w);
}

void raw_cudnn_convolution_forward_out(
    const Tensor& output, const Tensor& input, const Tensor& weight,
    const IntArrayRef padding, const IntArrayRef stride, const IntArrayRef dilation, const int64_t groups,
    const bool benchmark, const bool deterministic, const bool allow_tf32)
{
  if (output.numel() > 0) {
    run_single_conv(CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR,
      input, output, weight, padding, stride, dilation, groups,
      benchmark, deterministic, allow_tf32);
  }
}

void raw_cudnn_convolution_backward_input_out(
    const at::Tensor& grad_input,
    const at::Tensor& grad_output,
    const at::Tensor& weight,
    const IntArrayRef padding, const IntArrayRef stride, const IntArrayRef dilation, const int64_t groups,
    const bool benchmark, const bool deterministic, const bool allow_tf32) {
  if (grad_input.numel() > 0) {
    run_single_conv(CUDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_DATA_DESCRIPTOR,
      grad_input, grad_output, weight, padding, stride, dilation, groups,
      benchmark, deterministic, allow_tf32);
  }
}

void raw_cudnn_convolution_backward_weight_out(
    const Tensor& grad_weight, const Tensor& grad_output, const Tensor& input,
    const IntArrayRef padding, const IntArrayRef stride, const IntArrayRef dilation, const int64_t groups,
    const bool benchmark, const bool deterministic, const bool allow_tf32) {
  if (grad_weight.numel() > 0) {
    run_single_conv(CUDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_FILTER_DESCRIPTOR,
      input, grad_output, grad_weight, padding, stride, dilation, groups,
      benchmark, deterministic, allow_tf32);
  }
}

}} // at::native

#endif  // HAS_CUDNN_V8
#endif  // AT_CUDNN_ENABLED
