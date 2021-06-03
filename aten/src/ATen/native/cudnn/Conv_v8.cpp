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

#include <unordered_map>

namespace at { namespace native{

namespace {

uint8_t getAlignment(const Tensor &t) {
  // alignment are in bytes
  uint8_t alignment = 1;
  uintptr_t address = reinterpret_cast<uintptr_t>(t.data_ptr());
  while (address % alignment == 0 && alignment < 16) alignment *= 2;
  return alignment;
}

cudnn_frontend::Tensor getTensorDescriptor(const Tensor &t, int64_t id, uint8_t alignment) {
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

cudnn_frontend::ConvDesc_v8 getConvDescriptor(cudnnDataType_t dataType, IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation) {
  uint64_t convDim = stride.size();
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
      if (cudnn_frontend::hasNumericalNote<CUDNN_NUMERICAL_NOTE_NONDETERMINISTIC>(c)) return true;
    }
    if (scalar_type == kFloat || !allow_tf32) {
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

// FIXME: make this thread-safe by reusing the benchmark cache in Conv_v7.cpp
std::unordered_map<CacheKey, std::unique_ptr<cudnn_frontend::ExecutionPlan>, ParamsHash<CacheKey>, ParamsEqual<CacheKey>> engine_cache;

}

void get_cachekey(CacheKey& key, const cudnnBackendDescriptorType_t operation, const Tensor& y, const Tensor& x, const Tensor& w, IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups, bool deterministic, bool allow_tf32) {
   setConvolutionParams(&key.params, x, w, padding, stride, dilation, groups, deterministic, allow_tf32);
   key.operation = operation;
   key.x_alignment = getAlignment(x);
   key.y_alignment = getAlignment(y);
   key.w_alignment = getAlignment(w);
}

void run_conv_plan(cudnnHandle_t handle, const Tensor& x, const Tensor& y, const Tensor& w, cudnn_frontend::ExecutionPlan& plan) {
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

auto get_plans_from_find(cudnnHandle_t handle, cudnnBackendDescriptorType_t desc, const Tensor& x, const Tensor& y, const Tensor& w, CacheKey key, IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, bool deterministic, bool allow_tf32) {
   auto op = cudnn_frontend::OperationBuilder(desc)
      .setxDesc(getTensorDescriptor(x, 'x', key.x_alignment))
      .setyDesc(getTensorDescriptor(y, 'y', key.y_alignment))
      .setwDesc(getTensorDescriptor(w, 'w', key.w_alignment))
      .setcDesc(getConvDescriptor(key.params.dataType, padding, stride, dilation))
      .build();
  std::array<cudnn_frontend::Operation const *, 1> ops = {&op};
  auto opGraph = cudnn_frontend::OperationGraphBuilder()
      .setHandle(handle)
      .setOperationGraph(1, ops.data())
      .build();

  void *data_ptrs[] = {x.data_ptr(), y.data_ptr(), w.data_ptr()};
  int64_t uids[] = {'x', 'y', 'w'};
  auto variantPack  = cudnn_frontend::VariantPackBuilder().setDataPointers(3, data_ptrs).setUids(3, uids).build();
  auto sample_predicate_function = [](cudnn_frontend::ExecutionPlan const& plan) -> bool {
    return false;
  };

  // Method for engine config generator based on heuristics
  auto heurgen_method = [&](cudnn_frontend::OperationGraph &opGraph) -> cudnn_frontend::EngineConfigList {
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

  std::array<cudnn_frontend::GeneratorSource const, 2> sources = {heurgen_method, fallback_method};
  cudnn_frontend::EngineConfigGenerator generator(sources.size(), sources.data());
  auto options = generator.cudnnFindPlan<cudnn_frontend::CudnnFindSamplingTechnique::CUDNN_FIND_SAMPLE_TILL_STABLE>(handle, std::move(opGraph), variantPack, sample_predicate_function); 

  cudnn_frontend::executionPlans_t plans;
  for (auto& option : options) {
    plans.emplace_back(std::move(option.plan));
  }
  std::cout << "find plans: " << plans.size() << std::endl;
  return plans; 
}

auto get_plans_from_heuristics(cudnnHandle_t handle, cudnnBackendDescriptorType_t desc, const Tensor& x, const Tensor& y, const Tensor& w, CacheKey key, IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, bool deterministic, bool allow_tf32) {
  auto op = cudnn_frontend::OperationBuilder(desc)
      .setxDesc(getTensorDescriptor(x, 'x', key.x_alignment))
      .setyDesc(getTensorDescriptor(y, 'y', key.y_alignment))
      .setwDesc(getTensorDescriptor(w, 'w', key.w_alignment))
      .setcDesc(getConvDescriptor(key.params.dataType, padding, stride, dilation))
      .build();
  std::array<cudnn_frontend::Operation const *, 1> ops = {&op};
  auto opGraph = cudnn_frontend::OperationGraphBuilder()
      .setHandle(handle)
      .setOperationGraph(1, ops.data())
      .build();
  void *data_ptrs[] = {x.data_ptr(), y.data_ptr(), w.data_ptr()};
  int64_t uids[] = {'x', 'y', 'w'};
  auto variantPack  = cudnn_frontend::VariantPackBuilder().setDataPointers(3, data_ptrs).setUids(3, uids).build();
  auto sample_predicate_function = [](cudnn_frontend::ExecutionPlan const& plan) -> bool {
    return false;
  };

  // Method for engine config generator based on heuristics
  auto heurgen_method = [&](cudnn_frontend::OperationGraph &opGraph) -> cudnn_frontend::EngineConfigList {
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


  std::array<cudnn_frontend::GeneratorSource const, 2> sources = {heurgen_method, fallback_method};
  cudnn_frontend::EngineConfigGenerator generator(sources.size(), sources.data());
  auto plans = generator.cudnnGetPlan(handle, std::move(opGraph), sample_predicate_function);
  std::cout << "get plans: " << plans.size() << std::endl;
  return plans;
}

void try_plans(cudnn_frontend::executionPlans_t& plans, const CacheKey key, const cudnnHandle_t handle, const Tensor& x, const Tensor& y, const Tensor& w) {
  for (auto & plan : plans) {
    try {
      std::unique_ptr<cudnn_frontend::ExecutionPlan> plan_ptr(new cudnn_frontend::ExecutionPlan(std::move(plan)));

      run_conv_plan(handle, x, y, w, *plan_ptr.get());
      engine_cache[key] = std::move(plan_ptr);
      return;
    } catch (cudnn_frontend::cudnnException &e) {} catch(CuDNNError &e) {}
  }
  TORCH_CHECK(false, "Unable to find an engine to execute this computation");
}

void try_options(cudnn_frontend::executionOptions_t & options, const CacheKey key, const cudnnHandle_t handle, const Tensor& x, const Tensor& y, const Tensor& w) {
  std::cout << options.size() << std::endl;
  for (auto& option : options) {
    std::cout << " option finished in " << option.time_ms << std::endl;
  }
  for (auto& option : options) {
    try {
      std::unique_ptr<cudnn_frontend::ExecutionPlan> plan_ptr(new cudnn_frontend::ExecutionPlan(std::move(option.plan)));
      run_conv_plan(handle, x, y, w, *plan_ptr.get());
      engine_cache[key] = std::move(plan_ptr);
      return;
    } catch (cudnn_frontend::cudnnException &e) {} catch(CuDNNError &e) {}
  }
  TORCH_CHECK(false, "Unable to find an engine to execute this computation");
}

void run_single_conv(const cudnnBackendDescriptorType_t operation,
  const Tensor& x, const Tensor& y, const Tensor& w,
  IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups,
  bool benchmark, bool deterministic, bool allow_tf32) {

  cudnnHandle_t handle = getCudnnHandle();

  CacheKey key;
  get_cachekey(key, operation, y, x, w, padding, stride, dilation, groups, deterministic, allow_tf32);
  auto search = engine_cache.find(key);
  if (search != engine_cache.end()) {
    run_conv_plan(handle, x, y, w, *(search->second.get()));
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
    IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups,
    bool benchmark, bool deterministic, bool allow_tf32)
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
    IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups,
    bool benchmark, bool deterministic, bool allow_tf32) {
  if (grad_input.numel() > 0) {
    run_single_conv(CUDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_DATA_DESCRIPTOR,
      grad_input, grad_output, weight, padding, stride, dilation, groups,
      benchmark, deterministic, allow_tf32);
  }
}

void raw_cudnn_convolution_backward_weight_out(
    const Tensor& grad_weight, const Tensor& grad_output, const Tensor& input,
    IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups,
    bool benchmark, bool deterministic, bool allow_tf32) {
  if (grad_weight.numel() > 0) {
    run_single_conv(CUDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_FILTER_DESCRIPTOR,
      input, grad_output, grad_weight, padding, stride, dilation, groups,
      benchmark, deterministic, allow_tf32);
  }
}

}} // at::native

#endif  // HAS_CUDNN_V8
#endif  // AT_CUDNN_ENABLED
