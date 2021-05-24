#include <ATen/cuda/CUDAConfig.h>  // for the definition of AT_CUDNN_ENABLED

#if AT_CUDNN_ENABLED()

#include <ATen/native/cudnn/Macros.h>

#if HAS_CUDNN_V8()

#include <ATen/cudnn/cudnn-wrapper.h>
#include <cudnn_frontend.h>
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
  uint64_t address = reinterpret_cast<uint64_t>(t.data_ptr());
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
  cudnnBackendDescriptorType_t dir;
  uint8_t input_alignment;
  uint8_t weight_alignment;
  uint8_t output_alignment;
};

// FIXME: make this thread-safe by reusing the benchmark cache in Conv_v7.cpp
std::unordered_map<CacheKey, cudnn_frontend::ManagedOpaqueDescriptor, ParamsHash<CacheKey>, ParamsEqual<CacheKey>> engine_cache;

}

void get_cachekey(CacheKey& key, const cudnnBackendDescriptorType_t dir, const Tensor& convoutput, const Tensor& convinput1, const Tensor& convinput2, IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups, bool deterministic, bool allow_tf32) {
   setConvolutionParams(&key.params, convinput1, convinput2, padding, stride, dilation, groups, deterministic, allow_tf32);
   key.dir = dir;
   key.input_alignment = getAlignment(convinput1);
   key.output_alignment = getAlignment(convoutput);
   key.weight_alignment = getAlignment(convinput2);
}

void run_conv_cfg(cudnnHandle_t handle, const Tensor& x, const Tensor& y, const Tensor& w, cudnn_frontend::ManagedOpaqueDescriptor cfg) {
    auto plan = cudnn_frontend::ExecutionPlanBuilder()
        .setHandle(handle)
        .setEngineConfig(cfg)
        .build();

    auto workspace_size = plan.getWorkspaceSize();
    // TODO: is always using 'x' options() ok? e.g., in backward data
    Tensor workspace;
    workspace = at::empty({workspace_size}, x.options().dtype(kByte));
    void *data_ptrs[] = {x.data_ptr(), y.data_ptr(), w.data_ptr()};
    // std::cout << plan.describe() << " requires workspace " << workspace_size << std::endl;
    int64_t uids[] = {'x', 'y', 'w'};
    auto variantPack = cudnn_frontend::VariantPackBuilder()
        .setWorkspacePointer(workspace.data_ptr())
        .setDataPointers(3, data_ptrs)
        .setUids(3, uids)
        .build();
    AT_CUDNN_CHECK(cudnnBackendExecute(handle, plan.get_raw_desc(), variantPack.get_raw_desc()));
}

void get_configs(cudnn_frontend::EngineConfigList& filtered_configs, cudnnHandle_t handle, cudnnBackendDescriptorType_t desc, const Tensor& x, const Tensor& y, const Tensor& w, CacheKey key, IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, bool deterministic, bool allow_tf32) {
  auto op = cudnn_frontend::OperationBuilder(desc)
      .setxDesc(getTensorDescriptor(x, 'x', key.input_alignment))
      .setyDesc(getTensorDescriptor(y, 'y', key.output_alignment))
      .setwDesc(getTensorDescriptor(w, 'w', key.weight_alignment))
      .setcDesc(getConvDescriptor(key.params.dataType, padding, stride, dilation))
      .build();

  std::array<cudnn_frontend::Operation const *, 1> ops = {&op};

  auto opGraph = cudnn_frontend::OperationGraphBuilder()
      .setHandle(handle)
      .setOperationGraph(1, ops.data())
      .build();
  auto heuristics = cudnn_frontend::EngineHeuristicsBuilder()
      .setOperationGraph(opGraph)
      .setHeurMode(CUDNN_HEUR_MODE_INSTANT)
      .build();
  auto fallback = cudnn_frontend::EngineFallbackListBuilder()
                    .setOperationGraph(opGraph)
                    .setOperation(desc)
                    .build();
  auto& engine_configs = heuristics.getEngineConfig(heuristics.getEngineConfigCount());
  auto& fallback_list = fallback.getFallbackList();

  // TODO: Is this ok or do we need to change type deduction based on descriptor?
  filterEngineConfigs(engine_configs, filtered_configs, deterministic, allow_tf32, x.scalar_type());
  filterEngineConfigs(fallback_list, filtered_configs, deterministic, allow_tf32, x.scalar_type());
}


void try_filtered_configs(const cudnn_frontend::EngineConfigList filtered_configs, const CacheKey key, const cudnnHandle_t handle, const Tensor& x, const Tensor& y, const Tensor& w) {
  for (auto &cfg : filtered_configs) {
    try {
      run_conv_cfg(handle, x, y, w, cfg);
      engine_cache[key] = cfg;
      return;
    } catch (cudnn_frontend::cudnnException &e) {} catch(CuDNNError &e) {}
  }
  TORCH_CHECK(false, "Unable to find an engine to execute this computation");
}


void raw_cudnn_convolution_forward_out(
    const Tensor& output, const Tensor& input, const Tensor& weight,
    IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups,
    bool benchmark, bool deterministic, bool allow_tf32)
{
  TORCH_CHECK(!benchmark, "not supported yet");
  if (output.numel() == 0) {
    return;
  }
  cudnnHandle_t handle = getCudnnHandle();

  CacheKey key;
  get_cachekey(key, CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR, output, input, weight, padding, stride, dilation, groups, deterministic, allow_tf32);
  auto search = engine_cache.find(key);
  if (search != engine_cache.end()) {
    run_conv_cfg(handle, input, output, weight, search->second);
    return;
  }

  cudnn_frontend::EngineConfigList filtered_configs;
  get_configs(filtered_configs, handle, CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR, input, output, weight, key, padding, stride, dilation, deterministic, allow_tf32);

  try_filtered_configs(filtered_configs, key, handle, input, output, weight);
}

void raw_cudnn_convolution_backward_input_out(
    const at::Tensor& grad_input,
    const at::Tensor& grad_output,
    const at::Tensor& weight,
    IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups,
    bool benchmark, bool deterministic, bool allow_tf32) {
  TORCH_CHECK(!benchmark, "not supported yet");
  if (grad_input.numel() == 0) {
    return;
  }
  cudnnHandle_t handle = getCudnnHandle();

  CacheKey key;
  get_cachekey(key, CUDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_DATA_DESCRIPTOR, grad_output, grad_input, weight, padding, stride, dilation, groups, deterministic, allow_tf32);
  auto search = engine_cache.find(key);
  if (search != engine_cache.end()) {
    run_conv_cfg(handle, grad_input, grad_output, weight, search->second);
    return;
  }

  cudnn_frontend::EngineConfigList filtered_configs;
  get_configs(filtered_configs, handle, CUDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_DATA_DESCRIPTOR, grad_input, grad_output, weight, key, padding, stride, dilation, deterministic, allow_tf32);
  try_filtered_configs(filtered_configs, key, handle, grad_input, grad_output, weight);
}

void raw_cudnn_convolution_backward_weight_out(
    const Tensor& grad_weight, const Tensor& grad_output, const Tensor& input,
    IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups,
    bool benchmark, bool deterministic, bool allow_tf32) {
  TORCH_CHECK(!benchmark, "not supported yet");
  if (grad_weight.numel() == 0) {
    return;
  }
  cudnnHandle_t handle = getCudnnHandle();

  CacheKey key;
  get_cachekey(key, CUDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_FILTER_DESCRIPTOR, grad_output, input, grad_weight, padding, stride, dilation, groups, deterministic, allow_tf32);

  auto search = engine_cache.find(key);
  if (search != engine_cache.end()) {
    run_conv_cfg(handle, input, grad_output, grad_weight, search->second);
    return;
  }

  cudnn_frontend::EngineConfigList filtered_configs;
  get_configs(filtered_configs, handle, CUDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_FILTER_DESCRIPTOR, input, grad_output, grad_weight, key, padding, stride, dilation, deterministic, allow_tf32);
  try_filtered_configs(filtered_configs, key, handle, input, grad_output, grad_weight);
}

}} // at::native

#endif  // HAS_CUDNN_V8
#endif  // AT_CUDNN_ENABLED
