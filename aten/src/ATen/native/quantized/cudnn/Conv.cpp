#include <ATen/cuda/CUDAConfig.h>  // for the definition of AT_CUDNN_ENABLED

#if AT_CUDNN_ENABLED()

#include <ATen/native/cudnn/Macros.h>

#if HAS_CUDNN_V8()

#include <cudnn_frontend.h>
#include <ATen/ATen.h>
#include <ATen/TensorUtils.h>
#include <ATen/cuda/Exceptions.h>
#include <ATen/native/ConvUtils.h>
#include <ATen/native/cudnn/ConvShared.h>
#include <ATen/native/utils/ParamsHash.h>
#include <ATen/cudnn/Handle.h>
#include <ATen/TensorUtils.h>
#include <torch/library.h>

#include <unordered_map>
#include <iostream>

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
  std::cout << "determinsitic:" << deterministic << " allow_tf32:" << allow_tf32 << std::endl;
  auto filter = [=](cudnnBackendDescriptor_t c) {
    if (deterministic) {
      if (cudnn_frontend::hasNumericalNote<CUDNN_NUMERICAL_NOTE_NONDETERMINISTIC>(c)) return true;
    }
    if (scalar_type == kFloat || !allow_tf32) {
      if (cudnn_frontend::hasNumericalNote<CUDNN_NUMERICAL_NOTE_DOWN_CONVERT_INPUTS>(c)) return true;
      std::cout << "after down convert" << std::endl;
      if (cudnn_frontend::hasNumericalNote<CUDNN_NUMERICAL_NOTE_TENSOR_CORE>(c)) return true;
      std::cout << "after tensorcore" << std::endl;
    }
    return false;
  };
  cudnn_frontend::filter(from, to, filter);
}

cudnn_frontend::ExecutionPlan
get_execplan_from_heuristics_else_fall_back(cudnn_frontend::OperationGraph&& opGraph, cudnnHandle_t handle_) {
  auto heuristics = cudnn_frontend::EngineHeuristicsBuilder()
    .setOperationGraph(opGraph)
    .setHeurMode(CUDNN_HEUR_MODE_INSTANT)
    .build();

  std::cout << "Heuristic has " << heuristics.getEngineConfigCount() << " configurations " << std::endl;
  auto& engine_config = heuristics.getEngineConfig(heuristics.getEngineConfigCount());

  // Try engine configs returned by the heuristics and pick up the first one that works.
  for (auto& ecfg : engine_config) {
    try {
      auto plan = cudnn_frontend::ExecutionPlanBuilder()
        .setHandle(handle_)
        .setEngineConfig(ecfg, opGraph.getTag())
        .build();
      return plan;
    } catch (cudnn_frontend::cudnnException& e) {
      continue;
    }
  }
}

struct CacheKey {
  ConvolutionParams params;
  uint8_t input_alignment;
  uint8_t weight_alignment;
  uint8_t output_alignment;
};

// FIXME: make this thread-safe by reusing the benchmark cache in Conv_v7.cpp
std::unordered_map<CacheKey, cudnn_frontend::ManagedOpaqueDescriptor, ParamsHash<CacheKey>, ParamsEqual<CacheKey>> engine_cache;

template <int kSpatialDim>
at::SmallVector<int64_t, kSpatialDim + 2> MakeConvOutputShape(
    int N, // mini-batch
    int M, // output channels
    const std::array<int64_t, kSpatialDim>& input_image_shape,
    const std::vector<int64_t>& kernel,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation);

template <>
at::SmallVector<int64_t, 4> MakeConvOutputShape<2>(
    int N, // mini-batch
    int M, // output channels
    const std::array<int64_t, 2>& input_image_shape,
    const std::vector<int64_t>& kernel,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation) {
  const int H = input_image_shape[0];
  const int W = input_image_shape[1];
  const int64_t Y_H =
      (H + 2 * padding[0] - dilation[0] * (kernel[0] - 1) - 1) / stride[0] + 1;
  const int64_t Y_W =
      (W + 2 * padding[1] - dilation[1] * (kernel[1] - 1) - 1) / stride[1] + 1;
  return {N, M, Y_H, Y_W};
}

void raw_cudnn_convolution_forward_out(
    const Tensor& output,
    const Tensor& input,
    const Tensor& weight,
    IntArrayRef padding,
    IntArrayRef stride,
    IntArrayRef dilation,
    int64_t groups,
    bool benchmark, bool deterministic, bool allow_tf32) {
  TORCH_CHECK(!benchmark, "not supported yet");
  if (output.numel() == 0) {
    return;
  }

  cudnnHandle_t handle = getCudnnHandle();

  CacheKey key;
  setConvolutionParams(&key.params, input, weight, padding, stride, dilation, groups, deterministic, allow_tf32);
  // currently we are assuming accumulation data type is the same as the data
  // type for input, that is not ideal for int8 conv ops, we want to use fp32
  // for accumulation to preserve the accuracy
  key.params.dataType = getCudnnDataType(output);
  key.input_alignment = getAlignment(input);
  key.output_alignment = getAlignment(output);
  key.weight_alignment = getAlignment(weight);

  auto run = [&](cudnn_frontend::ManagedOpaqueDescriptor cfg) {
    std::cout << "building plan:" << std::endl;
    auto plan = cudnn_frontend::ExecutionPlanBuilder()
        .setHandle(handle)
        .setEngineConfig(cfg)
        .build();

    auto workspace_size = plan.getWorkspaceSize();
    auto workspace = at::empty({workspace_size}, input.options().dtype(kByte));
    void *data_ptrs[] = {reinterpret_cast<int8_t*>(input.data_ptr()), output.data_ptr(), reinterpret_cast<int8_t*>(weight.data_ptr())};
    std::cout << plan.describe() << " requires workspace " << workspace_size << std::endl;
    int64_t uids[] = {'x', 'y', 'w'};
    auto variantPack = cudnn_frontend::VariantPackBuilder()
        .setWorkspacePointer(workspace.data_ptr())
        .setDataPointers(3, data_ptrs)
        .setUids(3, uids)
        .build();
    AT_CUDNN_CHECK(cudnnBackendExecute(handle, plan.get_raw_desc(), variantPack.get_raw_desc()));
  };

  auto search = engine_cache.find(key);
  if (search != engine_cache.end()) {
    run(search->second);
    return;
  }

  auto op = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR)
      .setxDesc(getTensorDescriptor(input, 'x', key.input_alignment))
      .setyDesc(getTensorDescriptor(output, 'y', key.output_alignment))
      .setwDesc(getTensorDescriptor(weight, 'w', key.weight_alignment))
      .setcDesc(getConvDescriptor(key.params.dataType, padding, stride, dilation))
      .build();
  std::cout << "operator:" << op.describe() << std::endl;
  // TODO: add bias
  // TODO: move requantize here

  std::array<cudnn_frontend::Operation const *, 1> ops = {&op};

  auto opGraph = cudnn_frontend::OperationGraphBuilder()
      .setHandle(handle)
      .setOperationGraph(1, ops.data())
      .build();
  std::cout << "opGraph: " << opGraph.describe() << std::endl;

  auto heuristics = cudnn_frontend::EngineHeuristicsBuilder()
      .setOperationGraph(opGraph)
      .setHeurMode(CUDNN_HEUR_MODE_INSTANT)
      .build();
  auto fallback = cudnn_frontend::EngineFallbackListBuilder()
                    .setOperationGraph(opGraph)
                    .setOperation(CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR)
                    .build();

  auto& engine_configs = heuristics.getEngineConfig(heuristics.getEngineConfigCount());
  auto& fallback_list = fallback.getFallbackList();

  //cudnn_frontend::EngineConfigList filtered_configs;
  //filterEngineConfigs(engine_configs, filtered_configs, deterministic, allow_tf32, input.scalar_type());
  // filterEngineConfigs(fallback_list, filtered_configs, deterministic, allow_tf32, input.scalar_type());

  for (auto &cfg : engine_configs) {
    try {
      run(cfg);
      engine_cache[key] = cfg;
      return;
    } catch (cudnn_frontend::cudnnException &e) {std::cout << "cudnn:" << e.what() << std::endl;} catch(CuDNNError &e) { std::cout << e.what() << std::endl;}
  }
  TORCH_CHECK(false, "Unable to find an engine to execute this computation");
}

//
// output Tensor will be a fp32 Tensor
// both act and weight will be int8 Tensor
//
template <int kSpatialDim>
Tensor raw_cudnn_convolution_forward(
    const Tensor& act,
    const Tensor& weight,
    IntArrayRef padding,
    IntArrayRef stride,
    IntArrayRef dilation,
    int64_t groups,
    bool benchmark, bool deterministic, bool allow_tf32) {
  // TODO: add dimension validations for input/weight/bias
  const int N = act.size(0);
  const int C = act.size(1);
  const int D = kSpatialDim == 3 ? act.size(2) : 1;
  const int H = act.size(kSpatialDim);
  const int W = act.size(kSpatialDim + 1);
  const int M = weight.size(0); // output channels
  std::vector<int64_t> kernel_size = {weight.size(2), weight.size(3)};
  at::SmallVector<int64_t, kSpatialDim + 2> output_shape;
  output_shape = MakeConvOutputShape<kSpatialDim>(N, M, {H, W}, kernel_size, stride, padding, dilation);
  Tensor output_fp32 = at::empty(
      output_shape,
      at::device(at::kCUDA).dtype(at::kFloat),
      at::MemoryFormat::Contiguous
  );

  raw_cudnn_convolution_forward_out(
      output_fp32, act, weight,
      padding, stride, dilation, groups,
      benchmark,
      deterministic,
      allow_tf32);
  return output_fp32;
}


template <int kSpatialDim, bool kReluFused>
class QConvInt8 final {
 public:
  static Tensor run(
      Tensor act,
      Tensor weight,
      c10::optional<Tensor> bias,
      torch::List<int64_t> stride,
      torch::List<int64_t> padding,
      torch::List<int64_t> dilation,
      int64_t groups,
      double output_scale,
      int64_t output_zero_point) {
    TORCH_CHECK(!kReluFused, "conv relu not supported yet");
    TORCH_CHECK(!bias.has_value(), "bias is not supported yet");
    std::cout << "before run" << std::endl;
    // TODO: check all zero_points are zero/all tensors are symmetrically quantized
    Tensor output_fp32 = raw_cudnn_convolution_forward<kSpatialDim>(
        act.int_repr(), weight.int_repr(),
        IntArrayRef(padding.vec()), IntArrayRef(stride.vec()), IntArrayRef(dilation.vec()), groups,
        false /* benchmark */,
        true /* deterministic */,
        false /* allow_tf32 */);

    std::cout << "before requantize" << std::endl;
    // requantization
    // out_int8 = act_int8 * weight_int8 * act_scale * w_scale / output_scale
    auto act_scale = act.q_scale();
    auto weight_scale = weight.q_scale();
    auto requantize_multiplier = act_scale * weight_scale / output_scale;
    Tensor out = output_fp32 * requantize_multiplier;
    std::cout << "before reassemble" << std::endl;
    // convert output fp32 Tensor to int8 by clamping
    // TODO: get the range based on target output dtype, for now
    // we hardcode this to the range for int8
    Tensor clampped = out.clamp(-128, 127);
    Tensor quantized_output = at::_make_per_tensor_quantized_tensor(clampped, output_scale, output_zero_point);
    std::cout << "after reassemble" << std::endl;
    return quantized_output;
  }
};

TORCH_LIBRARY_IMPL(quantized, QuantizedCUDA, m) {
  m.impl(TORCH_SELECTIVE_NAME("quantized::conv2d_cudnn"), QConvInt8<2, false>::run);
}

} // namespace
}} // at::native

#endif  // HAS_CUDNN_V8
#endif  // AT_CUDNN_ENABLED
