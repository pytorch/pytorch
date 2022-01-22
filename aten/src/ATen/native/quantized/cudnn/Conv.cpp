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

cudnn_frontend::PointWiseDesc_v8 getPointWiseMulDescriptor(cudnnDataType_t dataType) {
  return cudnn_frontend::PointWiseDescBuilder()
    .setMode(CUDNN_POINTWISE_MUL)
    .setMathPrecision(dataType)
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
    if (scalar_type == kFloat || scalar_type == kChar || !allow_tf32) {
      if (cudnn_frontend::hasNumericalNote<CUDNN_NUMERICAL_NOTE_DOWN_CONVERT_INPUTS>(c)) return true;
      if (cudnn_frontend::hasNumericalNote<CUDNN_NUMERICAL_NOTE_TENSOR_CORE>(c)) return true;
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

  // std::cout << "Heuristic has " << heuristics.getEngineConfigCount() << " configurations " << std::endl;
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

  {
    auto total_engines = opGraph.getEngineCount();
    // std::cout << opGraph.describe() << " has " << total_engines << " engines." << std::endl;
    auto engine = cudnn_frontend::EngineBuilder().setGlobalEngineIdx(0).setOperationGraph(opGraph).build();
    // std::cout << engine.describe() << std::endl;

    auto engine_config = cudnn_frontend::EngineConfigBuilder().setEngine(engine).build();
    // std::cout << engine_config.describe() << std::endl;

    return cudnn_frontend::ExecutionPlanBuilder().setHandle(handle_).setEngineConfig(engine_config).build();
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
    bool benchmark,
    bool deterministic,
    bool allow_tf32,
    float requantize_multiplier
) {
  TORCH_CHECK(!benchmark, "not supported yet");
  if (output.numel() == 0) {
    return;
  }

  Tensor conv_output = at::empty_like(output);
  Tensor requantize_multiplier_tensor = at::empty_like(output);
  requantize_multiplier_tensor.fill_(requantize_multiplier);
  cudnnHandle_t handle = getCudnnHandle();

  CacheKey key;
  setConvolutionParams(&key.params, input, weight, padding, stride, dilation, groups, deterministic, allow_tf32);
  // operator datatype needs to be int32 for int8 convolution, but we can
  // set the datatype for output tensor to int32 or fp32
  key.params.dataType = CUDNN_DATA_INT32;
  key.input_alignment = getAlignment(input);
  key.output_alignment = getAlignment(conv_output);
  key.weight_alignment = getAlignment(weight);

  auto run = [&](cudnn_frontend::ManagedOpaqueDescriptor cfg) {
    auto plan = cudnn_frontend::ExecutionPlanBuilder()
        .setHandle(handle)
        .setEngineConfig(cfg)
        .build();

    auto workspace_size = plan.getWorkspaceSize();
    auto workspace = at::empty({workspace_size}, input.options().dtype(kByte));
    void *data_ptrs[] = {reinterpret_cast<int8_t*>(input.data_ptr()), conv_output.data_ptr(), reinterpret_cast<int8_t*>(weight.data_ptr()), requantize_multiplier_tensor.data_ptr(), output.data_ptr()};
    // std::cout << plan.describe() << " requires workspace " << workspace_size << std::endl;
    int64_t uids[] = {'x', 'y', 'w', 's', 'r'};
    auto variantPack = cudnn_frontend::VariantPackBuilder()
        .setWorkspacePointer(workspace.data_ptr())
        .setDataPointers(5, data_ptrs)
        .setUids(5, uids)
        .build();
    AT_CUDNN_CHECK(cudnnBackendExecute(handle, plan.get_raw_desc(), variantPack.get_raw_desc()));
  };

  auto search = engine_cache.find(key);
  if (search != engine_cache.end()) {
    run(search->second);
    return;
  }

  auto conv_op = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR)
      .setxDesc(getTensorDescriptor(input, 'x', key.input_alignment))
      .setyDesc(getTensorDescriptor(conv_output, 'y', key.output_alignment))
      .setwDesc(getTensorDescriptor(weight, 'w', key.weight_alignment))
      .setcDesc(getConvDescriptor(key.params.dataType, padding, stride, dilation))
      .build();
  // std::cout << "operator:" << conv_op.describe() << std::endl;
  // TODO: add support for bias

  auto requant_op = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
    .setxDesc(conv_op.getOutputTensor())
    .setbDesc(getTensorDescriptor(requantize_multiplier_tensor, 's', getAlignment(requantize_multiplier_tensor)))
    .setyDesc(getTensorDescriptor(output, 'r', getAlignment(output)))
    .setpwDesc(getPointWiseMulDescriptor(getCudnnDataType(output)))
    .build();
  // std::cout << "operator:" << requant_op.describe() << std::endl;

  std::array<cudnn_frontend::Operation const *, 2> ops = {&conv_op, &requant_op};

  auto opGraph = cudnn_frontend::OperationGraphBuilder()
      .setHandle(handle)
      .setOperationGraph(ops.size(), ops.data())
      .build();
  // std::cout << "opGraph: " << opGraph.describe() << std::endl;

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

  cudnn_frontend::EngineConfigList filtered_configs;
  filterEngineConfigs(engine_configs, filtered_configs, deterministic, allow_tf32, input.scalar_type());
  filterEngineConfigs(fallback_list, filtered_configs, deterministic, allow_tf32, input.scalar_type());

  for (auto &cfg : engine_configs) {
    try {
      run(cfg);
      engine_cache[key] = cfg;
      return;
    } catch (cudnn_frontend::cudnnException &e) {std::cout << "cudnn error:" << e.what() << std::endl;} catch(CuDNNError &e) { std::cout << "other error" << e.what() << std::endl;}
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
    bool benchmark,
    bool deterministic,
    bool allow_tf32,
    float requantize_multiplier) {
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
      at::MemoryFormat::ChannelsLast
  );

  raw_cudnn_convolution_forward_out(
      output_fp32, act, weight,
      padding, stride, dilation, groups,
      benchmark,
      deterministic,
      allow_tf32,
      requantize_multiplier);
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
    act = act.contiguous(c10::MemoryFormat::ChannelsLast);
    weight = weight.contiguous(c10::MemoryFormat::ChannelsLast);
    // requantization
    // out_int8 = act_int8 * weight_int8 * act_scale * w_scale / output_scale
    auto act_scale = act.q_scale();
    auto weight_scale = weight.q_scale();
    auto requantize_multiplier = act_scale * weight_scale / output_scale;
    //auto requantize_multiplier = output_scale / (act_scale * weight_scale);
    // TODO: check all zero_points are zero/all tensors are symmetrically quantized
    Tensor output_fp32_requant = raw_cudnn_convolution_forward<kSpatialDim>(
        act.int_repr(), weight.int_repr(),
        IntArrayRef(padding.vec()), IntArrayRef(stride.vec()), IntArrayRef(dilation.vec()), groups,
        false /* benchmark */,
        true /* deterministic */,
        false /* allow_tf32 */,
        requantize_multiplier
    );

    // convert output fp32 Tensor to int8 by clamping
    // TODO: get the range based on target output dtype, for now
    // we hardcode this to the range for int8
    Tensor clampped = output_fp32_requant.clamp(-128, 127);
    Tensor quantized_output = at::_make_per_tensor_quantized_tensor(clampped.to(at::kChar), output_scale, output_zero_point);
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
