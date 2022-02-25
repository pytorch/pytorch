#ifdef USE_CUDA
#include <ATen/cuda/CUDAConfig.h>  // for the definition of AT_CUDNN_ENABLED

#if AT_CUDNN_ENABLED()

#include <ATen/native/cudnn/Macros.h>
#include <c10/util/ArrayRef.h>

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

cudnn_frontend::Tensor getTensorDescriptor(const IntArrayRef& shape, const IntArrayRef& strides, cudnnDataType_t cudnn_dtype, int64_t id, uint8_t alignment) {
  return cudnn_frontend::TensorBuilder()
    .setDim(shape.size(), shape.data())
    .setStrides(strides.size(), strides.data())
    .setId(id)
    .setAlignment(alignment)
    .setDataType(cudnn_dtype)
    .build();
}

// TODO: there is a table from input dtype and weight dtype to operator dtype,
// we can derive the operator dtype based on input dtype
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

// TODO: there is a table from input dtype to operator dtype, we can derive
// the operator dtype based on input dtype
cudnn_frontend::PointWiseDesc_v8 getPointWiseMulDescriptor(cudnnDataType_t dataType) {
  return cudnn_frontend::PointWiseDescBuilder()
    .setMode(cudnnPointwiseMode_t::CUDNN_POINTWISE_MUL)
    .setMathPrecision(dataType)
    .build();
}

// TODO: there is a table from input dtype to operator dtype, we can derive
// the operator dtype based on input dtype
cudnn_frontend::PointWiseDesc_v8 getPointWiseAddDescriptor(cudnnDataType_t dataType) {
  return cudnn_frontend::PointWiseDescBuilder()
    .setMode(cudnnPointwiseMode_t::CUDNN_POINTWISE_ADD)
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
  // default to -1 when no bias
  int8_t bias_alignment;
};

// FIXME: make this thread-safe by reusing the benchmark cache in Conv_v7.cpp
std::unordered_map<CacheKey, cudnn_frontend::ManagedOpaqueDescriptor, ParamsHash<CacheKey>, ParamsEqual<CacheKey>> execution_plan_cache;

// TODO: we can use cudnn_frontend::ExecutionPlanCache when it supports caching
// multiple operators
// reference: https://github.com/NVIDIA/cudnn-frontend/blob/main/samples/conv_sample.cpp#L293
//static cudnn_frontend::ExecutionPlanCache plan_cache("sample_cache");

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

// the parameter quantized_output is a quantized tensor
void raw_cudnn_convolution_forward_out(
    const Tensor& quantized_output,
    const Tensor& input,
    const Tensor& weight,
    const c10::optional<Tensor> &bias,
    IntArrayRef padding,
    IntArrayRef stride,
    IntArrayRef dilation,
    int64_t groups,
    bool benchmark,
    bool deterministic,
    bool allow_tf32,
    float bias_multiplier,
    float requantize_multiplier
) {
  TORCH_CHECK(!benchmark, "not supported yet");
  if (quantized_output.numel() == 0) {
    return;
  }

  Tensor conv_output = at::empty(quantized_output.sizes(), at::device(at::kCUDA).dtype(at::kFloat), at::MemoryFormat::ChannelsLast);
  // TODO: combine empty & fill_ using full_like or full
  Tensor requantize_multiplier_tensor = at::empty(quantized_output.sizes(), at::device(at::kCUDA).dtype(at::kFloat), at::MemoryFormat::ChannelsLast);
  requantize_multiplier_tensor.fill_(requantize_multiplier);
  c10::optional<at::Tensor> bias_multiplier_tensor;
  c10::optional<at::Tensor> after_scales_bias;
  c10::optional<at::Tensor> after_add;
  c10::optional<at::Tensor> broadcasted_bias;
  if (bias.has_value()) {
    // the input bias is a 1-D tensor whose size is the same as the size of the second dimension of quantized_output.
    // we need to add trailing dimensions in order to properly broadcast bias, otherwise broadcast_to will fail.
    // the number of trailling dimensions is quantized_output.dim() - 2, so the new size of the broadcast_bias
    // becomes quantized_output.dim() - 2 + 1. nothing needs to be done for the leading dimensions
    std::vector<int64_t> new_size(quantized_output.dim() - 1, 1);
    new_size[0] = bias.value().size(0);
    broadcasted_bias = bias.value().reshape(new_size);
    broadcasted_bias.value() = broadcasted_bias.value().broadcast_to(quantized_output.sizes());
    broadcasted_bias.value() = broadcasted_bias.value().contiguous(c10::MemoryFormat::ChannelsLast);
    bias_multiplier_tensor = at::empty(quantized_output.sizes(), at::device(at::kCUDA).dtype(at::kFloat), at::MemoryFormat::ChannelsLast);
    bias_multiplier_tensor.value().fill_(bias_multiplier);
    after_scales_bias = at::empty(quantized_output.sizes(), at::device(at::kCUDA).dtype(at::kFloat), at::MemoryFormat::ChannelsLast);
    after_add = at::empty(quantized_output.sizes(), at::device(at::kCUDA).dtype(at::kFloat), at::MemoryFormat::ChannelsLast);
  }

  cudnnHandle_t handle = getCudnnHandle();
  CacheKey key;
  setConvolutionParams(&key.params, input, weight, padding, stride, dilation, groups, deterministic, allow_tf32);
  // operator datatype needs to be int32 for int8 convolution, but we can
  // set the datatype for output tensor to int32 or fp32
  key.params.dataType = CUDNN_DATA_INT32;
  key.input_alignment = getAlignment(input);
  key.output_alignment = getAlignment(conv_output);
  key.weight_alignment = getAlignment(weight);
  if (bias.has_value()) {
    key.bias_alignment = getAlignment(broadcasted_bias.value());
  } else {
    key.bias_alignment = -1;
  }

  auto run = [&](cudnn_frontend::ManagedOpaqueDescriptor plan_desc) {
    auto workspace_size = 0;
    auto workspace = at::empty({workspace_size}, input.options().dtype(kByte));
    std::vector<void *> data_ptrs;
    std::vector<int64_t> uids;
    data_ptrs.reserve(9);
    uids.reserve(9);
    data_ptrs = {reinterpret_cast<int8_t*>(input.data_ptr()), conv_output.data_ptr(),
                                           reinterpret_cast<int8_t*>(weight.data_ptr()),
                                           requantize_multiplier_tensor.data_ptr(),
                                           reinterpret_cast<int8_t*>(quantized_output.data_ptr())};
    uids = {'x', 'y', 'w', 's', 'r'};
    if (bias.has_value()) {
      void *ptrs2append[] = {broadcasted_bias.value().data_ptr(),
                             bias_multiplier_tensor.value().data_ptr(),
                             after_scales_bias.value().data_ptr(),
                             after_add.value().data_ptr()};
      int64_t ids2append[] = {'b', 'c', 'd', 'e'};
      data_ptrs.insert(data_ptrs.end(), std::begin(ptrs2append), std::end(ptrs2append));
      uids.insert(uids.end(), std::begin(ids2append), std::end(ids2append));
    }
    auto variantPack = cudnn_frontend::VariantPackBuilder()
      .setWorkspacePointer(workspace.data_ptr())
      .setDataPointers(uids.size(), data_ptrs.data())
      .setUids(uids.size(), uids.data())
      .build();
    auto variant_pack_desc = variantPack.get_raw_desc();
    AT_CUDNN_CHECK(cudnnBackendExecute(handle, plan_desc->get_backend_descriptor(), variant_pack_desc));
  };

  auto search = execution_plan_cache.find(key);
  if (search != execution_plan_cache.end()) {
    cudnn_frontend::ManagedOpaqueDescriptor plan_desc = search->second;
    run(plan_desc);
    return;
  }
  auto conv_op = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR)
      .setxDesc(getTensorDescriptor(input, 'x', key.input_alignment))
      .setyDesc(getTensorDescriptor(conv_output, 'y', key.output_alignment))
      .setwDesc(getTensorDescriptor(weight, 'w', key.weight_alignment))
      .setcDesc(getConvDescriptor(key.params.dataType, padding, stride, dilation))
      .build();
  // std::cout << "operator:" << conv_op.describe() << std::endl;

  c10::optional<cudnn_frontend::Operation> bias_mult_op;
  c10::optional<cudnn_frontend::Operation> sum_conv_bias_op;
  if (bias.has_value()) {
    // we can't directly assign bias_mult_op becauase operator= is deleted for cudnn_frontend::Operation;
    // alternatively, I think we can use std::unique_ptr and dynamically allocate these builder ops
    // but here, we chose to do it statically. c10::optional<T>::emplace() enables this approach
    // TODO: can we assign the result back into bias and get rid of after_scales_bias? pending NVIDIA response
    bias_mult_op.emplace(cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
      .setxDesc(getTensorDescriptor(broadcasted_bias.value(), 'b', getAlignment(broadcasted_bias.value())))
      .setbDesc(getTensorDescriptor(bias_multiplier_tensor.value(), 'c', getAlignment(bias_multiplier_tensor.value())))
      .setyDesc(getTensorDescriptor(after_scales_bias.value(), 'd', getAlignment(after_scales_bias.value())))
      .setpwDesc(getPointWiseMulDescriptor(getCudnnDataType(bias_multiplier_tensor.value())))
      .build());

    // TODO: can we assign the result back into conv_output and get rid of after_add?
    sum_conv_bias_op.emplace(cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
      .setxDesc(conv_op.getOutputTensor())
      .setbDesc(getTensorDescriptor(after_scales_bias.value(), 'd', getAlignment(after_scales_bias.value())))
      .setyDesc(getTensorDescriptor(after_add.value(), 'e', getAlignment(after_add.value())))
      .setpwDesc(getPointWiseAddDescriptor(getCudnnDataType(after_scales_bias.value())))
      .build());
  }

  auto requant_op = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
    .setxDesc(bias.has_value() ? sum_conv_bias_op.value().getOutputTensor() : conv_op.getOutputTensor())
    .setbDesc(getTensorDescriptor(requantize_multiplier_tensor, 's', getAlignment(requantize_multiplier_tensor)))
    .setyDesc(getTensorDescriptor(quantized_output.sizes(), quantized_output.strides(), CUDNN_DATA_INT8, 'r', getAlignment(quantized_output)))
    .setpwDesc(getPointWiseMulDescriptor(getCudnnDataType(requantize_multiplier_tensor)))
    .build();
  // std::cout << "operator:" << requant_op.describe() << std::endl;

  std::vector<cudnn_frontend::Operation const *> ops{&conv_op};
  if (bias.has_value()) {
    ops.emplace_back(&(bias_mult_op.value()));
    ops.emplace_back(&(sum_conv_bias_op.value()));
  }
  ops.emplace_back(&requant_op);

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
      auto plan = cudnn_frontend::ExecutionPlanBuilder()
        .setHandle(handle)
        .setEngineConfig(cfg)
        .build();
      auto plan_desc = plan.get_desc();
      run(plan_desc);
      execution_plan_cache[key] = plan_desc;
      return;
    } catch (cudnn_frontend::cudnnException &e) {std::cout << "cudnn error:" << e.what() << std::endl;} catch(CuDNNError &e) { std::cout << "other error" << e.what() << std::endl;}
  }

  TORCH_CHECK(false, "Unable to find an engine to execute this computation");
}

//
// output Tensor will be a clampped int8 Tensor
// both act and weight will be int8 Tensor
/*
Numerics:
Operation:
out_fp32 = conv_fp32(act_fp32, w_fp32, …)
                    = act_fp32 * w_fp32 + bias_fp32
act_int8 = act_fp32 / act_scale + act_zero_point
w_int8 = w_fp32 / w_scale + w_zero_point
out_int8 = out_fp32 / out_scale + out_zero_point
out_int8 = (act_fp32 * w_fp32 + [bias_fp32]) / out_scale + out_zero_point
              = (act_int8 - act_zero_point) * act_scale * (w_int8 - w_zero_point) * w_scale / out_scale + out_zero_point + [bias_fp32 / out_scale]
             = (act_int8 * w_int8 - act_int8 * w_zero_point - act_zero_point * w_int8 + act_zero_point * w_zero_point) * act_scale * w_scale / out_scale + out_zero_point + [bias_fp32 / out_scale]
             = (if both act and weight are symmetrically quantized, int8, then act_zero_point = w_zero_point = 0)
             = (act_int8 * w_int8 + [bias_fp32/(act_scale * w_scale)]) * act_scale * w_scale / out_scale
             = (act_int8 * w_int8 + [bias_fp32/(act_scale * w_scale)]) / (out_scale / (act_scale * w_scale))
             = requantize((act_int8 * w_int8 + [bias_fp32/(act_scale * w_scale)]), out_scale / (act_scale * w_scale))
*/
template <int kSpatialDim>
Tensor raw_cudnn_convolution_forward(
    const Tensor& act,
    const Tensor& weight,
    c10::optional<Tensor> bias,
    IntArrayRef padding,
    IntArrayRef stride,
    IntArrayRef dilation,
    int64_t groups,
    bool benchmark,
    bool deterministic,
    bool allow_tf32,
    float bias_multiplier,
    float requantize_multiplier,
    double output_scale,
    int64_t output_zero_point) {
  // TODO: add dimension validations for input/weight/bias
  const int N = act.size(0);
  const int D = kSpatialDim == 3 ? act.size(2) : 1;
  const int H = act.size(kSpatialDim);
  const int W = act.size(kSpatialDim + 1);
  const int M = weight.size(0); // output channels
  std::vector<int64_t> kernel_size = {weight.size(2), weight.size(3)};
  at::SmallVector<int64_t, kSpatialDim + 2> output_shape{MakeConvOutputShape<kSpatialDim>(N, M, {H, W},
  kernel_size, stride, padding, dilation)};
  Tensor quantized_output = at::_empty_affine_quantized(
      output_shape,
      at::device(at::kCUDA).dtype(ScalarType::QInt8),
      output_scale,
      output_zero_point,
      at::MemoryFormat::ChannelsLast);
  raw_cudnn_convolution_forward_out(
      quantized_output, act, weight, bias,
      padding, stride, dilation, groups,
      benchmark,
      deterministic,
      allow_tf32,
      bias_multiplier,
      requantize_multiplier);

  return quantized_output;
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
    act = act.contiguous(c10::MemoryFormat::ChannelsLast);
    weight = weight.contiguous(c10::MemoryFormat::ChannelsLast);
    // requantization
    // out_int8 = act_int8 * weight_int8 * act_scale * w_scale / output_scale
    auto act_scale = act.q_scale();
    auto weight_scale = weight.q_scale();
    auto requantize_multiplier = act_scale * weight_scale / output_scale;
    auto bias_multiplier = 1.0 / (act_scale * weight_scale);

    // TODO: check all zero_points are zero/all tensors are symmetrically quantized
    return raw_cudnn_convolution_forward<kSpatialDim>(
        act.int_repr(), weight.int_repr(), bias,
        IntArrayRef(padding.vec()), IntArrayRef(stride.vec()), IntArrayRef(dilation.vec()), groups,
        false /* benchmark */,
        true /* deterministic */,
        false /* allow_tf32 */,
        bias_multiplier,
        requantize_multiplier,
        output_scale,
        output_zero_point
    );
  }
};

TORCH_LIBRARY_IMPL(quantized, QuantizedCUDA, m) {
  m.impl(TORCH_SELECTIVE_NAME("quantized::conv2d_cudnn"), QConvInt8<2, false>::run);
}

} // namespace
}} // at::native

#endif  // HAS_CUDNN_V8
#endif  // AT_CUDNN_ENABLED
#endif  // USE_CUDA
