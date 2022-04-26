#ifdef USE_CUDA
#include <ATen/cuda/CUDAConfig.h>  // for the definition of AT_CUDNN_ENABLED

#if AT_CUDNN_ENABLED()

#include <ATen/native/cudnn/Macros.h>
#include <c10/util/ArrayRef.h>

#if HAS_CUDNN_V8()

#include <ATen/ATen.h>
#include <ATen/cuda/Exceptions.h>
#include <ATen/cudnn/Handle.h>
#include <ATen/native/ConvUtils.h>
#include <ATen/native/cudnn/ConvShared.h>
#include <ATen/native/quantized/cudnn/utils.h>
#include <ATen/native/quantized/packed_params.h>
#include <ATen/native/utils/ParamsHash.h>
#include <ATen/TensorUtils.h>
#include <cudnn_frontend.h>
#include <torch/library.h>

#include <iostream>
#include <unordered_map>
#include <vector>

// TODO: there is a table from input dtype and weight dtype to operator qdtype,
// we can derive the operator dtype based on input dtype
cudnn_frontend::ConvDesc_v8 getConvDescriptor(cudnnDataType_t dataType, c10::IntArrayRef padding, c10::IntArrayRef stride, c10::IntArrayRef dilation) {
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

// FIXME: make this thread-safe by reusing the benchmark cache in Conv_v7.cpp
namespace {
struct CacheKey {
  at::native::ConvolutionParams params;
  uint8_t input_alignment;
  uint8_t weight_alignment;
  uint8_t output_alignment;
  // default to -1 when no bias
  int8_t bias_alignment;
  bool kReluFused;
};
std::unordered_map<CacheKey, cudnn_frontend::ManagedOpaqueDescriptor, at::native::ParamsHash<CacheKey>, at::native::ParamsEqual<CacheKey>> execution_plan_cache;
}
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
    const torch::List<int64_t>& stride,
    const torch::List<int64_t>& padding,
    const torch::List<int64_t>& dilation);

template <>
at::SmallVector<int64_t, 4> MakeConvOutputShape<2>(
    int N, // mini-batch
    int M, // output channels
    const std::array<int64_t, 2>& input_image_shape,
    const std::vector<int64_t>& kernel,
    const torch::List<int64_t>& stride,
    const torch::List<int64_t>& padding,
    const torch::List<int64_t>& dilation) {
  const int H = input_image_shape[0];
  const int W = input_image_shape[1];
  const int64_t Y_H =
      (H + 2 * padding[0] - dilation[0] * (kernel[0] - 1) - 1) / stride[0] + 1;
  const int64_t Y_W =
      (W + 2 * padding[1] - dilation[1] * (kernel[1] - 1) - 1) / stride[1] + 1;
  return {N, M, Y_H, Y_W};
}


// the parameter quantized_output is a quantized tensor
template <int kSpatialDim>
template <bool kReluFused>
void PackedConvWeightCudnn<kSpatialDim>::apply_impl_helper(const at::Tensor& quantized_output, const at::Tensor& input, double output_scale) {
  if (quantized_output.numel() == 0) {
    return;
  }
  at::Tensor conv_output = at::empty(quantized_output.sizes(), at::device(at::kCUDA).dtype(at::kFloat), at::MemoryFormat::ChannelsLast);
  // TODO: combine empty & fill_ using full_like or full
  at::Tensor requantize_multiplier_tensor = at::empty(quantized_output.sizes(), at::device(at::kCUDA).dtype(at::kFloat), at::MemoryFormat::ChannelsLast);
  auto act_scale = input.q_scale();
  auto weight_scale = orig_weight_.q_scale();
  auto requantize_multiplier = act_scale * weight_scale / output_scale;
  requantize_multiplier_tensor.fill_(requantize_multiplier);
  c10::optional<at::Tensor> bias_multiplier_tensor;
  c10::optional<at::Tensor> broadcasted_bias;
  if (bias_.has_value()) {
    // the input bias is a 1-D tensor whose size is the same as the size of the second dimension of quantized_output.
    // we need to add trailing dimensions in order to properly broadcast bias, otherwise broadcast_to will fail.
    // the number of trailling dimensions is quantized_output.dim() - 2, so the new size of the broadcast_bias
    // becomes quantized_output.dim() - 2 + 1. nothing needs to be done for the leading dimensions
    std::vector<int64_t> new_size(quantized_output.dim() - 1, 1);
    new_size[0] = bias_.value().size(0);
    broadcasted_bias = bias_.value().reshape(new_size);
    broadcasted_bias.value() = broadcasted_bias.value().broadcast_to(quantized_output.sizes());
    broadcasted_bias.value() = broadcasted_bias.value().to(c10::MemoryFormat::ChannelsLast);
    bias_multiplier_tensor = at::empty(quantized_output.sizes(), at::device(at::kCUDA).dtype(at::kFloat), at::MemoryFormat::ChannelsLast);
    auto bias_multiplier = 1.0 / (act_scale * weight_scale);
    bias_multiplier_tensor.value().fill_(bias_multiplier);
  }

  cudnnHandle_t handle = at::native::getCudnnHandle();
  CacheKey key;
  bool deterministic{true};
  bool allow_tf32{false};
  auto padding_vec = padding_.vec();
  auto stride_vec = stride_.vec();
  auto dilation_vec = dilation_.vec();
  setConvolutionParams(&key.params, input, orig_weight_, padding_vec, stride_vec, dilation_vec, groups_, deterministic, allow_tf32);

  // operator datatype needs to be int32 for int8 convolution, but we can
  // set the datatype for output tensor to int32 or fp32
  key.params.dataType = CUDNN_DATA_INT32;
  key.input_alignment = cudnn_utils::getAlignment(input);
  key.output_alignment = cudnn_utils::getAlignment(conv_output);
  key.weight_alignment = cudnn_utils::getAlignment(orig_weight_);
  if (bias_.has_value()) {
    key.bias_alignment = cudnn_utils::getAlignment(broadcasted_bias.value());
  } else {
    key.bias_alignment = -1;
  }
  key.kReluFused = kReluFused;

  auto run = [&](cudnn_frontend::ManagedOpaqueDescriptor plan_desc) {
    auto workspace_size = 0;
    auto workspace = at::empty({workspace_size}, input.options().dtype(at::kByte));
    std::vector<void *> data_ptrs;
    std::vector<int64_t> uids;
    data_ptrs.reserve(10);
    uids.reserve(10);
    data_ptrs = {reinterpret_cast<int8_t*>(input.data_ptr()), conv_output.data_ptr(),
                                           reinterpret_cast<int8_t*>(orig_weight_.data_ptr()),
                                           requantize_multiplier_tensor.data_ptr(),
                                           reinterpret_cast<int8_t*>(quantized_output.data_ptr())};
    uids = {'x', 'y', 'w', 's', 'r'};
    if (bias_.has_value()) {
      data_ptrs.insert(data_ptrs.end(), {broadcasted_bias.value().data_ptr(), bias_multiplier_tensor.value().data_ptr(),
                                         broadcasted_bias.value().data_ptr(), conv_output.data_ptr()});
      uids.insert(uids.end(), {'b', 'c', 'd', 'e'});
      if (kReluFused) {
        data_ptrs.emplace_back(conv_output.data_ptr()),
        uids.emplace_back('f');
      }
    } else {
      if (kReluFused) {
        data_ptrs.emplace_back(conv_output.data_ptr());
        uids.emplace_back('f');
      }
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
  // conv_op computes act_fp32 * w_fp32 (matrix multiplication)
  // where act_fp32 and w_fp32 are the input and weight variables, resp.
  // output is a fp32 tensor
  auto conv_op = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR)
      .setxDesc(cudnn_utils::getTensorDescriptor(input.sizes(), input.strides(), CUDNN_DATA_INT8, 'x', key.input_alignment))
      .setyDesc(cudnn_utils::getTensorDescriptor(conv_output, 'y', key.output_alignment))
      .setwDesc(cudnn_utils::getTensorDescriptor(orig_weight_.sizes(), orig_weight_.strides(), CUDNN_DATA_INT8, 'w', key.weight_alignment))
      .setcDesc(getConvDescriptor(key.params.dataType, padding_vec, stride_vec, dilation_vec))
      .build();
  // std::cout << "operator:" << conv_op.describe() << std::endl;

  c10::optional<cudnn_frontend::Operation> bias_mult_op;
  c10::optional<cudnn_frontend::Operation> sum_conv_bias_op;
  if (bias_.has_value()) {
    // we can't directly assign bias_mult_op becauase operator= is deleted for cudnn_frontend::Operation;
    // alternatively, I think we can use std::unique_ptr and dynamically allocate these builder ops
    // but here, we chose to do it statically. c10::optional<T>::emplace() enables this approach

    // bias_mult_op computes bias_fp32 / (act_scale * w_scale) or bias_fp32 * (1 / (act_scale * w_scale))
    // where bias_multiplier = (1 / (act_scale * w_scale))
    // output is a fp32 tensor
    // we use inplace operation here where the output is assigned to the input
    bias_mult_op.emplace(cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
      .setxDesc(cudnn_utils::getTensorDescriptor(broadcasted_bias.value(), 'b', cudnn_utils::getAlignment(broadcasted_bias.value())))
      .setbDesc(cudnn_utils::getTensorDescriptor(bias_multiplier_tensor.value(), 'c', cudnn_utils::getAlignment(bias_multiplier_tensor.value())))
      .setyDesc(cudnn_utils::getTensorDescriptor(broadcasted_bias.value(), 'd', cudnn_utils::getAlignment(broadcasted_bias.value())))
      .setpwDesc(cudnn_utils::getPointWiseMulDescriptor(at::native::getCudnnDataType(bias_multiplier_tensor.value())))
      .build());

    // computes (act_int8 * w_int8 + [bias_fp32/(act_scale * w_scale)])
    // where the 1st and 2nd summands is conv_output and broadcasted_bias, resp.
    // output is a fp32 tensor
    // we use inplace operation here where the output is assigned to the input
    sum_conv_bias_op.emplace(cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
      .setxDesc(conv_op.getOutputTensor())
      .setbDesc(cudnn_utils::getTensorDescriptor(broadcasted_bias.value(), 'd', cudnn_utils::getAlignment(broadcasted_bias.value())))
      .setyDesc(cudnn_utils::getTensorDescriptor(conv_output, 'e', key.output_alignment))
      .setpwDesc(cudnn_utils::getPointWiseAddDescriptor(at::native::getCudnnDataType(broadcasted_bias.value())))
      .build());
  }

  // relu_op computes relu(act_int8 * w_int8 + [bias_fp32/(act_scale * w_scale)]
  // or relu(act_int8 * w_int8) if bias is not present.
  // output is a fp32 tensor
  c10::optional<cudnn_frontend::Operation> relu_op;
  std::shared_ptr<cudnn_frontend::OpaqueBackendPointer> tensor2requant_ptr = bias_.has_value() ? sum_conv_bias_op.value().getOutputTensor() : conv_op.getOutputTensor();
  if (kReluFused) {
    // we use inplace operation here where the output is assigned to the input
    relu_op.emplace(cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
      .setxDesc(tensor2requant_ptr)
      .setyDesc(cudnn_utils::getTensorDescriptor(conv_output, 'f', key.output_alignment))
      .setpwDesc(cudnn_utils::getPointWiseReluDescriptor(at::native::getCudnnDataType(conv_output)))
      .build());
  }

  // relu_op computes relu(act_int8 * w_int8 + [bias_fp32/(act_scale * w_scale)]) / (out_scale / (act_scale * w_scale))
  // or relu(act_int8 * w_int8) / (out_scale / (act_scale * w_scale))) if bias is not present.
  // output is a fp32 tensor
  auto requant_op = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
    .setxDesc(kReluFused ? relu_op.value().getOutputTensor() : tensor2requant_ptr)
    .setbDesc(cudnn_utils::getTensorDescriptor(requantize_multiplier_tensor, 's', cudnn_utils::getAlignment(requantize_multiplier_tensor)))
    .setyDesc(cudnn_utils::getTensorDescriptor(quantized_output.sizes(), quantized_output.strides(), CUDNN_DATA_INT8, 'r', cudnn_utils::getAlignment(quantized_output)))
    .setpwDesc(cudnn_utils::getPointWiseMulDescriptor(at::native::getCudnnDataType(requantize_multiplier_tensor)))
    .build();
  // std::cout << "operator:" << requant_op.describe() << std::endl;

  std::vector<cudnn_frontend::Operation const *> ops{&conv_op};
  if (bias_.has_value()) {
    ops.emplace_back(&(bias_mult_op.value()));
    ops.emplace_back(&(sum_conv_bias_op.value()));
  }
  if (kReluFused) {
    ops.emplace_back(&(relu_op.value()));
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
  cudnn_utils::filterEngineConfigs(engine_configs, filtered_configs, deterministic, allow_tf32, at::kChar);
  cudnn_utils::filterEngineConfigs(fallback_list, filtered_configs, deterministic, allow_tf32, at::kChar);

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
    } catch (cudnn_frontend::cudnnException &e) {std::cout << "cudnn error:" << e.what() << std::endl;} catch(c10::CuDNNError &e) { std::cout << "other error" << e.what() << std::endl;}
  }

  TORCH_CHECK(false, "Unable to find an engine to execute this computation");
}

//
// output Tensor will be a clampped int8 Tensor
// both act and weight will be int8 Tensor
/*
Numerics:
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
template <bool kReluFused>
at::Tensor PackedConvWeightCudnn<kSpatialDim>::apply_impl(
    const at::Tensor& act,
    double output_scale,
    int64_t output_zero_point) {
  const int N = act.size(0);
  const int D = kSpatialDim == 3 ? act.size(2) : 1;
  const int H = act.size(kSpatialDim);
  const int W = act.size(kSpatialDim + 1);
  const int M = orig_weight_.size(0); // output channels
  std::vector<int64_t> kernel_size = {orig_weight_.size(2), orig_weight_.size(3)};
  at::SmallVector<int64_t, kSpatialDim + 2> output_shape = MakeConvOutputShape<kSpatialDim>(N, M, {H, W},
  kernel_size, stride_, padding_, dilation_);
  at::Tensor quantized_output = at::_empty_affine_quantized(
      output_shape,
      at::device(at::kCUDA).dtype(at::ScalarType::QInt8),
      output_scale,
      output_zero_point,
      at::MemoryFormat::ChannelsLast);
  // requantization
  // out_int8 = act_int8 * weight_int8 * act_scale * w_scale / output_scale
  apply_impl_helper<kReluFused>(
      quantized_output, act, output_scale);
  return quantized_output;
}

template <int kSpatialDim>
at::Tensor PackedConvWeightCudnn<kSpatialDim>::apply(
    const at::Tensor& input,
    double output_scale,
    int64_t output_zero_point) {
  return apply_impl<false>(input, output_scale, output_zero_point);
}

template <int kSpatialDim>
at::Tensor PackedConvWeightCudnn<kSpatialDim>::apply_relu(
    const at::Tensor& input,
    double output_scale,
    int64_t output_zero_point) {
  return apply_impl<true>(input, output_scale, output_zero_point);
}

template at::Tensor PackedConvWeightCudnn<2>::apply(
    const at::Tensor& act,
    double output_scale,
    int64_t output_zero_point);

template at::Tensor PackedConvWeightCudnn<2>::apply_relu(
    const at::Tensor& act,
    double output_scale,
    int64_t output_zero_point);

namespace at {
namespace native {
namespace {

template <int kSpatialDim, bool kReluFused>
class QConvInt8 final {
 public:
  static at::Tensor run(
      at::Tensor act,
      const c10::intrusive_ptr<ConvPackedParamsBase<kSpatialDim>>& packed_weight,
      double output_scale,
      int64_t output_zero_point) {
    act = act.to(c10::MemoryFormat::ChannelsLast);
    // TODO: check all zero_points are zero/all tensors are symmetrically quantized
    if (kReluFused) {
      return packed_weight->apply_relu(act, output_scale, output_zero_point);
    } else {
      return packed_weight->apply(act, output_scale, output_zero_point);
    }
  }
};

TORCH_LIBRARY_IMPL(quantized, QuantizedCUDA, m) {
  m.impl(TORCH_SELECTIVE_NAME("quantized::conv2d.new"), QConvInt8<2, false>::run);
  m.impl(TORCH_SELECTIVE_NAME("quantized::conv2d_relu.new"), QConvInt8<2, true>::run);
}

} // namespace
} // namespace native
} // namespace at


#endif  // HAS_CUDNN_V8
#endif  // AT_CUDNN_ENABLED
#endif  // USE_CUDA
