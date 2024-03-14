#ifdef USE_CUDA
#include <ATen/cuda/CUDAConfig.h>  // for the definition of AT_CUDNN_ENABLED

#if AT_CUDNN_ENABLED()

#include <c10/util/ArrayRef.h>

#include <ATen/ATen.h>
#include <ATen/cuda/Exceptions.h>
#include <ATen/cudnn/Handle.h>
#include <ATen/native/cudnn/ConvShared.h>
#include <ATen/native/quantized/cudnn/utils.h>
#include <ATen/native/quantized/ConvUtils.h>
#include <ATen/native/quantized/PackedParams.h>
#include <ATen/native/utils/ParamsHash.h>
#include <ATen/TensorUtils.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <cudnn_frontend.h>
#include <torch/library.h>

#include <iostream>
#include <unordered_map>
#include <vector>

template <int kSpatialDim = 2>
int register_conv_params();

extern template int register_conv_params<2>();
extern template int register_conv_params<3>();

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
std::unordered_map<CacheKey, cudnn_frontend::ExecutionPlan, at::native::ParamsHash<CacheKey>, at::native::ParamsEqual<CacheKey>> execution_plan_cache;
} // anonymous namespace
// TODO: we can use cudnn_frontend::ExecutionPlanCache when it supports caching
// multiple operators
// reference: https://github.com/NVIDIA/cudnn-frontend/blob/main/samples/conv_sample.cpp#L293
//static cudnn_frontend::ExecutionPlanCache plan_cache("sample_cache");

// the parameter quantized_output is a quantized tensor
template <int kSpatialDim>
template <bool kReluFused>
void PackedConvWeightCudnn<kSpatialDim>::apply_impl_helper(const at::Tensor& quantized_output, const at::Tensor& input, double output_scale) {
  auto act_scale = input.q_scale();
  auto weight_scale = maybe_padded_weight_.q_scale();
  auto requantize_multiplier = act_scale * weight_scale / output_scale;
  at::Tensor requantize_multiplier_tensor = cudnn_utils::getRequantMultiplierTensor(requantize_multiplier, kSpatialDim + 2);

  c10::optional<at::Tensor> bias_multiplier_tensor;
  c10::optional<at::Tensor> broadcasted_bias;
  if (bias_.has_value()) {
    // the input bias is a 1-D tensor whose size is the same as the size of the second dimension of quantized_output.
    // we need to add trailing dimensions in order to properly broadcast bias, otherwise broadcast_to will fail.
    // the number of trailing dimensions is quantized_output.dim() - 2, so the new size of the broadcast_bias
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
  // memset is needed here because there is implicit packing added for CacheKey, and this can result in uninitialized padded values that are
  // used for hashing (see how at::native::ParamsHash is defined). without memset, we can potentially come across a situation where two
  // CacheKey objects have the same user defined parameters, but
  // different padded values, resulting in different hash outputs.
  memset(&key, 0, sizeof(key));
  bool deterministic{true};
  bool allow_tf32{false};
  auto padding_vec = padding_.vec();
  auto stride_vec = stride_.vec();
  auto dilation_vec = dilation_.vec();
  setConvolutionParams(&key.params, input, maybe_padded_weight_, padding_vec, stride_vec, dilation_vec, groups_, deterministic, allow_tf32, input.suggest_memory_format());

  // operator datatype needs to be int32 for int8 convolution, but we can
  // set the datatype for output tensor to int32 or fp32
  key.params.dataType = CUDNN_DATA_INT32;
  key.input_alignment = cudnn_utils::getAlignment(input);
  key.output_alignment = cudnn_utils::getAlignment(quantized_output);
  key.weight_alignment = cudnn_utils::getAlignment(maybe_padded_weight_);
  if (bias_.has_value()) {
    key.bias_alignment = cudnn_utils::getAlignment(broadcasted_bias.value());
  } else {
    key.bias_alignment = -1;
  }
  key.kReluFused = kReluFused;

  auto run = [&](const cudnn_frontend::ExecutionPlan& plan_desc) {
    auto workspace_size = plan_desc.getWorkspaceSize();
    auto workspace_ptr = c10::cuda::CUDACachingAllocator::get()->allocate(workspace_size);
    at::SmallVector<void *, 7> data_ptrs;
    at::SmallVector<int64_t, 7> uids;
    data_ptrs = {input.data_ptr<int8_t>(), maybe_padded_weight_.data_ptr<int8_t>(),
                 requantize_multiplier_tensor.data_ptr(), quantized_output.data_ptr<int8_t>()};
    uids = {'x', 'w', 's', 'r'};
    if (bias_.has_value()) {
      data_ptrs.insert(data_ptrs.end(), {broadcasted_bias.value().data_ptr(), bias_multiplier_tensor.value().data_ptr(),
                                         broadcasted_bias.value().data_ptr()});
      uids.insert(uids.end(), {'b', 'c', 'd'});
    }
    auto variantPack = cudnn_frontend::VariantPackBuilder()
      .setWorkspacePointer(workspace_size ? workspace_ptr.get() : nullptr)
      .setDataPointers(uids.size(), data_ptrs.data())
      .setUids(uids.size(), uids.data())
      .build();
    auto variant_pack_desc = variantPack.get_raw_desc();
    AT_CUDNN_CHECK(cudnnBackendExecute(handle, plan_desc.get_raw_desc(), variant_pack_desc));
  };

  auto search = execution_plan_cache.find(key);
  if (search != execution_plan_cache.end()) {
    cudnn_frontend::ExecutionPlan plan_desc = search->second;
    run(plan_desc);
    return;
  }
  // conv_op computes act_fp32 * w_fp32 (matrix multiplication)
  // where act_fp32 and w_fp32 are the input and weight variables, resp.
  // output is a fp32 tensor
  auto conv_op = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR)
      .setxDesc(cudnn_utils::getTensorDescriptor(input.sizes(), input.strides(), CUDNN_DATA_INT8, 'x', key.input_alignment))
      // for virtual tensors, the alignment is not used, so we can just put an arbitrary value here, e.g., key.output_alignment
      .setyDesc(cudnn_utils::getTensorDescriptor(quantized_output.sizes(), quantized_output.strides(), CUDNN_DATA_FLOAT, 'y', key.output_alignment, true))
      .setwDesc(cudnn_utils::getTensorDescriptor(maybe_padded_weight_.sizes(), maybe_padded_weight_.strides(), CUDNN_DATA_INT8, 'w', key.weight_alignment))
      .setcDesc(getConvDescriptor(key.params.dataType, padding_vec, stride_vec, dilation_vec))
      .build();
  // std::cout << "operator:" << conv_op.describe() << std::endl;

  c10::optional<cudnn_frontend::Operation> bias_mult_op;
  c10::optional<cudnn_frontend::Operation> sum_conv_bias_op;
  if (bias_.has_value()) {
    // we can't directly assign bias_mult_op because operator= is deleted for cudnn_frontend::Operation;
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
    // where the 1st and 2nd summands is output of conv_op and broadcasted_bias, resp.
    // output is a fp32 tensor
    // we use inplace operation here where the output is assigned to the input
    sum_conv_bias_op.emplace(cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
      .setxDesc(conv_op.getOutputTensor())
      .setbDesc(cudnn_utils::getTensorDescriptor(broadcasted_bias.value(), 'd', cudnn_utils::getAlignment(broadcasted_bias.value())))
      // for virtual tensors, the alignment is not used, so we can just put an arbitrary value here, e.g., key.output_alignment
      .setyDesc(cudnn_utils::getTensorDescriptor(quantized_output.sizes(), quantized_output.strides(), CUDNN_DATA_FLOAT, 'e', key.output_alignment, true))
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
      // for virtual tensors, the alignment is not used, so we can just put an arbitrary value here, e.g., key.output_alignment
      .setyDesc(cudnn_utils::getTensorDescriptor(quantized_output.sizes(), quantized_output.strides(), CUDNN_DATA_FLOAT, 'f', key.output_alignment, true))
      .setpwDesc(cudnn_utils::getPointWiseReluDescriptor(CUDNN_DATA_FLOAT))
      .build());
  }

  // relu_op computes relu(act_int8 * w_int8 + [bias_fp32/(act_scale * w_scale)]) / (out_scale / (act_scale * w_scale))
  // or relu(act_int8 * w_int8) / (out_scale / (act_scale * w_scale))) if bias is not present.
  // output is a fp32 tensor
  auto requant_op = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
    .setxDesc(kReluFused ? relu_op.value().getOutputTensor() : tensor2requant_ptr)
    .setbDesc(cudnn_utils::getTensorDescriptor(requantize_multiplier_tensor, 's', cudnn_utils::getAlignment(requantize_multiplier_tensor)))
    .setyDesc(cudnn_utils::getTensorDescriptor(quantized_output.sizes(), quantized_output.strides(), CUDNN_DATA_INT8, 'r', key.output_alignment))
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
      run(plan);
      execution_plan_cache.emplace(key, plan);
      return;
    } catch (cudnn_frontend::cudnnException &e) {std::cout << "cudnn error:" << e.what() << std::endl;} catch(c10::CuDNNError &e) { std::cout << "other error" << e.what() << std::endl;}
  }

  TORCH_CHECK(false, "Unable to find an engine to execute this computation in Quantized Conv2D Cudnn");
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
  const auto batch_size = kSpatialDim == 2 ? act.size(0) : 1;
  const auto num_input_channels = act.size(kSpatialDim - 1);
  const auto H = act.size(kSpatialDim);
  const auto W = act.size(kSpatialDim + 1);
  const auto num_output_channels = maybe_padded_weight_.size(0); // output channels
  std::vector<int64_t> kernel_size = {maybe_padded_weight_.size(2), maybe_padded_weight_.size(3)};
  auto output_shape = at::native::quantized::MakeConvOutputShape<kSpatialDim>(batch_size, num_output_channels, {H, W},
  kernel_size, stride_, padding_, dilation_);
  at::Tensor quantized_output = at::_empty_affine_quantized(
      output_shape,
      at::device(at::kCUDA).dtype(at::ScalarType::QInt8),
      output_scale,
      output_zero_point,
      at::MemoryFormat::ChannelsLast);

  // cudnn v8.4.0 expects conv2d's int8 activation tensor's input channels to be a multiple of 4. if it is not
  // we need to explicitly pad it to a multiple of 4 ourselves as cudnn does not currently support padding.
  // TODO: when and if cudnn enables padding in their operators, we can remove padding on our end;
  // currently, limit padding support to groups=1 (ungrouped conv)
  // TODO: implement this for groups > 1; should be straightforward since we're only padding a single dimension
  auto act_maybe_padded = act;
  if (num_input_channels % 4 != 0) {
    int8_t num_slices = 4 - num_input_channels % 4; // number of slices we need to pad
    act_maybe_padded = at::pad(act, {0, 0, 0, 0, 0, num_slices, 0, 0}, "constant", 0);
  }
  apply_impl_helper<kReluFused>(
      quantized_output, act_maybe_padded.to(c10::MemoryFormat::ChannelsLast), output_scale);

  // need to return sliced tensor if output_channels was padded
  if (num_unpadded_output_channels_ != maybe_padded_weight_.size(0)) {
    return quantized_output.slice(1, 0, num_unpadded_output_channels_);
  }
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

namespace at:: native {
namespace {

template <bool kReluFused>
class QConv1dInt8 final {
 public:
  static Tensor run(
      Tensor act,
      const c10::intrusive_ptr<ConvPackedParamsBase<2>>& packed_weight,
      double output_scale,
      int64_t output_zero_point) {
    at::Tensor output;
    // we currently use conv2d kernel for conv1d by making the input and weight tensors
    // 4D rather than 3D. we add a dummy width dimension of size 1
    // N, C, L -> N, C, 1, L
    act = act.unsqueeze(-2);
    if (kReluFused) {
      output = packed_weight->apply_relu(act, output_scale, output_zero_point);
    } else {
      output = packed_weight->apply(act, output_scale, output_zero_point);
    }
    // N, C, 1, L -> N, C, L
    return output.squeeze_(-2);
  }
};

template <int kSpatialDim, bool kReluFused>
class QConvInt8 final {
 public:
  static at::Tensor run(
      at::Tensor act,
      const c10::intrusive_ptr<ConvPackedParamsBase<kSpatialDim>>& packed_weight,
      double output_scale,
      int64_t output_zero_point) {
    TORCH_CHECK(kSpatialDim == 1 || kSpatialDim == 2, "Error in quantized cudnn conv2d operator: "
                "Expected kSpatialDim == 1 || kSpatialDim == 2; received kSpatialDim=", kSpatialDim);
    // TODO: check all zero_points are zero/all tensors are symmetrically quantized
    if (kReluFused) {
      return packed_weight->apply_relu(act, output_scale, output_zero_point);
    } else {
      return packed_weight->apply(act, output_scale, output_zero_point);
    }
  }
};

TORCH_LIBRARY_IMPL(quantized, QuantizedCUDA, m) {
  // the cpu conv1d doesn't use the quantized::conv1d*.new variant for packed weights. instead it just uses
  // quantized::conv1d for packed weights (see quantized/library.cpp).
  // this is inconsistent with what has been done for conv2d where new variants use packed weights, and
  // old variant does not. we adopt this inconsistency for now to be consistent with QuantizedCPU's conv1d
  // and will eventually deprecate the old variants
  register_conv_params<2>();
  register_conv_params<3>();
  m.impl(TORCH_SELECTIVE_NAME("quantized::conv1d"), QConv1dInt8<false>::run);
  m.impl(TORCH_SELECTIVE_NAME("quantized::conv1d_relu"), QConv1dInt8<true>::run);
  m.impl(TORCH_SELECTIVE_NAME("quantized::conv2d.new"), QConvInt8<2, false>::run);
  m.impl(TORCH_SELECTIVE_NAME("quantized::conv2d_relu.new"), QConvInt8<2, true>::run);
}

} // anonymous namespace
} // namespace at::native


#endif  // AT_CUDNN_ENABLED
#endif  // USE_CUDA
