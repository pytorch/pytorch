#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/TensorUtils.h>
#include <ATen/core/Tensor.h>
#include <ATen/mps/MPSProfiler.h>
#include <ATen/native/EmbeddingBag.h>
#include <ATen/native/Pool.h>
#include <ATen/native/mps/OperationUtils.h>
#include <ATen/native/mps/kernels/EmbeddingBag.h>

#include <fmt/format.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_embedding_bag_dense_backward_native.h>
#include <ATen/ops/_embedding_bag_forward_only_native.h>
#include <ATen/ops/_embedding_bag_native.h>
#include <ATen/ops/_embedding_bag_per_sample_weights_backward_native.h>
#include <ATen/ops/empty.h>
#endif

namespace at::native {

#ifndef PYTORCH_JIT_COMPILE_SHADERS
static auto& lib = mps::MetalShaderLibrary::getBundledLibrary();
#else
#include <ATen/native/mps/EmbeddingBag_metallib.h>
#endif

namespace {

std::pair<Tensor, Tensor> promoteIndicesAndOffsets(const Tensor& indices, const Tensor& offsets) {
  const auto commonType = promoteTypes(offsets.scalar_type(), indices.scalar_type());
  return {indices.scalar_type() == commonType ? indices : indices.toType(commonType),
          offsets.scalar_type() == commonType ? offsets : offsets.toType(commonType)};
}

} // namespace

namespace mps {

static std::tuple<Tensor, Tensor, Tensor, Tensor> _embedding_bag_mps_impl(
    const Tensor& weight,
    const Tensor& indices_,
    const Tensor& offsets_,
    const bool scale_grad_by_freq,
    const int64_t mode,
    bool sparse,
    const std::optional<Tensor>& per_sample_weights_opt,
    bool include_last_offset,
    int64_t padding_idx) {
  TORCH_CHECK(indices_.dim() == 1, "input has to be a 1D Tensor, but got Tensor of dimension ", indices_.dim());
  if (indices_.dim() == 1) {
    TORCH_CHECK(offsets_.dim() == 1, "offsets has to be a 1D Tensor, but got Tensor of dimension ", offsets_.dim());
  }
  TORCH_CHECK(weight.dim() == 2, "weight has to be a 2D Tensor, but got Tensor of dimension ", weight.dim());

  Tensor indices, offsets;
  std::tie(indices, offsets) = promoteIndicesAndOffsets(indices_, offsets_);
  auto indices_arg = TensorArg(indices, "indices", 1);
  checkScalarTypes("embedding_bag_mps", indices_arg, {kLong, kInt});
  auto offsets_arg = TensorArg(offsets, "offsets", 1);
  checkScalarTypes("embedding_bag_mps", offsets_arg, {kLong, kInt});
  checkSameType("embedding_bag_mps", indices_arg, offsets_arg);
  auto weight_arg = TensorArg(weight, "weight", 1);

  int64_t num_indices = indices.size(0);
  int64_t num_bags = offsets.size(0);
  if (include_last_offset) {
    TORCH_CHECK(num_bags >= 1, "include_last_offset: number of offsets should be at least 1");
    num_bags -= 1;
  }
  int64_t feature_size = weight.size(1);

  auto bag_size = at::empty({num_bags}, indices.options());
  auto offset2bag = at::empty({indices.size(0)}, indices.options());
  auto output = at::empty({num_bags, feature_size}, weight.options());

  Tensor max_indices;

  if (mode == EmbeddingBagMode::MAX) {
    max_indices = at::empty({num_bags, feature_size}, indices.options());
  } else {
    max_indices = at::empty({0}, indices.options());
  }

  EmbeddingBagParams<uint32_t> params;

  for (const auto dim : c10::irange(weight.dim())) {
    params.weight_strides[dim] = safe_downcast<uint32_t, int64_t>(weight.stride(dim));
    params.output_strides[dim] = safe_downcast<uint32_t, int64_t>(output.stride(dim));

    if (mode == EmbeddingBagMode::MAX) {
      params.max_indices_strides[dim] = safe_downcast<uint32_t, int64_t>(max_indices.stride(dim));
    }
  }

  bool use_per_sample_weights = per_sample_weights_opt.has_value() && per_sample_weights_opt->defined();
  params.use_per_sample_weights = use_per_sample_weights;
  params.per_sample_weights_stride = use_per_sample_weights ? per_sample_weights_opt->stride(0) : 0;

  params.num_indices = num_indices;
  params.num_bags = num_bags;
  params.feature_size = feature_size;
  params.mode = static_cast<EmbeddingBagMode>(mode);
  params.padding_idx = padding_idx;

  auto num_threads = output.numel();
  MPSStream* stream = getCurrentMPSStream();

  dispatch_sync_with_rethrow(stream->queue(), ^() {
    @autoreleasepool {
      id<MTLComputeCommandEncoder> computeEncoder = stream->commandEncoder();
      auto pipeline_state = lib.getPipelineStateForFunc(
          fmt::format("embedding_bag_{}_{}", scalarToMetalTypeString(weight), scalarToMetalTypeString(indices)));

      getMPSProfiler().beginProfileKernel(pipeline_state, "embedding_bag", {weight, indices, offsets});
      [computeEncoder setComputePipelineState:pipeline_state];
      mtl_setArgs(computeEncoder,
                  weight,
                  indices,
                  offsets,
                  use_per_sample_weights ? per_sample_weights_opt : std::nullopt,
                  output,
                  offset2bag,
                  bag_size,
                  max_indices,
                  params);

      mtl_dispatch1DJob(computeEncoder, pipeline_state, num_threads);
      getMPSProfiler().endProfileKernel(pipeline_state);
    }
  });

  return std::tuple<Tensor, Tensor, Tensor, Tensor>(
      std::move(output), std::move(offset2bag), std::move(bag_size), std::move(max_indices));
}

} // namespace mps

std::tuple<Tensor, Tensor, Tensor, Tensor> _embedding_bag_mps(const Tensor& weight,
                                                              const Tensor& indices,
                                                              const Tensor& offsets,
                                                              const bool scale_grad_by_freq,
                                                              const int64_t mode,
                                                              bool sparse,
                                                              const std::optional<Tensor>& per_sample_weights_opt,
                                                              bool include_last_offset,
                                                              int64_t padding_idx) {
  return mps::_embedding_bag_mps_impl(weight,
                                      indices,
                                      offsets,
                                      scale_grad_by_freq,
                                      mode,
                                      sparse,
                                      per_sample_weights_opt,
                                      include_last_offset,
                                      padding_idx);
}

std::tuple<Tensor, Tensor, Tensor, Tensor> _embedding_bag_forward_only_mps(
    const Tensor& weight,
    const Tensor& indices,
    const Tensor& offsets,
    const bool scale_grad_by_freq,
    const int64_t mode,
    bool sparse,
    const std::optional<Tensor>& per_sample_weights_opt,
    bool include_last_offset,
    int64_t padding_idx) {
  return _embedding_bag_mps(weight,
                            indices,
                            offsets,
                            scale_grad_by_freq,
                            mode,
                            sparse,
                            per_sample_weights_opt,
                            include_last_offset,
                            padding_idx);
}

Tensor _embedding_bag_dense_backward_mps(const Tensor& output_grad,
                                         const Tensor& indices,
                                         const Tensor& offset2bag,
                                         const Tensor& bag_size,
                                         const Tensor& max_indices,
                                         int64_t num_weights,
                                         bool scale_grad_by_freq,
                                         int64_t mode,
                                         const std::optional<Tensor>& per_sample_weights_opt,
                                         int64_t padding_idx) {
  // indices and offset2bag are assumed having correct dtypes and
  // contiguous here due to the checks in _embedding_bag_backward in
  // EmbeddingBag.cpp.
  // Also see NOTE [ embedding_bag Native Functions ] in native_functions.yaml
  // for more details.

  int64_t feature_size = output_grad.size(1);
  auto weight_grad = at::zeros({num_weights, feature_size}, output_grad.options());
  EmbeddingBagBackwardParams<uint32_t> params;

  for (const auto dim : c10::irange(2)) {
    params.output_grad_strides[dim] = output_grad.stride(dim);
    params.weight_grad_strides[dim] = weight_grad.stride(dim);

    if (mode == EmbeddingBagMode::MAX) {
      params.max_indices_strides[dim] = safe_downcast<uint32_t, int64_t>(max_indices.stride(dim));
    }
  }

  bool use_per_sample_weights = per_sample_weights_opt.has_value() && per_sample_weights_opt->defined();
  params.use_per_sample_weights = use_per_sample_weights;
  params.per_sample_weights_stride = use_per_sample_weights ? per_sample_weights_opt->stride(0) : 0;
  params.feature_size = output_grad.size(1);
  params.mode = static_cast<EmbeddingBagMode>(mode);
  params.padding_idx = padding_idx;

  auto num_indices = offset2bag.numel();
  auto num_threads = (params.mode == EmbeddingBagMode::MAX) ? output_grad.numel() : num_indices * params.feature_size;
  MPSStream* stream = getCurrentMPSStream();

  dispatch_sync_with_rethrow(stream->queue(), ^() {
    @autoreleasepool {
      id<MTLComputeCommandEncoder> computeEncoder = stream->commandEncoder();
      auto pipeline_state = lib.getPipelineStateForFunc(fmt::format("embedding_bag_backward_{}_{}",
                                                                    mps::scalarToMetalTypeString(output_grad),
                                                                    mps::scalarToMetalTypeString(indices)));

      getMPSProfiler().beginProfileKernel(
          pipeline_state, "embedding_bag", {output_grad, indices, offset2bag, bag_size});
      [computeEncoder setComputePipelineState:pipeline_state];
      mps::mtl_setArgs(computeEncoder,
                       output_grad,
                       indices,
                       offset2bag,
                       bag_size,
                       max_indices,
                       use_per_sample_weights ? per_sample_weights_opt : std::nullopt,
                       weight_grad,
                       params);

      mps::mtl_dispatch1DJob(computeEncoder, pipeline_state, num_threads);
      getMPSProfiler().endProfileKernel(pipeline_state);
    }
  });

  return std::move(weight_grad);
}

Tensor _embedding_bag_per_sample_weights_backward_mps(const Tensor& output_grad,
                                                      const Tensor& weight,
                                                      const Tensor& indices,
                                                      const Tensor& offsets,
                                                      const Tensor& offset2bag,
                                                      int64_t mode,
                                                      int64_t padding_idx) {
  TORCH_INTERNAL_ASSERT(static_cast<EmbeddingBagMode>(mode) == EmbeddingBagMode::SUM);
  int64_t num_indices = indices.size(0);
  int64_t feature_size = output_grad.size(1);
  auto per_sample_weights_grad = at::zeros({num_indices}, output_grad.options());
  EmbeddingBagPerSampleWeightsBackwardParams params;

  for (const auto dim : c10::irange(2)) {
    params.output_grad_strides[dim] = output_grad.stride(dim);
    params.weight_strides[dim] = weight.stride(dim);
  }

  params.per_sample_weights_grad_stride = per_sample_weights_grad.stride(0);
  params.feature_size = feature_size;
  params.padding_idx = padding_idx;

  auto num_threads = num_indices * feature_size;
  MPSStream* stream = getCurrentMPSStream();

  dispatch_sync_with_rethrow(stream->queue(), ^() {
    @autoreleasepool {
      id<MTLComputeCommandEncoder> computeEncoder = stream->commandEncoder();
      auto pipeline_state = lib.getPipelineStateForFunc(fmt::format("embedding_bag_per_sample_weights_backward_{}_{}",
                                                                    mps::scalarToMetalTypeString(output_grad),
                                                                    mps::scalarToMetalTypeString(indices)));

      getMPSProfiler().beginProfileKernel(
          pipeline_state, "embedding_bag_per_sample_weights_backward", {output_grad, weight, indices, offset2bag});
      [computeEncoder setComputePipelineState:pipeline_state];
      mps::mtl_setArgs(computeEncoder, output_grad, weight, indices, offset2bag, per_sample_weights_grad, params);

      mps::mtl_dispatch1DJob(computeEncoder, pipeline_state, num_threads);
      getMPSProfiler().endProfileKernel(pipeline_state);
    }
  });

  return std::move(per_sample_weights_grad);
}

} // namespace at::native
