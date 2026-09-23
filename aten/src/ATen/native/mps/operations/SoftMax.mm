//  Copyright © 2022 Apple Inc.
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/ceil_div.h>
#include <ATen/native/mps/OperationUtils.h>
#include <ATen/native/mps/kernels/SoftMax.h>
#include <c10/util/TypeCast.h>
#include <c10/util/accumulate.h>
#include <bit>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_log_softmax_native.h>
#include <ATen/ops/_softmax_backward_data_native.h>
#include <ATen/ops/_softmax_native.h>
#include <ATen/ops/empty.h>
#endif

namespace at::native {

using namespace mps;

#ifndef PYTORCH_JIT_COMPILE_SHADERS
static auto& lib = MetalShaderLibrary::getBundledLibrary();
#else
#include <ATen/native/mps/SoftMax_metallib.h>
#endif

static SoftmaxParams<uint32_t> narrow_params(const SoftmaxParams<uint64_t>& params) {
  SoftmaxParams<uint32_t> result{.dim_size = c10::checked_convert<uint32_t>(params.dim_size, "uint32_t"),
                                 .num_rows = c10::checked_convert<uint32_t>(params.num_rows, "uint32_t"),
                                 .inner_size = c10::checked_convert<uint32_t>(params.inner_size, "uint32_t"),
                                 .chunk_size = c10::checked_convert<uint32_t>(params.chunk_size, "uint32_t"),
                                 .n_chunks = c10::checked_convert<uint32_t>(params.n_chunks, "uint32_t"),
                                 .ndim = params.ndim,
                                 .dim = params.dim};
  for (const auto d : c10::irange(params.ndim)) {
    result.sizes[d] = c10::checked_convert<uint32_t>(params.sizes[d], "uint32_t");
    result.strides[d] = c10::checked_convert<uint32_t>(params.strides[d], "uint32_t");
  }
  return result;
}

template <typename... Tensors>
static void run_softmax(MPSStream* stream,
                        const std::string& kernel,
                        const Tensor& self,
                        bool use_u32,
                        MTLSize grid,
                        MTLSize group,
                        const SoftmaxParams<uint64_t>& params,
                        const Tensors&... tensors) {
  auto encoder = stream->commandEncoder();
  auto pso =
      lib.getPipelineStateForFunc(fmt::format("{}_{}{}", kernel, scalarToMetalTypeString(self), mtlIdxSuffix(use_u32)));
  getMPSProfiler().beginProfileKernel(pso, kernel, {self}, stream);
  [encoder setComputePipelineState:pso];
  if (use_u32) {
    mtl_setArgs(encoder, tensors..., narrow_params(params));
  } else {
    mtl_setArgs(encoder, tensors..., params);
  }
  [encoder dispatchThreadgroups:grid threadsPerThreadgroup:group];
  getMPSProfiler().endProfileKernel(pso, stream);
}

static void softmax_mps_impl(const Tensor& self, int64_t dim, bool half_to_float, const Tensor& out, bool log_softmax) {
  const std::string kernel = log_softmax ? "log_softmax" : "softmax";
  TORCH_CHECK(!half_to_float, kernel, " with half to float conversion is not supported on MPS");
  if (self.numel() == 0) {
    return;
  }
  TORCH_CHECK_NOT_IMPLEMENTED(supportedFloatingType(self), kernel, " not implemented on MPS for ", self.scalar_type());

  const auto self_ = self.dim() == 0 ? self.view(1) : self;
  const auto wrapped_dim = maybe_wrap_dim(dim, self_.dim());
  const auto dim_size = static_cast<uint64_t>(self_.size(wrapped_dim));
  const auto inner_size = static_cast<uint64_t>(c10::multiply_integers(self_.sizes().slice(wrapped_dim + 1)));
  const auto num_rows = static_cast<uint64_t>(self_.numel()) / dim_size;
  const auto output = out.is_contiguous() ? out : at::empty(out.sizes(), out.options());
  const bool use_u32 = offsetsFitIn<int32_t>(self_, output);
  const bool contiguous_rows = self_.is_contiguous() && wrapped_dim == self_.dim() - 1;
  // one simdgroup per row stops paying off past this width
  constexpr uint64_t row_kernel_max_dim = 2048;
  // rows wider than this need many rows to fill the GPU
  constexpr uint64_t row_kernel_solo_dim = 1024;
  constexpr uint64_t row_kernel_min_rows = 128;
  const bool row_dim_fits = dim_size <= row_kernel_solo_dim || num_rows >= row_kernel_min_rows;
  const bool use_row_kernel = contiguous_rows && dim_size <= row_kernel_max_dim && row_dim_fits;
  // split long rows into enough chunks to occupy the GPU
  constexpr uint64_t split_target_groups = 512;
  // below this width a chunk cannot amortize the second pass
  constexpr uint64_t split_min_chunk = 2048;
  const auto n_chunks = std::clamp(split_target_groups / num_rows, uint64_t(1), ceil_div(dim_size, split_min_chunk));
  const auto chunk_size = ceil_div(dim_size, n_chunks);
  const bool use_split = contiguous_rows && !use_row_kernel && n_chunks > 1;
  SoftmaxParams<uint64_t> params{.dim_size = dim_size,
                                 .num_rows = num_rows,
                                 .inner_size = inner_size,
                                 .chunk_size = chunk_size,
                                 .n_chunks = n_chunks,
                                 .ndim = static_cast<uint32_t>(self_.dim()),
                                 .dim = static_cast<uint32_t>(wrapped_dim)};
  for (const auto d : c10::irange(self_.dim())) {
    params.sizes[d] = self_.size(d);
    params.strides[d] = self_.size(d) == 1 ? 0 : self_.stride(d);
  }
  const auto partials = use_split
      ? at::empty({static_cast<int64_t>(num_rows), static_cast<int64_t>(n_chunks), 2}, self.options().dtype(kFloat))
      : Tensor();

  MPSStream* stream = getCurrentMPSStream();
  dispatch_sync_with_rethrow(stream->queue(), ^() {
    @autoreleasepool {
      if (use_row_kernel) {
        constexpr auto rows_per_group = kSoftmaxThreads / c10::metal::simdgroup_size;
        const auto grid = MTLSizeMake(ceil_div(num_rows, uint64_t(rows_per_group)), 1, 1);
        const auto group = MTLSizeMake(kSoftmaxThreads, 1, 1);
        run_softmax(stream, kernel + "_row", self, use_u32, grid, group, params, self_, output);
      } else if (use_split) {
        const auto grid = MTLSizeMake(n_chunks, num_rows, 1);
        const auto group = MTLSizeMake(kSoftmaxThreads, 1, 1);
        run_softmax(stream, "softmax_partial", self, use_u32, grid, group, params, self_, partials);
        run_softmax(stream, kernel + "_finalize", self, use_u32, grid, group, params, self_, output, partials);
      } else {
        // double the width for 2-byte dtypes so a threadgroup row spans a full 128-byte cache line
        const auto max_tg_x = uint64_t((self.element_size() == 2 ? 2u : 1u) * c10::metal::simdgroup_size);
        const auto tg_x = std::min(inner_size, max_tg_x);
        const auto tg_y =
            std::min({std::bit_ceil(dim_size), uint64_t(kSoftmaxThreads), std::bit_floor(kSoftmaxMaxThreads / tg_x)});
        const auto grid = MTLSizeMake(ceil_div(inner_size, tg_x), num_rows / inner_size, 1);
        const auto group = MTLSizeMake(tg_x, tg_y, 1);
        run_softmax(stream, kernel, self, use_u32, grid, group, params, self_, output);
      }
    }
  });
  if (!out.is_contiguous()) {
    out.copy_(output);
  }
}

TORCH_IMPL_FUNC(softmax_mps_out)
(const Tensor& self, const int64_t dim, const bool half_to_float, const Tensor& out) {
  softmax_mps_impl(self, dim, half_to_float, out, false);
}

TORCH_IMPL_FUNC(log_softmax_mps_out)
(const Tensor& self, const int64_t dim, const bool half_to_float, const Tensor& out) {
  softmax_mps_impl(self, dim, half_to_float, out, true);
}

TORCH_IMPL_FUNC(softmax_backward_mps_out)
(const Tensor& grad_, const Tensor& output_, int64_t dim, ScalarType input_dtype, const Tensor& grad_input) {
  if (output_.numel() == 0) {
    return;
  }

  Tensor grad;
  if (grad_.dim() == 0) {
    grad = grad_.view(1);
  } else
    grad = grad_;

  Tensor output;
  if (output_.dim() == 0) {
    output = output_.view(1);
  } else
    output = output_;

  int64_t dim_ = maybe_wrap_dim(dim, grad.dim());
  TORCH_CHECK(dim_ >= 0 && dim_ < grad.dim(), "Grad:dim must be non-negative and less than input dimensions");

  using namespace mps;
  using CachedGraph = MPSUnaryGradCachedGraph;
  MPSStream* stream = getCurrentMPSStream();

  @autoreleasepool {
    MPSShape* grad_shape = mps::getMPSShape(grad);
    NSString* ns_shape_key = [[grad_shape valueForKey:@"description"] componentsJoinedByString:@","];

    std::string key = "softmax_backward_mps_out:" + getMPSTypeString(output) + ":" + [ns_shape_key UTF8String] + ":" +
        std::to_string(dim_);
    auto cachedGraph = LookUpOrCreateCachedGraph<CachedGraph>(key, [&](auto mpsGraph, auto newCachedGraph) {
      MPSGraphTensor* softmaxTensor = mpsGraphRankedPlaceHolder(mpsGraph, getMPSDataType(output), grad_shape);
      MPSGraphTensor* gradOutputTensor = mpsGraphRankedPlaceHolder(mpsGraph, getMPSDataType(grad), grad_shape);

      MPSGraphTensor* mulTensor = [mpsGraph multiplicationWithPrimaryTensor:softmaxTensor
                                                            secondaryTensor:gradOutputTensor
                                                                       name:nil];
      MPSGraphTensor* mulSumTensor = [mpsGraph reductionSumWithTensor:mulTensor axis:(NSInteger)dim_ name:nil];
      MPSGraphTensor* gradSubTensor = [mpsGraph subtractionWithPrimaryTensor:gradOutputTensor
                                                             secondaryTensor:mulSumTensor
                                                                        name:nil];
      MPSGraphTensor* gradInputTensor = [mpsGraph multiplicationWithPrimaryTensor:softmaxTensor
                                                                  secondaryTensor:gradSubTensor
                                                                             name:nil];

      newCachedGraph->outputTensor_ = softmaxTensor;
      newCachedGraph->gradOutputTensor_ = gradOutputTensor;
      newCachedGraph->gradInputTensor_ = gradInputTensor;
    });

    Placeholder softmaxPlaceholder = Placeholder(cachedGraph->outputTensor_, output, grad_shape);
    Placeholder gradOutputPlaceholder = Placeholder(cachedGraph->gradOutputTensor_, grad, grad_shape);
    Placeholder gradInputPlaceholder = Placeholder(cachedGraph->gradInputTensor_, grad_input);

    auto feeds = dictionaryFromPlaceholders(softmaxPlaceholder, gradOutputPlaceholder);
    runMPSGraph(stream, cachedGraph->graph(), feeds, gradInputPlaceholder);
  }
}

} // namespace at::native
