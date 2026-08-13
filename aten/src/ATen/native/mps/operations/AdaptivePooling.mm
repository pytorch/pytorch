//  Copyright © 2022 Apple Inc.
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <fmt/format.h>
#include <string_view>

#include <ATen/mps/MPSProfiler.h>
#include <ATen/native/Pool.h>
#include <ATen/native/mps/OperationUtils.h>
#include <ATen/native/mps/kernels/AdaptivePooling.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_adaptive_avg_pool2d_backward_native.h>
#include <ATen/ops/_adaptive_avg_pool2d_native.h>
#include <ATen/ops/adaptive_avg_pool2d.h>
#include <ATen/ops/adaptive_avg_pool2d_native.h>
#include <ATen/ops/avg_pool2d.h>
#include <ATen/ops/avg_pool2d_backward.h>
#include <ATen/ops/mul.h>
#include <ATen/ops/ones_like.h>
#endif
namespace at::native {
namespace mps {

#ifndef PYTORCH_JIT_COMPILE_SHADERS
static auto& lib = MetalShaderLibrary::getBundledLibrary();
#else
#include <ATen/native/mps/AdaptivePooling_metallib.h>
#endif

static AdaptiveAvgPool2DParams adaptive_avg_pool2d_params(const Tensor& input, const Tensor& output) {
  const bool batched = input.dim() == 4;
  const auto input_strides = input.strides();
  const auto output_strides = output.strides();
  return {
      .B = batched ? input.size(0) : 1,
      .C = input.size(-3),
      .input_height = input.size(-2),
      .input_width = input.size(-1),
      .output_height = output.size(-2),
      .output_width = output.size(-1),
      .input_strides = {batched ? input_strides[0] : 0,
                        input_strides[batched ? 1 : 0],
                        input_strides[batched ? 2 : 1],
                        input_strides[batched ? 3 : 2]},
      .output_strides = {batched ? output_strides[0] : 0,
                         output_strides[batched ? 1 : 0],
                         output_strides[batched ? 2 : 1],
                         output_strides[batched ? 3 : 2]},
  };
}

static void adaptive_avg_pool2d_metal(const Tensor& input, Tensor& output, bool backward) {
  using namespace std::string_view_literals;

  auto stream = getCurrentMPSStream();
  const auto direction = backward ? "backward"sv : "forward"sv;
  // TODO: Use 32-bit indexing when input is small enough.
  const auto kernel = fmt::format("adaptive_avg_pool2d_{}_{}", direction, scalarToMetalTypeString(input));
  const auto params = backward ? adaptive_avg_pool2d_params(output, input) : adaptive_avg_pool2d_params(input, output);
  @autoreleasepool {
    auto pso = lib.getPipelineStateForFunc(kernel);
    dispatch_sync_with_rethrow(stream->queue(), ^() {
      @autoreleasepool {
        auto encoder = stream->commandEncoder();
        getMPSProfiler().beginProfileKernel(pso, kernel, {input});
        [encoder setComputePipelineState:pso];
        mtl_setArgs(encoder, input, output, params);
        mtl_dispatch1DJob(encoder, pso, output.numel());
        getMPSProfiler().endProfileKernel(pso);
      }
    });
  }
}

static void set_kernel_params(int64_t isizeH,
                              int64_t isizeW,
                              int64_t osizeH,
                              int64_t osizeW,
                              int64_t& strideH,
                              int64_t& strideW,
                              int64_t& kernel_sizeH,
                              int64_t& kernel_sizeW,
                              bool check_avg_pooling = false) {
  TORCH_CHECK((isizeH >= osizeH && isizeW >= osizeW) || (isizeH <= osizeH && isizeW <= osizeW),
              "Adaptive pool MPS: Input height and width must both be greater than, "
              "or equal to, or lesser than output height and width")

  if (isizeH >= osizeH) {
    if (check_avg_pooling) {
      TORCH_CHECK(
          (isizeH % osizeH == 0 && isizeW % osizeW == 0),
          "Adaptive pool MPS: input sizes must be divisible by output sizes. Non-divisible input sizes are not implemented on MPS device yet. For now, you can manually transfer tensor to cpu in this case. Please refer to [this issue](https://github.com/pytorch/pytorch/issues/96056)");
    }
    strideH = (int64_t)(isizeH / osizeH);
    strideW = (int64_t)(isizeW / osizeW);
    kernel_sizeH = isizeH - (osizeH - 1) * strideH;
    kernel_sizeW = isizeW - (osizeW - 1) * strideW;
  } else {
    if (check_avg_pooling) {
      TORCH_CHECK(
          (osizeH % isizeH == 0 && osizeW % isizeW == 0),
          "Adaptive pool MPS: output sizes must be divisible by input sizes. Non-divisible input sizes are not implemented on MPS device yet. For now, you can manually transfer tensor to cpu in this case. Please refer to [this issue](https://github.com/pytorch/pytorch/issues/96056)");
    }
    strideH = (int64_t)(osizeH / isizeH);
    strideW = (int64_t)(osizeW / isizeW);
    kernel_sizeH = osizeH - (isizeH - 1) * strideH;
    kernel_sizeW = osizeW - (isizeW - 1) * strideW;
  }
}
} // namespace mps

// Adaptive average pooling
Tensor& adaptive_avg_pool2d_out_mps(const Tensor& input, IntArrayRef output_size, Tensor& output) {
  for (int64_t i = 1; i < input.ndimension(); i++) {
    TORCH_CHECK(input.size(i) > 0,
                "adaptive_avg_pool2d(): Expected input to have non-zero size for non-batch dimensions, "
                "but input has sizes ",
                input.sizes(),
                " with dimension ",
                i,
                " being empty");
  }

  int64_t isizeH = input.size(-2);
  int64_t isizeW = input.size(-1);
  int64_t osizeH = output_size[0];
  int64_t osizeW = output_size[1];

  int64_t strideH = 0, strideW = 0;
  int64_t kernel_sizeH = 0, kernel_sizeW = 0;

  const bool divisible_downsample =
      isizeH >= osizeH && isizeW >= osizeW && isizeH % osizeH == 0 && isizeW % osizeW == 0;
  const bool divisible_upsample = isizeH <= osizeH && isizeW <= osizeW && osizeH % isizeH == 0 && osizeW % isizeW == 0;
  if (!divisible_downsample && !divisible_upsample) {
    mps::adaptive_avg_pool2d_metal(input, output, false);
    return output;
  }

  mps::set_kernel_params(isizeH, isizeW, osizeH, osizeW, strideH, strideW, kernel_sizeH, kernel_sizeW);

  if (isizeH >= osizeH) {
    output = at::avg_pool2d(input,
                            IntArrayRef({kernel_sizeH, kernel_sizeW}),
                            IntArrayRef({strideH, strideW}),
                            IntArrayRef({0, 0}),
                            false,
                            true,
                            std::nullopt);
  } else {
    Tensor phony_grad = at::ones_like(input, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
    auto input_sizes = input.sizes();
    std::vector<int64_t> phony_shape{input_sizes.begin(), input_sizes.end() - 2};
    phony_shape.push_back(output_size[0]);
    phony_shape.push_back(output_size[1]);
    phony_grad.resize_(IntArrayRef(phony_shape));
    output = at::avg_pool2d_backward(input,
                                     phony_grad,
                                     IntArrayRef({kernel_sizeH, kernel_sizeW}),
                                     IntArrayRef({strideH, strideW}),
                                     IntArrayRef({0, 0}),
                                     false,
                                     true,
                                     std::nullopt);
    // Multiply output by kernel size
    output = at::mul(output, kernel_sizeH * kernel_sizeW);
  }

  return output;
}

Tensor adaptive_avg_pool2d_mps(at::Tensor const& input, IntArrayRef output_size) {
  IntArrayRef output_shape;

  auto osizeH = output_size[0];
  auto osizeW = output_size[1];

  std::vector<long long> out_dims = {};

  if (input.ndimension() == 4) {
    auto sizeB = input.size(0);
    auto sizeD = input.size(1);

    out_dims.push_back(sizeB);
    out_dims.push_back(sizeD);
    out_dims.push_back(osizeH);
    out_dims.push_back(osizeW);
    output_shape = IntArrayRef(out_dims);
  } else {
    auto sizeD = input.size(0);
    out_dims.push_back(sizeD);
    out_dims.push_back(osizeH);
    out_dims.push_back(osizeW);
    output_shape = IntArrayRef(out_dims);
  }

  const auto memory_format = input.suggest_memory_format();
  Tensor output = at::empty(output_shape, input.scalar_type(), std::nullopt, kMPS, std::nullopt, memory_format);
  return adaptive_avg_pool2d_out_mps(input, output_size, output);
}

Tensor adaptive_avg_pool2d_backward_mps(const Tensor& gradOutput, const Tensor& input) {
  int64_t isizeH = input.size(-2);
  int64_t isizeW = input.size(-1);
  int64_t osizeH = gradOutput.size(-2);
  int64_t osizeW = gradOutput.size(-1);

  int64_t strideH = 0, strideW = 0;
  int64_t kernel_sizeH = 0, kernel_sizeW = 0;

  const bool regular_downsample = isizeH >= osizeH && isizeW >= osizeW && isizeH % osizeH == 0 && isizeW % osizeW == 0;
  const bool regular_upsample = isizeH <= osizeH && isizeW <= osizeW && osizeH % isizeH == 0 && osizeW % isizeW == 0;
  if (!regular_downsample && !regular_upsample) {
    auto gradInput = at::empty_like(input, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
    if (gradInput.numel() != 0) {
      mps::adaptive_avg_pool2d_metal(gradOutput, gradInput, true);
    }
    return gradInput;
  }

  mps::set_kernel_params(isizeH, isizeW, osizeH, osizeW, strideH, strideW, kernel_sizeH, kernel_sizeW);

  auto gradInput = at::zeros_like(input, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  if (gradInput.numel() != 0) {
    if (isizeH >= osizeH) {
      gradInput = at::avg_pool2d_backward(gradOutput,
                                          input,
                                          IntArrayRef({kernel_sizeH, kernel_sizeW}),
                                          IntArrayRef({strideH, strideW}),
                                          IntArrayRef({0, 0}),
                                          false,
                                          true,
                                          std::nullopt);
    } else {
      gradInput = at::avg_pool2d(gradOutput,
                                 IntArrayRef({kernel_sizeH, kernel_sizeW}),
                                 IntArrayRef({strideH, strideW}),
                                 IntArrayRef({0, 0}),
                                 false,
                                 true,
                                 std::nullopt);
      gradInput = at::mul(gradInput, kernel_sizeH * kernel_sizeW);
    }
  }

  return gradInput;
}

} // namespace at::native
