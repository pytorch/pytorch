//  Copyright © 2023 Apple Inc.
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/ceil_div.h>
#include <ATen/native/UpSample.h>
#include <ATen/native/mps/OperationUtils.h>
#include <c10/util/accumulate.h>
#include <fmt/format.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_upsample_bicubic2d_aa_backward_native.h>
#include <ATen/ops/_upsample_bicubic2d_aa_native.h>
#include <ATen/ops/_upsample_bilinear2d_aa_backward_native.h>
#include <ATen/ops/_upsample_bilinear2d_aa_native.h>
#include <ATen/ops/_upsample_nearest_exact1d.h>
#include <ATen/ops/_upsample_nearest_exact1d_backward.h>
#include <ATen/ops/_upsample_nearest_exact1d_backward_native.h>
#include <ATen/ops/_upsample_nearest_exact1d_native.h>
#include <ATen/ops/_upsample_nearest_exact2d.h>
#include <ATen/ops/_upsample_nearest_exact2d_backward.h>
#include <ATen/ops/_upsample_nearest_exact2d_backward_native.h>
#include <ATen/ops/_upsample_nearest_exact2d_native.h>
#include <ATen/ops/_upsample_nearest_exact3d_backward_native.h>
#include <ATen/ops/_upsample_nearest_exact3d_native.h>
#include <ATen/ops/upsample_bicubic2d_backward_native.h>
#include <ATen/ops/upsample_bicubic2d_native.h>
#include <ATen/ops/upsample_bilinear2d.h>
#include <ATen/ops/upsample_bilinear2d_backward.h>
#include <ATen/ops/upsample_bilinear2d_backward_native.h>
#include <ATen/ops/upsample_bilinear2d_native.h>
#include <ATen/ops/upsample_linear1d.h>
#include <ATen/ops/upsample_linear1d_backward.h>
#include <ATen/ops/upsample_linear1d_backward_native.h>
#include <ATen/ops/upsample_linear1d_native.h>
#include <ATen/ops/upsample_nearest1d.h>
#include <ATen/ops/upsample_nearest1d_backward.h>
#include <ATen/ops/upsample_nearest1d_backward_native.h>
#include <ATen/ops/upsample_nearest1d_native.h>
#include <ATen/ops/upsample_nearest2d.h>
#include <ATen/ops/upsample_nearest2d_backward.h>
#include <ATen/ops/upsample_nearest2d_backward_native.h>
#include <ATen/ops/upsample_nearest2d_native.h>
#include <ATen/ops/upsample_nearest3d_backward_native.h>
#include <ATen/ops/upsample_nearest3d_native.h>
#include <ATen/ops/upsample_trilinear3d_backward_native.h>
#include <ATen/ops/upsample_trilinear3d_native.h>
#endif

#include <ATen/native/mps/kernels/UpSample.h>

namespace at::native {
using namespace mps;

namespace {

#ifndef PYTORCH_JIT_COMPILE_SHADERS
static auto& lib = MetalShaderLibrary::getBundledLibrary();
#else
#include <ATen/native/mps/UpSample_metallib.h>
#endif

// Encode a forward/backward upsample kernel: bind the PSO, the two tensors and
// the params struct, then launch one thread per output spatial element.
template <typename params_t>
static void dispatch_upsample(const std::string& fname,
                              const Tensor& a,
                              const Tensor& b,
                              const params_t& params,
                              int64_t njobs) {
  auto upsamplePSO = lib.getPipelineStateForFunc(fname);
  auto stream = getCurrentMPSStream();
  dispatch_sync_with_rethrow(stream->queue(), ^() {
    @autoreleasepool {
      auto computeEncoder = stream->commandEncoder();
      [computeEncoder setComputePipelineState:upsamplePSO];
      mtl_setArgs(computeEncoder, a, b, params);
      mtl_dispatch1DJob(computeEncoder, upsamplePSO, njobs);
    }
  });
}

// scales are innermost-first (w, h, d), matching UpsampleParams; the constructor
// consumes the first N-2 entries.
template <unsigned N>
static void upsample_kernel_out_template(const Tensor& input,
                                         IntArrayRef output_size,
                                         bool align_corners,
                                         std::initializer_list<std::optional<double>> scales,
                                         const Tensor& output,
                                         const std::string& name) {
  if (output.numel() == 0) {
    return;
  }
  UpsampleParams<N> params(input, output, align_corners, scales);
  dispatch_upsample(fmt::format("upsample_{}_{}", name, scalarToMetalTypeString(input)),
                    input,
                    output,
                    params,
                    c10::multiply_integers(output_size));
}

template <unsigned N>
static void upsample_kernel_backward_out_template(const Tensor& grad_input,
                                                  const Tensor& grad_output,
                                                  IntArrayRef output_size,
                                                  bool align_corners,
                                                  std::initializer_list<std::optional<double>> scales,
                                                  const std::string& name) {
  grad_input.zero_();
  if (grad_output.numel() == 0) {
    return;
  }

  // See Note [Writing Nondeterministic Operations]
  // Nondeterministic due to atomic_add
  at::globalContext().alertNotDeterministic(fmt::format("upsample_{}_backward", name));

  UpsampleParams<N> params(grad_input, grad_output, align_corners, scales);
  dispatch_upsample(fmt::format("upsample_{}_backward_{}", name, scalarToMetalTypeString(grad_input)),
                    grad_input,
                    grad_output,
                    params,
                    c10::multiply_integers(output_size));
}

static void upsample_gather_backward_out_template(const Tensor& grad_input,
                                                  const Tensor& grad_output,
                                                  bool align_corners,
                                                  std::initializer_list<std::optional<double>> scales,
                                                  const std::string& name) {
  TORCH_CHECK_NOT_IMPLEMENTED(at::isFloatingType(grad_output.scalar_type()),
                              "upsample_",
                              name,
                              "_backward not implemented for ",
                              grad_output.scalar_type());
  if (grad_input.numel() == 0) {
    return;
  }
  const UpsampleParams<4> params(grad_input, grad_output, align_corners, scales);
  const bool channels_last = grad_input.suggest_memory_format() == MemoryFormat::ChannelsLast;
  const auto height = static_cast<NSUInteger>(grad_input.size(2));
  const auto width = static_cast<NSUInteger>(grad_input.size(3));
  const auto all_planes = static_cast<NSUInteger>(grad_input.size(0) * grad_input.size(1));
  static const auto core_count = std::max(MPSDevice::getInstance()->getCoreCount(), 1u);
  auto stream = getCurrentMPSStream();
  dispatch_sync_with_rethrow(stream->queue(), ^() {
    @autoreleasepool {
      auto pso = lib.getPipelineStateForFunc(
          fmt::format("upsample_{}_backward_{}", name, scalarToMetalTypeString(grad_input)));
      const auto max_threads = [pso maxTotalThreadsPerThreadgroup];
      const auto simd_width = [pso threadExecutionWidth];
      // Threads the grid gets before planes fold into per-thread loops: folding
      // amortizes the range search, fewer threads leave GPU cores idle.
      const auto threads_in_flight = core_count * max_threads * GATHER_BACKWARD_TGS_PER_CORE;
      const auto planes =
          std::min(all_planes, at::round_up(at::ceil_div(threads_in_flight, height * width), simd_width));
      const auto tg_width = std::min(width, max_threads);
      const auto threadgroup = channels_last ? MTLSizeMake(1, 1, std::min(planes, max_threads))
                                             : MTLSizeMake(tg_width, 1, std::min(planes, max_threads / tg_width));
      auto encoder = stream->commandEncoder();
      [encoder setComputePipelineState:pso];
      mtl_setArgs(encoder, grad_input, grad_output, params);
      [encoder dispatchThreads:MTLSizeMake(width, height, planes) threadsPerThreadgroup:threadgroup];
    }
  });
}

} // anonymous namespace

TORCH_IMPL_FUNC(upsample_nearest1d_out_mps)
(const Tensor& input, IntArrayRef output_size, std::optional<double> scale, const Tensor& output) {
  upsample_kernel_out_template<3>(input, output_size, false, {scale}, output, "nearest1d");
}

TORCH_IMPL_FUNC(upsample_nearest1d_backward_out_mps)
(const Tensor& grad_output,
 IntArrayRef output_size,
 IntArrayRef input_size,
 std::optional<double> scale,
 const Tensor& grad_input) {
  upsample_gather_backward_out_template(
      grad_input.unsqueeze(2), grad_output.unsqueeze(2), false, {scale, std::nullopt}, "nearest2d");
}

TORCH_IMPL_FUNC(_upsample_nearest_exact1d_out_mps)
(const Tensor& input, IntArrayRef output_size, std::optional<double> scale, const Tensor& output) {
  upsample_kernel_out_template<3>(input, output_size, false, {scale}, output, "nearest_exact1d");
}

TORCH_IMPL_FUNC(_upsample_nearest_exact1d_backward_out_mps)
(const Tensor& grad_output,
 IntArrayRef output_size,
 IntArrayRef input_size,
 std::optional<double> scale,
 const Tensor& grad_input) {
  upsample_gather_backward_out_template(
      grad_input.unsqueeze(2), grad_output.unsqueeze(2), false, {scale, std::nullopt}, "nearest_exact2d");
}

TORCH_IMPL_FUNC(upsample_nearest2d_out_mps)
(const Tensor& input,
 IntArrayRef output_size,
 std::optional<double> scales_h,
 std::optional<double> scales_w,
 const Tensor& output) {
  upsample_kernel_out_template<4>(input, output_size, false, {scales_w, scales_h}, output, "nearest2d");
}

TORCH_IMPL_FUNC(upsample_nearest2d_backward_out_mps)
(const Tensor& grad_output,
 IntArrayRef output_size,
 IntArrayRef input_size,
 std::optional<double> scales_h,
 std::optional<double> scales_w,
 const Tensor& grad_input) {
  upsample_gather_backward_out_template(grad_input, grad_output, false, {scales_w, scales_h}, "nearest2d");
}

TORCH_IMPL_FUNC(_upsample_nearest_exact2d_out_mps)
(const Tensor& input,
 IntArrayRef output_size,
 std::optional<double> scales_h,
 std::optional<double> scales_w,
 const Tensor& output) {
  upsample_kernel_out_template<4>(input, output_size, false, {scales_w, scales_h}, output, "nearest_exact2d");
}

TORCH_IMPL_FUNC(_upsample_nearest_exact2d_backward_out_mps)
(const Tensor& grad_output,
 IntArrayRef output_size,
 IntArrayRef input_size,
 std::optional<double> scales_h,
 std::optional<double> scales_w,
 const Tensor& grad_input) {
  upsample_gather_backward_out_template(grad_input, grad_output, false, {scales_w, scales_h}, "nearest_exact2d");
}

TORCH_IMPL_FUNC(upsample_linear1d_out_mps)
(const Tensor& input, IntArrayRef output_size, bool align_corners, std::optional<double> scale, const Tensor& output) {
  upsample_kernel_out_template<3>(input, output_size, align_corners, {scale}, output, "linear1d");
}

TORCH_IMPL_FUNC(upsample_linear1d_backward_out_mps)
(const Tensor& grad_output,
 IntArrayRef output_size,
 IntArrayRef input_size,
 bool align_corners,
 std::optional<double> scale,
 const Tensor& grad_input) {
  upsample_gather_backward_out_template(
      grad_input.unsqueeze(2), grad_output.unsqueeze(2), align_corners, {scale, std::nullopt}, "bilinear2d");
}

TORCH_IMPL_FUNC(upsample_bilinear2d_out_mps)
(const Tensor& input,
 IntArrayRef output_size,
 bool align_corners,
 std::optional<double> scales_h,
 std::optional<double> scales_w,
 const Tensor& output) {
  upsample_kernel_out_template<4>(input, output_size, align_corners, {scales_w, scales_h}, output, "bilinear2d");
}

TORCH_IMPL_FUNC(upsample_bilinear2d_backward_out_mps)
(const Tensor& grad_output,
 IntArrayRef output_size,
 IntArrayRef input_size,
 bool align_corners,
 std::optional<double> scales_h,
 std::optional<double> scales_w,
 const Tensor& grad_input) {
  upsample_gather_backward_out_template(grad_input, grad_output, align_corners, {scales_w, scales_h}, "bilinear2d");
}

TORCH_IMPL_FUNC(upsample_bicubic2d_out_mps)
(const Tensor& input,
 IntArrayRef output_size,
 bool align_corners,
 std::optional<double> scales_h,
 std::optional<double> scales_w,
 const Tensor& output) {
  upsample_kernel_out_template<4>(input, output_size, align_corners, {scales_w, scales_h}, output, "bicubic2d");
}

TORCH_IMPL_FUNC(upsample_bicubic2d_backward_out_mps)
(const Tensor& grad_output,
 IntArrayRef output_size,
 IntArrayRef input_size,
 bool align_corners,
 std::optional<double> scales_h,
 std::optional<double> scales_w,
 const Tensor& grad_input) {
  upsample_kernel_backward_out_template<4>(
      grad_input, grad_output, output_size, align_corners, {scales_w, scales_h}, "bicubic2d");
}

TORCH_IMPL_FUNC(_upsample_bilinear2d_aa_out_mps)
(const Tensor& input,
 IntArrayRef output_size,
 bool align_corners,
 std::optional<double> scales_h,
 std::optional<double> scales_w,
 const Tensor& output) {
  TORCH_CHECK(at::isFloatingType(input.scalar_type()),
              "_upsample_bilineard2d_aa_out_mps only supports floating-point dtypes");
  upsample_kernel_out_template<4>(input, output_size, align_corners, {scales_w, scales_h}, output, "bilinear2d_aa");
}

TORCH_IMPL_FUNC(_upsample_bilinear2d_aa_backward_out_mps)
(const Tensor& grad_output,
 IntArrayRef output_size,
 IntArrayRef input_size,
 bool align_corners,
 std::optional<double> scales_h,
 std::optional<double> scales_w,
 const Tensor& grad_input) {
  upsample_kernel_backward_out_template<4>(
      grad_input, grad_output, output_size, align_corners, {scales_w, scales_h}, "bilinear2d_aa");
}

TORCH_IMPL_FUNC(_upsample_bicubic2d_aa_backward_out_mps)
(const Tensor& grad_output,
 IntArrayRef output_size,
 IntArrayRef input_size,
 bool align_corners,
 std::optional<double> scales_h,
 std::optional<double> scales_w,
 const Tensor& grad_input) {
  upsample_kernel_backward_out_template<4>(
      grad_input, grad_output, output_size, align_corners, {scales_w, scales_h}, "bicubic2d_aa");
}

TORCH_IMPL_FUNC(_upsample_bicubic2d_aa_out_mps)
(const Tensor& input,
 IntArrayRef output_size,
 bool align_corners,
 std::optional<double> scales_h,
 std::optional<double> scales_w,
 const Tensor& output) {
  TORCH_CHECK(at::isFloatingType(input.scalar_type()),
              "_upsample_bicubic2d_aa_out_mps only supports floating-point dtypes");
  upsample_kernel_out_template<4>(input, output_size, align_corners, {scales_w, scales_h}, output, "bicubic2d_aa");
}

TORCH_IMPL_FUNC(upsample_nearest3d_out_mps)(const Tensor& input,
                                            IntArrayRef output_size,
                                            std::optional<double> scales_d,
                                            std::optional<double> scales_h,
                                            std::optional<double> scales_w,
                                            const Tensor& output) {
  upsample_kernel_out_template<5>(input, output_size, false, {scales_w, scales_h, scales_d}, output, "nearest_3d");
}

TORCH_IMPL_FUNC(_upsample_nearest_exact3d_out_mps)(const Tensor& input,
                                                   IntArrayRef output_size,
                                                   std::optional<double> scales_d,
                                                   std::optional<double> scales_h,
                                                   std::optional<double> scales_w,
                                                   const Tensor& output) {
  upsample_kernel_out_template<5>(
      input, output_size, false, {scales_w, scales_h, scales_d}, output, "nearest_exact_3d");
}

TORCH_IMPL_FUNC(upsample_nearest3d_backward_out_mps)(const Tensor& grad_output,
                                                     IntArrayRef output_size,
                                                     IntArrayRef input_size,
                                                     std::optional<double> scales_d,
                                                     std::optional<double> scales_h,
                                                     std::optional<double> scales_w,
                                                     const Tensor& grad_input) {
  upsample_kernel_backward_out_template<5>(
      grad_input, grad_output, output_size, false, {scales_w, scales_h, scales_d}, "nearest_3d");
}

TORCH_IMPL_FUNC(_upsample_nearest_exact3d_backward_out_mps)(const Tensor& grad_output,
                                                            IntArrayRef output_size,
                                                            IntArrayRef input_size,
                                                            std::optional<double> scales_d,
                                                            std::optional<double> scales_h,
                                                            std::optional<double> scales_w,
                                                            const Tensor& grad_input) {
  upsample_kernel_backward_out_template<5>(
      grad_input, grad_output, output_size, false, {scales_w, scales_h, scales_d}, "nearest_exact_3d");
}

TORCH_IMPL_FUNC(upsample_trilinear3d_out_mps)(const Tensor& input,
                                              IntArrayRef output_size,
                                              bool align_corners,
                                              std::optional<double> scales_d,
                                              std::optional<double> scales_h,
                                              std::optional<double> scales_w,
                                              const Tensor& output) {
  upsample_kernel_out_template<5>(
      input, output_size, align_corners, {scales_w, scales_h, scales_d}, output, "trilinear");
}
TORCH_IMPL_FUNC(upsample_trilinear3d_backward_out_mps)(const Tensor& grad_output,
                                                       IntArrayRef output_size,
                                                       IntArrayRef input_size,
                                                       bool align_corners,
                                                       std::optional<double> scales_d,
                                                       std::optional<double> scales_h,
                                                       std::optional<double> scales_w,
                                                       const Tensor& grad_input) {
  upsample_kernel_backward_out_template<5>(
      grad_input, grad_output, output_size, align_corners, {scales_w, scales_h, scales_d}, "trilinear");
}

} // namespace at::native
