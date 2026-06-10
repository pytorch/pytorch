//  Copyright © 2023 Apple Inc.
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/native/UpSample.h>
#include <ATen/native/mps/OperationUtils.h>
#include <fmt/format.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
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
namespace mps {

#ifndef PYTORCH_JIT_COMPILE_SHADERS
static auto& lib = mps::MetalShaderLibrary::getBundledLibrary();
#else
#include <ATen/native/mps/UpSample_metallib.h>
#endif

// see NOTE [ Nearest neighbor upsampling kernel implementation ]
template <typename accscalar_t>
static accscalar_t compute_scales_value_backwards(const std::optional<double> scale,
                                                  int64_t src_size,
                                                  int64_t dst_size) {
  // FIXME: remove magic > 0 after we ensure no models were serialized with -1 defaults.
  return (scale.value_or(0.) > 0.) ? (accscalar_t)scale.value() : (accscalar_t)src_size / dst_size;
}

template <typename accscalar_t>
static accscalar_t area_pixel_compute_scale(int input_size,
                                            int output_size,
                                            bool align_corners,
                                            const std::optional<double> scale) {
  if (align_corners) {
    if (output_size > 1) {
      return (accscalar_t)(input_size - 1) / (output_size - 1);
    } else {
      return static_cast<accscalar_t>(0);
    }
  } else {
    return compute_scales_value<accscalar_t>(scale, input_size, output_size);
  }
}

static void upsample_kernel_out_template(const Tensor& input,
                                         IntArrayRef output_size,
                                         bool align_corners,
                                         std::optional<double> scale_h_opt,
                                         std::optional<double> scale_w_opt,
                                         const Tensor& output,
                                         const std::string& name) {
  if (output.numel() == 0) {
    return;
  }
  std::array<float, 2> scales = {
      area_pixel_compute_scale<float>(input.size(-1), output.size(-1), align_corners, scale_w_opt),
      area_pixel_compute_scale<float>(input.size(2), output.size(2), align_corners, scale_h_opt)};
  auto upsamplePSO = lib.getPipelineStateForFunc(fmt::format("upsample_{}_{}", name, scalarToMetalTypeString(input)));
  auto stream = getCurrentMPSStream();
  dispatch_sync_with_rethrow(stream->queue(), ^() {
    @autoreleasepool {
      auto computeEncoder = stream->commandEncoder();
      [computeEncoder setComputePipelineState:upsamplePSO];
      mtl_setArgs(computeEncoder,
                  input,
                  output,
                  input.strides(),
                  output.strides(),
                  input.sizes(),
                  output.sizes(),
                  scales,
                  align_corners);
      if (output.ndimension() == 4) {
        mtl_dispatch1DJob(computeEncoder, upsamplePSO, output_size[0] * output_size[1]);
      } else {
        mtl_dispatch1DJob(computeEncoder, upsamplePSO, output_size[0]);
      }
    }
  });
}

static void upsample_kernel_out_template(const Tensor& input,
                                         IntArrayRef output_size,
                                         bool align_corners,
                                         std::optional<double> scale_d_opt,
                                         std::optional<double> scale_h_opt,
                                         std::optional<double> scale_w_opt,
                                         const Tensor& output,
                                         const std::string& name) {
  if (output.numel() == 0) {
    return;
  }
  UpsampleParams<5> params;
  memcpy(params.input_sizes.data(), input.sizes().data(), 5 * sizeof(long));
  memcpy(params.input_strides.data(), input.strides().data(), 5 * sizeof(long));
  memcpy(params.output_strides.data(), output.strides().data(), 5 * sizeof(long));
  memcpy(params.output_sizes.data(), output.sizes().data(), 5 * sizeof(long));
  params.scales[0] = area_pixel_compute_scale<float>(input.size(4), output.size(4), align_corners, scale_w_opt);
  params.scales[1] = area_pixel_compute_scale<float>(input.size(3), output.size(3), align_corners, scale_h_opt);
  params.scales[2] = area_pixel_compute_scale<float>(input.size(2), output.size(2), align_corners, scale_d_opt);
  params.align_corners = align_corners;
  auto upsamplePSO = lib.getPipelineStateForFunc(fmt::format("upsample_{}_{}", name, scalarToMetalTypeString(input)));
  auto stream = getCurrentMPSStream();
  dispatch_sync_with_rethrow(stream->queue(), ^() {
    @autoreleasepool {
      auto computeEncoder = stream->commandEncoder();
      [computeEncoder setComputePipelineState:upsamplePSO];
      mtl_setArgs(computeEncoder, input, output, params);
      mtl_dispatch1DJob(computeEncoder, upsamplePSO, output_size[0] * output_size[1] * output_size[2]);
    }
  });
}

static void upsample_kernel_backward_out_template(const Tensor& grad_input,
                                                  const Tensor& grad_output,
                                                  IntArrayRef output_size,
                                                  IntArrayRef input_size,
                                                  bool align_corners,
                                                  std::optional<double> scale_d_opt,
                                                  std::optional<double> scale_h_opt,
                                                  std::optional<double> scale_w_opt,
                                                  const std::string& name) {
  grad_input.zero_();
  if (grad_output.numel() == 0) {
    return;
  }
  // See note in the 1D/2D backward template: accumulate reduced precision in
  // fp32 to avoid losing the signal across many scattered atomic adds.
  const auto scalar_type = grad_input.scalar_type();
  const bool low_precision = scalar_type == at::kHalf || scalar_type == at::kBFloat16;
  const Tensor grad_in =
      low_precision ? grad_input.new_zeros(grad_input.sizes(), grad_input.options().dtype(at::kFloat)) : grad_input;
  const Tensor grad_out = low_precision ? grad_output.to(at::kFloat) : grad_output;
  auto upsamplePSO =
      lib.getPipelineStateForFunc(fmt::format("upsample_{}_backward_{}", name, scalarToMetalTypeString(grad_in)));
  UpsampleParams<5> params;
  memcpy(params.input_sizes.data(), grad_in.sizes().data(), 5 * sizeof(long));
  memcpy(params.input_strides.data(), grad_in.strides().data(), 5 * sizeof(long));
  memcpy(params.output_strides.data(), grad_out.strides().data(), 5 * sizeof(long));
  memcpy(params.output_sizes.data(), grad_out.sizes().data(), 5 * sizeof(long));
  params.scales[0] = area_pixel_compute_scale<float>(grad_in.size(4), grad_out.size(4), align_corners, scale_w_opt);
  params.scales[1] = area_pixel_compute_scale<float>(grad_in.size(3), grad_out.size(3), align_corners, scale_h_opt);
  params.scales[2] = area_pixel_compute_scale<float>(grad_in.size(2), grad_out.size(2), align_corners, scale_d_opt);
  params.align_corners = align_corners;
  auto stream = getCurrentMPSStream();
  dispatch_sync_with_rethrow(stream->queue(), ^() {
    @autoreleasepool {
      auto computeEncoder = stream->commandEncoder();
      [computeEncoder setComputePipelineState:upsamplePSO];
      mtl_setArgs(computeEncoder, grad_in, grad_out, params);
      mtl_dispatch1DJob(computeEncoder, upsamplePSO, output_size[0] * output_size[1] * output_size[2]);
    }
  });
  if (low_precision) {
    grad_input.copy_(grad_in);
  }
}

// 1D/2D backward dispatcher. Picks the grid and scales based on dimensionality,
// matching the forward upsample_kernel_out_template.
static void upsample_kernel_backward_out_template(const Tensor& grad_input,
                                                  const Tensor& grad_output,
                                                  IntArrayRef output_size,
                                                  IntArrayRef input_size,
                                                  bool align_corners,
                                                  std::optional<double> scale_h_opt,
                                                  std::optional<double> scale_w_opt,
                                                  const std::string& name) {
  grad_input.zero_();
  if (grad_output.numel() == 0) {
    return;
  }
  // Many output pixels scatter into a single input pixel via atomic adds.
  // Accumulating those in fp16/bf16 loses the signal at large scale factors, so
  // for reduced precision accumulate into an fp32 buffer and cast back, matching
  // the fp32 acc_type the CPU/CUDA kernels use.
  const auto scalar_type = grad_input.scalar_type();
  const bool low_precision = scalar_type == at::kHalf || scalar_type == at::kBFloat16;
  const Tensor grad_in =
      low_precision ? grad_input.new_zeros(grad_input.sizes(), grad_input.options().dtype(at::kFloat)) : grad_input;
  const Tensor grad_out = low_precision ? grad_output.to(at::kFloat) : grad_output;
  const bool is_2d = grad_in.ndimension() == 4;
  std::array<float, 2> scales = {
      area_pixel_compute_scale<float>(grad_in.size(-1), grad_out.size(-1), align_corners, scale_w_opt),
      is_2d ? area_pixel_compute_scale<float>(grad_in.size(2), grad_out.size(2), align_corners, scale_h_opt)
            : 0.0f};
  auto upsamplePSO = lib.getPipelineStateForFunc(
      fmt::format("upsample_{}_backward_{}", name, mps::scalarToMetalTypeString(grad_in)));
  auto stream = getCurrentMPSStream();
  dispatch_sync_with_rethrow(stream->queue(), ^() {
    @autoreleasepool {
      auto computeEncoder = stream->commandEncoder();
      [computeEncoder setComputePipelineState:upsamplePSO];
      mtl_setArgs(computeEncoder,
                  grad_in,
                  grad_out,
                  grad_in.strides(),
                  grad_out.strides(),
                  grad_in.sizes(),
                  grad_out.sizes(),
                  scales,
                  align_corners);
      mtl_dispatch1DJob(
          computeEncoder, upsamplePSO, is_2d ? output_size[0] * output_size[1] : output_size[0]);
    }
  });
  if (low_precision) {
    grad_input.copy_(grad_in);
  }
}

// 1D/2D gather backward dispatcher: one thread per input pixel. The kernel sums
// every contributing output in an fp32 register and writes once, so it needs no
// atomics, no fp32 scratch, and is deterministic.
static void upsample_kernel_backward_gather_out_template(const Tensor& grad_input,
                                                         const Tensor& grad_output,
                                                         bool align_corners,
                                                         std::optional<double> scale_h_opt,
                                                         std::optional<double> scale_w_opt,
                                                         const std::string& name) {
  if (grad_output.numel() == 0) {
    grad_input.zero_();
    return;
  }
  const bool is_2d = grad_input.ndimension() == 4;
  std::array<float, 2> scales = {
      area_pixel_compute_scale<float>(grad_input.size(-1), grad_output.size(-1), align_corners, scale_w_opt),
      is_2d ? area_pixel_compute_scale<float>(grad_input.size(2), grad_output.size(2), align_corners, scale_h_opt)
            : 0.0f};
  auto upsamplePSO = lib.getPipelineStateForFunc(
      fmt::format("upsample_{}_backward_{}", name, mps::scalarToMetalTypeString(grad_input)));
  const int64_t num_threads = is_2d ? grad_input.size(-2) * grad_input.size(-1) : grad_input.size(-1);
  auto stream = getCurrentMPSStream();
  dispatch_sync_with_rethrow(stream->queue(), ^() {
    @autoreleasepool {
      auto computeEncoder = stream->commandEncoder();
      [computeEncoder setComputePipelineState:upsamplePSO];
      mtl_setArgs(computeEncoder,
                  grad_input,
                  grad_output,
                  grad_input.strides(),
                  grad_output.strides(),
                  grad_input.sizes(),
                  grad_output.sizes(),
                  scales,
                  align_corners);
      mtl_dispatch1DJob(computeEncoder, upsamplePSO, num_threads);
    }
  });
}

} // namespace mps

TORCH_IMPL_FUNC(upsample_nearest1d_out_mps)
(const Tensor& input, IntArrayRef output_size, std::optional<double> scale, const Tensor& output) {
  mps::upsample_kernel_out_template(input, output_size, false, scale, scale, output, "nearest1d");
}

TORCH_IMPL_FUNC(upsample_nearest1d_backward_out_mps)
(const Tensor& grad_output,
 IntArrayRef output_size,
 IntArrayRef input_size,
 std::optional<double> scale,
 const Tensor& grad_input) {
  mps::upsample_kernel_backward_gather_out_template(grad_input, grad_output, false, scale, scale, "nearest1d");
}

TORCH_IMPL_FUNC(_upsample_nearest_exact1d_out_mps)
(const Tensor& input, IntArrayRef output_size, std::optional<double> scale, const Tensor& output) {
  mps::upsample_kernel_out_template(input, output_size, false, scale, scale, output, "nearest_exact1d");
}

TORCH_IMPL_FUNC(_upsample_nearest_exact1d_backward_out_mps)
(const Tensor& grad_output,
 IntArrayRef output_size,
 IntArrayRef input_size,
 std::optional<double> scale,
 const Tensor& grad_input) {
  mps::upsample_kernel_backward_gather_out_template(grad_input, grad_output, false, scale, scale, "nearest_exact1d");
}

TORCH_IMPL_FUNC(upsample_nearest2d_out_mps)
(const Tensor& input,
 IntArrayRef output_size,
 std::optional<double> scales_h,
 std::optional<double> scales_w,
 const Tensor& output) {
  mps::upsample_kernel_out_template(input, output_size, false, scales_h, scales_w, output, "nearest2d");
}

TORCH_IMPL_FUNC(upsample_nearest2d_backward_out_mps)
(const Tensor& grad_output,
 IntArrayRef output_size,
 IntArrayRef input_size,
 std::optional<double> scales_h,
 std::optional<double> scales_w,
 const Tensor& grad_input) {
  mps::upsample_kernel_backward_gather_out_template(grad_input, grad_output, false, scales_h, scales_w, "nearest2d");
}

TORCH_IMPL_FUNC(_upsample_nearest_exact2d_out_mps)
(const Tensor& input,
 IntArrayRef output_size,
 std::optional<double> scales_h,
 std::optional<double> scales_w,
 const Tensor& output) {
  mps::upsample_kernel_out_template(input, output_size, false, scales_h, scales_w, output, "nearest_exact2d");
}

TORCH_IMPL_FUNC(_upsample_nearest_exact2d_backward_out_mps)
(const Tensor& grad_output,
 IntArrayRef output_size,
 IntArrayRef input_size,
 std::optional<double> scales_h,
 std::optional<double> scales_w,
 const Tensor& grad_input) {
  mps::upsample_kernel_backward_gather_out_template(
      grad_input, grad_output, false, scales_h, scales_w, "nearest_exact2d");
}

TORCH_IMPL_FUNC(upsample_linear1d_out_mps)
(const Tensor& input, IntArrayRef output_size, bool align_corners, std::optional<double> scale, const Tensor& output) {
  mps::upsample_kernel_out_template(input, output_size, align_corners, scale, scale, output, "linear1d");
}

TORCH_IMPL_FUNC(upsample_linear1d_backward_out_mps)
(const Tensor& grad_output,
 IntArrayRef output_size,
 IntArrayRef input_size,
 bool align_corners,
 std::optional<double> scale,
 const Tensor& grad_input) {
  mps::upsample_kernel_backward_gather_out_template(grad_input, grad_output, align_corners, scale, scale, "linear1d");
}

TORCH_IMPL_FUNC(upsample_bilinear2d_out_mps)
(const Tensor& input,
 IntArrayRef output_size,
 bool align_corners,
 std::optional<double> scales_h,
 std::optional<double> scales_w,
 const Tensor& output) {
  mps::upsample_kernel_out_template(input, output_size, align_corners, scales_h, scales_w, output, "bilinear2d");
}

TORCH_IMPL_FUNC(upsample_bilinear2d_backward_out_mps)
(const Tensor& grad_output,
 IntArrayRef output_size,
 IntArrayRef input_size,
 bool align_corners,
 std::optional<double> scales_h,
 std::optional<double> scales_w,
 const Tensor& grad_input) {
  mps::upsample_kernel_backward_gather_out_template(
      grad_input, grad_output, align_corners, scales_h, scales_w, "bilinear2d");
}

TORCH_IMPL_FUNC(upsample_bicubic2d_out_mps)
(const Tensor& input,
 IntArrayRef output_size,
 bool align_corners,
 std::optional<double> scales_h,
 std::optional<double> scales_w,
 const Tensor& output) {
  mps::upsample_kernel_out_template(input, output_size, align_corners, scales_h, scales_w, output, "bicubic2d");
}

TORCH_IMPL_FUNC(upsample_bicubic2d_backward_out_mps)
(const Tensor& grad_output,
 IntArrayRef output_size,
 IntArrayRef input_size,
 bool align_corners,
 std::optional<double> scales_h,
 std::optional<double> scales_w,
 const Tensor& grad_input) {
  mps::upsample_kernel_backward_out_template(
      grad_input, grad_output, output_size, input_size, align_corners, scales_h, scales_w, "bicubic2d");
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
  mps::upsample_kernel_out_template(input, output_size, align_corners, scales_h, scales_w, output, "bilinear2d_aa");
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
  mps::upsample_kernel_out_template(input, output_size, align_corners, scales_h, scales_w, output, "bicubic2d_aa");
}

TORCH_IMPL_FUNC(upsample_nearest3d_out_mps)(const Tensor& input,
                                            IntArrayRef output_size,
                                            std::optional<double> scales_d,
                                            std::optional<double> scales_h,
                                            std::optional<double> scales_w,
                                            const Tensor& output) {
  mps::upsample_kernel_out_template(input, output_size, false, scales_d, scales_h, scales_w, output, "nearest_3d");
}

TORCH_IMPL_FUNC(_upsample_nearest_exact3d_out_mps)(const Tensor& input,
                                                   IntArrayRef output_size,
                                                   std::optional<double> scales_d,
                                                   std::optional<double> scales_h,
                                                   std::optional<double> scales_w,
                                                   const Tensor& output) {
  mps::upsample_kernel_out_template(
      input, output_size, false, scales_d, scales_h, scales_w, output, "nearest_exact_3d");
}

TORCH_IMPL_FUNC(upsample_nearest3d_backward_out_mps)(const Tensor& grad_output,
                                                     IntArrayRef output_size,
                                                     IntArrayRef input_size,
                                                     std::optional<double> scales_d,
                                                     std::optional<double> scales_h,
                                                     std::optional<double> scales_w,
                                                     const Tensor& grad_input) {
  mps::upsample_kernel_backward_out_template(
      grad_input, grad_output, output_size, input_size, false, scales_d, scales_h, scales_w, "nearest_3d");
}

TORCH_IMPL_FUNC(_upsample_nearest_exact3d_backward_out_mps)(const Tensor& grad_output,
                                                            IntArrayRef output_size,
                                                            IntArrayRef input_size,
                                                            std::optional<double> scales_d,
                                                            std::optional<double> scales_h,
                                                            std::optional<double> scales_w,
                                                            const Tensor& grad_input) {
  mps::upsample_kernel_backward_out_template(
      grad_input, grad_output, output_size, input_size, false, scales_d, scales_h, scales_w, "nearest_exact_3d");
}

TORCH_IMPL_FUNC(upsample_trilinear3d_out_mps)(const Tensor& input,
                                              IntArrayRef output_size,
                                              bool align_corners,
                                              std::optional<double> scales_d,
                                              std::optional<double> scales_h,
                                              std::optional<double> scales_w,
                                              const Tensor& output) {
  mps::upsample_kernel_out_template(
      input, output_size, align_corners, scales_d, scales_h, scales_w, output, "trilinear");
}
TORCH_IMPL_FUNC(upsample_trilinear3d_backward_out_mps)(const Tensor& grad_output,
                                                       IntArrayRef output_size,
                                                       IntArrayRef input_size,
                                                       bool align_corners,
                                                       std::optional<double> scales_d,
                                                       std::optional<double> scales_h,
                                                       std::optional<double> scales_w,
                                                       const Tensor& grad_input) {
  mps::upsample_kernel_backward_out_template(
      grad_input, grad_output, output_size, input_size, align_corners, scales_d, scales_h, scales_w, "trilinear");
}

} // namespace at::native
