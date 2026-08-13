//  Copyright © 2022 Apple Inc.
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/mps/MPSDevice.h>
#include <ATen/native/CanUse32BitIndexMath.h>
#include <ATen/native/ConvUtils.h>
#include <ATen/native/mps/OperationUtils.h>
#include <ATen/native/mps/kernels/Convolution.h>
#include <ATen/native/mps/operations/ConvolutionHeuristics.h>
#include <ATen/ops/_mps_convolution_native.h>
#include <ATen/ops/_mps_convolution_transpose_native.h>
#include <ATen/ops/addmm.h>
#include <ATen/ops/constant_pad_nd.h>
#include <ATen/ops/mm.h>
#include <ATen/ops/mps_convolution_backward_native.h>
#include <ATen/ops/mps_convolution_transpose_backward_native.h>
#include <c10/util/TypeCast.h>
#include <c10/util/env.h>
#include <fmt/format.h>

#include <algorithm>
#include <limits>
#include <string_view>
#include <vector>

namespace at::native {

#ifndef PYTORCH_JIT_COMPILE_SHADERS
static auto& lib = mps::MetalShaderLibrary::getBundledLibrary();
#else
#include <ATen/native/mps/Convolution_metallib.h>
#endif

// `memory_format` selects NDHWC vs NCDHW; `use_dhwio` selects DHWIO vs OIDHW
// (caller must insert the matching in-graph weight transpose).
static void fill_conv3d_desc(MPSGraphConvolution3DOpDescriptor* descriptor_,
                             NSUInteger strideInX,
                             NSUInteger strideInY,
                             NSUInteger strideInZ,
                             NSUInteger dilationRateInX,
                             NSUInteger dilationRateInY,
                             NSUInteger dilationRateInZ,
                             NSUInteger paddingHorizontal,
                             NSUInteger paddingVertical,
                             NSUInteger paddingDepth,
                             c10::MemoryFormat memory_format,
                             bool use_dhwio,
                             NSUInteger groups) {
  descriptor_.strideInX = strideInX;
  descriptor_.strideInY = strideInY;
  descriptor_.strideInZ = strideInZ;
  descriptor_.dilationRateInX = dilationRateInX;
  descriptor_.dilationRateInY = dilationRateInY;
  descriptor_.dilationRateInZ = dilationRateInZ;

  // TODO: Program the padding style
  descriptor_.paddingStyle = MPSGraphPaddingStyleExplicit;

  descriptor_.paddingLeft = paddingHorizontal;
  descriptor_.paddingRight = paddingHorizontal;
  descriptor_.paddingTop = paddingVertical;
  descriptor_.paddingBottom = paddingVertical;
  descriptor_.paddingFront = paddingDepth;
  descriptor_.paddingBack = paddingDepth;

  descriptor_.dataLayout = (memory_format == at::MemoryFormat::ChannelsLast3d) ? MPSGraphTensorNamedDataLayoutNDHWC
                                                                               : MPSGraphTensorNamedDataLayoutNCDHW;
  descriptor_.weightsLayout = use_dhwio ? MPSGraphTensorNamedDataLayoutDHWIO : MPSGraphTensorNamedDataLayoutOIDHW;

  descriptor_.groups = groups; // not yet tested in Xcode/C++
}

// Exact-stride match: a sliced view of CL3d has CL-like strides but isn't
// NHWC-packed; the raw-buffer NDHWC path would misread it (#180984).
static bool is_packed_channels_last_3d(const Tensor& t) {
  return t.dim() == 5 &&
      t.suggest_memory_format(/*channels_last_strides_exact_match=*/true) == at::MemoryFormat::ChannelsLast3d;
}

// DHWIO costs one in-graph weight transpose per call; only worth it when
// Cin/groups is large enough and the kernel is not factorized.
static bool conv3d_dhwio_is_beneficial(IntArrayRef weight_size) {
  constexpr int64_t kMinCinPerGroup = 4; // skip first-layer Cin=3, depthwise Cin/g=1.
  constexpr int64_t kMinKernelDim = 2; // skip 1x3x3, 3x1x1, 1x1x1.
  return weight_size.size() == 5 && weight_size[1] >= kMinCinPerGroup && weight_size[2] >= kMinKernelDim &&
      weight_size[3] >= kMinKernelDim && weight_size[4] >= kMinKernelDim;
}

// Force the tensor's stride pattern to match `desc_layout`; MPSGraph's 3D
// conv path takes a slow strided route otherwise. 4D tensors pass through.
static Tensor materialize_for_conv(const Tensor& t, c10::MemoryFormat desc_layout) {
  if (desc_layout == at::MemoryFormat::ChannelsLast3d) {
    return t.contiguous(at::MemoryFormat::ChannelsLast3d);
  }
  if (t.dim() == 5) {
    return t.contiguous();
  }
  return t;
}

// CL3d needs the NDArray path for explicit NDHWC ordering; NCDHW takes the
// tensor-direct Placeholder. Caller must materialize_for_conv first.
static at::native::mps::Placeholder make_conv_placeholder(MPSGraphTensor* graphTensor,
                                                          const at::Tensor& t,
                                                          c10::MemoryFormat desc_layout) {
  if (desc_layout == at::MemoryFormat::Contiguous) {
    return at::native::mps::Placeholder(graphTensor, t);
  }
  return at::native::mps::Placeholder(graphTensor,
                                      at::native::mps::getMPSNDArray(t, at::native::mps::getMPSShape(t, desc_layout)));
}

// NDHWC copy of a 5D tensor. Packed channels-last passes through; channels <= 8
// barely fills a transpose tile, so the plain strided copy wins there.
static Tensor conv3d_to_ndhwc(const Tensor& tensor) {
  using namespace mps;
  if (tensor.is_contiguous(MemoryFormat::ChannelsLast3d)) {
    return tensor;
  }
  const auto channels = tensor.size(1);
  const int64_t spatial_size = tensor.size(2) * tensor.size(3) * tensor.size(4);
  if (channels <= 8 || spatial_size > std::numeric_limits<int32_t>::max()) {
    return tensor.contiguous(MemoryFormat::ChannelsLast3d);
  }
  const auto source = tensor.contiguous();
  auto output = at::empty(tensor.sizes(), tensor.options(), MemoryFormat::ChannelsLast3d);
  const int64_t batches = tensor.size(0);
  // 1D activations (D == H == 1) are one long X row; wide tiles keep the
  // transpose bandwidth-bound instead of threadgroup-bound.
  const bool wide = tensor.size(2) == 1 && tensor.size(3) == 1 && channels >= 32;
  const int channel_tile = wide ? 32 : (channels <= 16 ? 16 : 32);
  const int spatial_tile = wide ? 128 : (channels <= 16 ? 64 : 32);
  // vec2 loads need the buffer base 2-element aligned too
  const bool vectorized_read = spatial_size % 2 == 0 && source.storage_offset() % 2 == 0;
  const bool vectorized_write = channels % 2 == 0;
  auto pipeline = lib.getPipelineStateForFunc(fmt::format("nchw_to_nhwc_{}_{}_{}_{}_{}",
                                                          scalarToMetalTypeString(tensor),
                                                          channel_tile,
                                                          spatial_tile,
                                                          vectorized_read,
                                                          vectorized_write));

  const std::array<int32_t, 2> dimensions = {static_cast<int32_t>(channels), static_cast<int32_t>(spatial_size)};
  const auto threadgroups = MTLSizeMake(c10::metal::ceil_div(spatial_size, int64_t(spatial_tile)),
                                        c10::metal::ceil_div(channels, int64_t(channel_tile)),
                                        batches);
  auto stream = getCurrentMPSStream();
  dispatch_sync_with_rethrow(stream->queue(), ^() {
    @autoreleasepool {
      auto encoder = stream->commandEncoder();
      getMPSProfiler().beginProfileKernel(pipeline, "nchw_to_nhwc", {source}, stream);
      [encoder setComputePipelineState:pipeline];
      mtl_setArgs(encoder, source, output, dimensions);
      [encoder dispatchThreadgroups:threadgroups threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
      getMPSProfiler().endProfileKernel(pipeline, stream);
    }
  });
  return output;
}

// DHWIO weight copy; ATen's strided permute copy runs well below memory
// bandwidth for this permutation, 2-4x slower than the flat kernel.
static Tensor conv3d_weights_to_dhwio(const Tensor& weight) {
  using namespace mps;
  const auto output_channels = weight.size(0);
  const auto input_channels_per_group = weight.size(1);
  const auto kernel_depth = weight.size(2);
  const auto kernel_height = weight.size(3);
  const auto kernel_width = weight.size(4);
  auto output = at::empty({kernel_depth, kernel_height, kernel_width, input_channels_per_group, output_channels},
                          weight.options());
  const ConvWeightPermuteParams params{
      .output_channels = static_cast<uint32_t>(output_channels),
      .input_channels_per_group = static_cast<uint32_t>(input_channels_per_group),
      .kernel_height = static_cast<uint32_t>(kernel_height),
      .kernel_width = static_cast<uint32_t>(kernel_width),
      .output_channel_stride = static_cast<uint32_t>(weight.stride(0)),
      .input_channel_stride = static_cast<uint32_t>(weight.stride(1)),
      .depth_stride = static_cast<uint32_t>(weight.stride(2)),
      .height_stride = static_cast<uint32_t>(weight.stride(3)),
      .width_stride = static_cast<uint32_t>(weight.stride(4)),
  };
  auto pipeline = lib.getPipelineStateForFunc(fmt::format("conv_weight_to_dhwio_{}", scalarToMetalTypeString(weight)));
  auto stream = getCurrentMPSStream();
  dispatch_sync_with_rethrow(stream->queue(), ^() {
    @autoreleasepool {
      auto encoder = stream->commandEncoder();
      getMPSProfiler().beginProfileKernel(pipeline, "conv_weight_to_dhwio", {weight}, stream);
      [encoder setComputePipelineState:pipeline];
      mtl_setArgs(encoder, weight, output, params);
      [encoder dispatchThreads:MTLSizeMake(output_channels, input_channels_per_group, kernel_depth * kernel_height)
          threadsPerThreadgroup:MTLSizeMake(std::min<int64_t>(output_channels, 256), 1, 1)];
      getMPSProfiler().endProfileKernel(pipeline, stream);
    }
  });
  return output;
}

struct Conv3dMppSpecialization {
  std::string dtype;
  int kernel_depth, kernel_height, kernel_width;
  int stride_depth, stride_height, stride_width;
  int dilation_depth, dilation_height, dilation_width;
  int src_channels, src_width, src_height;
  bool has_bias, out_ncdhw, grouped;
};

static id<MTLComputePipelineState> conv3d_mpp_pipeline(const Conv3dMppSpecialization& specialization,
                                                       const mps::Conv3dMppTile& tile) {
  // src_width states the shape hint the entry point was compiled with: 16384
  // for conv3d entries, the wide 1D hint for the flat conv1d entries.
  if ((specialization.src_width != 16384 && specialization.src_width != conv1d_mpp_src_width_hint) ||
      specialization.src_height != 16384) {
    return nil;
  }
  auto build_name = [&](const std::string& src_channels) {
    return fmt::format("conv3d_mpp_{}_b{}_w{}_h{}_s{}_k{}{}{}_s{}{}{}_d{}{}{}_c{}_{}_{}_{}",
                       specialization.dtype,
                       tile.output_channels,
                       tile.output_width,
                       tile.output_height,
                       tile.simdgroups,
                       specialization.kernel_depth,
                       specialization.kernel_height,
                       specialization.kernel_width,
                       specialization.stride_depth,
                       specialization.stride_height,
                       specialization.stride_width,
                       specialization.dilation_depth,
                       specialization.dilation_height,
                       specialization.dilation_width,
                       src_channels,
                       specialization.has_bias ? "bias" : "nobias",
                       specialization.out_ncdhw ? "ncdhw" : "ndhwc",
                       specialization.grouped ? "grouped" : "ungrouped");
  };
  if (specialization.src_channels >= 0) {
    const auto name = build_name(std::to_string(specialization.src_channels));
    if (lib.hasFunction(name)) {
      return lib.getPipelineStateForFunc(name);
    }
  }
  const auto name = build_name("dyn");
  return lib.hasFunction(name) ? lib.getPipelineStateForFunc(name) : nil;
}

MTLComputePipelineState_t mps::Conv3dSimdTile::pipeline_state(mps::MetalShaderLibrary& library,
                                                              const std::string& dtype,
                                                              bool use_long_index) const {
  const char* prefix = use_long_index ? "conv3d_simd_long" : "conv3d_simd";
  const auto name =
      fmt::format("{}_{}_{}_{}_{}_{}", prefix, dtype, block_rows, block_columns, simdgroups_rows, simdgroups_columns);
  return library.getPipelineStateForFunc(name);
}

// Prefer direct Metal for oversized planes or a catalogued flat-tile MPP
// specialization. Takes the (N, C, 1, L) view4d tensors.
static bool conv1d_direct_metal_eligible(const Tensor& input_t,
                                         const Tensor& weight_t,
                                         bool has_bias,
                                         int64_t stride,
                                         int64_t dilation,
                                         int64_t groups,
                                         const Tensor& output_t) {
  using namespace mps;
  constexpr int64_t kInt32Max = std::numeric_limits<int32_t>::max();
  const int64_t input_channels = input_t.size(1);
  const int64_t input_length = input_t.size(3);
  const int64_t output_channels = output_t.size(1);
  const int64_t output_length = output_t.size(3);
  if (input_channels == 0 || input_channels * input_length > kInt32Max || output_channels * output_length > kInt32Max) {
    return true;
  }
  if (!has_mpp()) {
    return false;
  }
  const int64_t input_channels_per_group = input_channels / groups;
  Conv3DParams params{};
  params.conv2d.N = c10::checked_convert<int32_t>(input_t.size(0), "batch size");
  params.conv2d.C_in = c10::checked_convert<int32_t>(input_channels, "input channels");
  params.conv2d.C_out = c10::checked_convert<int32_t>(output_channels, "output channels");
  params.conv2d.H = 1;
  params.conv2d.W = c10::checked_convert<int32_t>(input_length, "input length");
  params.conv2d.outH = 1;
  params.conv2d.outW = c10::checked_convert<int32_t>(output_length, "output length");
  params.conv2d.kH = 1;
  params.conv2d.kW = c10::checked_convert<int32_t>(weight_t.size(3), "kernel size");
  params.conv2d.sH = 1;
  params.conv2d.sW = c10::checked_convert<int32_t>(stride, "stride");
  params.conv2d.dH = 1;
  params.conv2d.dW = c10::checked_convert<int32_t>(dilation, "dilation");
  params.conv2d.C_in_per_group = c10::checked_convert<int32_t>(input_channels_per_group, "input channels per group");
  params.conv2d.C_out_per_group = c10::checked_convert<int32_t>(output_channels / groups, "output channels per group");
  params.D = params.outD = params.kD = params.sD = params.dD = 1;
  Conv3dMppSpecialization specialization;
  specialization.dtype = scalarToMetalTypeString(input_t);
  specialization.kernel_depth = specialization.kernel_height = 1;
  specialization.kernel_width = params.conv2d.kW;
  specialization.stride_depth = specialization.stride_height = 1;
  specialization.stride_width = params.conv2d.sW;
  specialization.dilation_depth = specialization.dilation_height = 1;
  specialization.dilation_width = params.conv2d.dW;
  specialization.src_channels = input_channels_per_group <= 64
      ? c10::checked_convert<int32_t>(input_channels_per_group, "input channels per group")
      : -1;
  // ungrouped flat entries are compiled with the wide 1D source hint
  const int64_t width_hint = groups == 1 ? conv1d_mpp_src_width_hint : 16384;
  specialization.src_width = c10::checked_convert<int32_t>(std::max<int64_t>(input_length, width_hint), "source width");
  specialization.src_height = 16384;
  specialization.has_bias = has_bias;
  specialization.out_ncdhw = output_t.is_contiguous();
  specialization.grouped = groups > 1;
  const auto tile = select_conv3d_mpp_tile(params, groups);
  return conv3d_mpp_pipeline(specialization, tile) != nil;
}

static void conv3d_metal_launch(id<MTLComputePipelineState> pipeline,
                                const char* kernel_name,
                                const Tensor& activation,
                                const Tensor& weights,
                                const std::optional<Tensor>& bias,
                                const Tensor& output,
                                Conv3DParams params,
                                MTLSize threadgroups,
                                MTLSize threads_per_threadgroup) {
  using namespace mps;
  auto stream = getCurrentMPSStream();
  dispatch_sync_with_rethrow(stream->queue(), ^() {
    @autoreleasepool {
      auto encoder = stream->commandEncoder();
      getMPSProfiler().beginProfileKernel(pipeline, kernel_name, {activation, weights}, stream);
      [encoder setComputePipelineState:pipeline];
      mtl_setArgs(encoder, activation, weights, output, params, bias ? *bias : activation);
      [encoder dispatchThreadgroups:threadgroups threadsPerThreadgroup:threads_per_threadgroup];
      getMPSProfiler().endProfileKernel(pipeline, stream);
    }
  });
}

// conv3d forward on the Metal kernels: MPP on macOS 26.2+, simdgroup otherwise;
// planes past int32 take the long-indexed simdgroup variant on any macOS.
static void conv3d_metal_forward(const Tensor& input_t,
                                 const Tensor& weight_t,
                                 const std::optional<Tensor>& bias_opt,
                                 IntArrayRef padding,
                                 IntArrayRef stride,
                                 IntArrayRef dilation,
                                 int64_t groups,
                                 const Tensor& output_t) {
  using namespace mps;
  const auto dtype = input_t.scalar_type();
  const bool bias_defined = bias_opt && bias_opt->defined();
  const bool out_ncdhw = output_t.is_contiguous();
  TORCH_INTERNAL_ASSERT(out_ncdhw || is_packed_channels_last_3d(output_t));
  const int64_t input_channels = input_t.size(1);
  const int64_t input_depth = input_t.size(2);
  const int64_t input_height = input_t.size(3);
  const int64_t input_width = input_t.size(4);
  const int64_t output_channels = output_t.size(1);
  const int64_t output_depth = output_t.size(2);
  const int64_t output_height = output_t.size(3);
  const int64_t output_width = output_t.size(4);
  if (input_channels == 0) {
    // reduction over an empty channel set: bias (or zeros)
    if (bias_defined) {
      output_t.copy_(bias_opt->reshape({1, output_channels, 1, 1, 1}).expand_as(output_t));
    } else {
      output_t.zero_();
    }
    return;
  }
  // The MPP path addresses one (batch, depth) activation plane and one output
  // plane with int32 tensor extents; larger planes run the long-indexed
  // simdgroup kernel.
  constexpr int64_t kInt32Max = std::numeric_limits<int32_t>::max();
  const bool use_long_index = input_channels * input_height * input_width > kInt32Max ||
      output_channels * output_height * output_width > kInt32Max;
  const int64_t input_channels_per_group = input_channels / groups;
  TORCH_CHECK(weight_t.size(2) * weight_t.size(3) * weight_t.size(4) * input_channels_per_group <= kInt32Max,
              "conv3d: kernel volume times channels per group exceeds int32");
  TORCH_CHECK(
      std::max({input_depth, input_height, input_width, output_depth, output_height, output_width}) <= kInt32Max,
      "conv3d: a single spatial extent exceeds int32");
  const bool use_mpp = !use_long_index && has_mpp();

  const auto activation = conv3d_to_ndhwc(input_t); // NDHWC
  const auto weights = conv3d_weights_to_dhwio(weight_t); // DHWIO
  std::optional<Tensor> bias;
  if (bias_defined) {
    bias = bias_opt->scalar_type() == dtype ? bias_opt->contiguous() : bias_opt->to(dtype).contiguous();
  }

  const int64_t output_channels_per_group = output_channels / groups;
  Conv3DParams params;
  params.conv2d.N = static_cast<int32_t>(input_t.size(0));
  params.conv2d.C_in = static_cast<int32_t>(input_channels);
  params.conv2d.C_out = static_cast<int32_t>(output_channels);
  params.conv2d.H = static_cast<int32_t>(input_height);
  params.conv2d.W = static_cast<int32_t>(input_width);
  params.conv2d.outH = static_cast<int32_t>(output_height);
  params.conv2d.outW = static_cast<int32_t>(output_width);
  params.conv2d.kH = static_cast<int32_t>(weight_t.size(3));
  params.conv2d.kW = static_cast<int32_t>(weight_t.size(4));
  params.conv2d.sH = static_cast<int32_t>(stride[1]);
  params.conv2d.sW = static_cast<int32_t>(stride[2]);
  params.conv2d.padH = static_cast<int32_t>(padding[1]);
  params.conv2d.padW = static_cast<int32_t>(padding[2]);
  params.conv2d.dH = static_cast<int32_t>(dilation[1]);
  params.conv2d.dW = static_cast<int32_t>(dilation[2]);
  params.conv2d.C_in_per_group = static_cast<int32_t>(input_channels_per_group);
  params.conv2d.C_out_per_group = static_cast<int32_t>(output_channels_per_group);
  params.conv2d.has_bias = bias_defined;
  params.D = static_cast<int32_t>(input_depth);
  params.outD = static_cast<int32_t>(output_depth);
  params.kD = static_cast<int32_t>(weight_t.size(2));
  params.sD = static_cast<int32_t>(stride[0]);
  params.padD = static_cast<int32_t>(padding[0]);
  params.dD = static_cast<int32_t>(dilation[0]);
  params.out_ncdhw = out_ncdhw;

  const auto dtype_name = scalarToMetalTypeString(input_t);
  Conv3dMppSpecialization specialization;
  specialization.dtype = dtype_name;
  specialization.kernel_depth = params.kD;
  specialization.kernel_height = params.conv2d.kH;
  specialization.kernel_width = params.conv2d.kW;
  specialization.stride_depth = params.sD;
  specialization.stride_height = params.conv2d.sH;
  specialization.stride_width = params.conv2d.sW;
  specialization.dilation_depth = params.dD;
  specialization.dilation_height = params.conv2d.dH;
  specialization.dilation_width = params.conv2d.dW;
  specialization.src_channels = input_channels_per_group <= 64 ? static_cast<int>(input_channels_per_group) : -1;
  const bool flat1d = groups == 1 && params.conv2d.outH == 1 && params.kD == 1 && params.conv2d.kH == 1 &&
      params.sD == 1 && params.conv2d.sH == 1 && params.dD == 1 && params.conv2d.dH == 1;
  const int64_t width_hint = flat1d ? conv1d_mpp_src_width_hint : 16384;
  specialization.src_width = static_cast<int>(std::max<int64_t>(input_width, width_hint));
  specialization.src_height = static_cast<int>(std::max<int64_t>(input_height, 16384));
  specialization.has_bias = bias_defined;
  specialization.out_ncdhw = out_ncdhw;
  specialization.grouped = groups > 1;

  const auto& conv2d = params.conv2d;
  if (use_mpp) {
    const auto tile = select_conv3d_mpp_tile(params, groups);
    if (auto pipeline = conv3d_mpp_pipeline(specialization, tile)) {
      const auto output_channel_tiles =
          c10::metal::ceil_div(int64_t(conv2d.C_out_per_group), int64_t(tile.output_channels));
      const auto width_tiles = c10::metal::ceil_div(int64_t(conv2d.outW), int64_t(tile.output_width));
      const auto height_tiles = c10::metal::ceil_div(int64_t(conv2d.outH), int64_t(tile.output_height));
      const auto threadgroups = MTLSizeMake(static_cast<NSUInteger>(output_channel_tiles * groups),
                                            static_cast<NSUInteger>(width_tiles),
                                            static_cast<NSUInteger>(height_tiles) * conv2d.N * params.outD);
      const auto threads_per_threadgroup = MTLSizeMake(tile.simdgroups * 32, 1, 1);
      conv3d_metal_launch(
          pipeline, "conv3d_mpp", activation, weights, bias, output_t, params, threadgroups, threads_per_threadgroup);
      return;
    }
  }

  const auto tile = select_conv3d_simd_tile(output_height, output_width, use_long_index);
  const auto pipeline = tile.pipeline_state(lib, dtype_name, use_long_index);
  const auto output_channel_tiles = c10::metal::ceil_div(int64_t(conv2d.C_out_per_group), int64_t(tile.block_columns));
  const auto output_position_tiles =
      c10::metal::ceil_div(int64_t(conv2d.outH) * conv2d.outW, static_cast<int64_t>(tile.block_rows));
  const auto threadgroups = MTLSizeMake(static_cast<NSUInteger>(output_channel_tiles * groups),
                                        static_cast<NSUInteger>(output_position_tiles),
                                        static_cast<NSUInteger>(conv2d.N) * params.outD);
  const auto threads_per_threadgroup =
      MTLSizeMake(static_cast<NSUInteger>(tile.simdgroups_rows) * tile.simdgroups_columns * 32, 1, 1);
  conv3d_metal_launch(
      pipeline, "conv3d_simd", activation, weights, bias, output_t, params, threadgroups, threads_per_threadgroup);
}

// A 1x1x1 stride-1 no-pad conv is a per-voxel GEMM over channels; both the
// direct kernel and im2col (which trivially matches the patch-embed gate here)
// lose to a plain matmul, so peel it off first.
static bool conv3d_is_pointwise(const Tensor& weight, IntArrayRef stride, IntArrayRef padding, int64_t groups) {
  return groups == 1 && weight.size(2) == 1 && weight.size(3) == 1 && weight.size(4) == 1 && stride[0] == 1 &&
      stride[1] == 1 && stride[2] == 1 && padding[0] == 0 && padding[1] == 0 && padding[2] == 0;
}

static void conv3d_pointwise_matmul(const Tensor& input,
                                    const Tensor& weight,
                                    const std::optional<Tensor>& bias_opt,
                                    const Tensor& output) {
  const int64_t batch_size = input.size(0);
  const int64_t input_channels = input.size(1);
  const int64_t output_channels = weight.size(0);
  const int64_t input_depth = input.size(2);
  const int64_t input_height = input.size(3);
  const int64_t input_width = input.size(4);
  const int64_t spatial_size = input_depth * input_height * input_width;
  // addmm fuses bias into each batch's output-channels x spatial-size matrix
  // in the GEMM epilogue (one kernel, no separate bias pass), so it beats a
  // batched baddbmm here. Written straight into the output buffer when possible.
  const auto input_matrix = input.contiguous().reshape({batch_size, input_channels, spatial_size});
  const auto weight_matrix = weight.reshape({output_channels, input_channels});
  const bool has_bias = bias_opt && bias_opt->defined();
  std::optional<Tensor> converted_bias;
  if (has_bias) {
    const auto& bias = *bias_opt;
    converted_bias = bias.scalar_type() == output.scalar_type() ? bias : bias.to(output.scalar_type());
  }
  if (output.is_contiguous()) {
    auto output_matrix = output.view({batch_size, output_channels, spatial_size});
    const auto bias_matrix = has_bias ? converted_bias->reshape({output_channels, 1}) : Tensor();
    for (const auto batch : c10::irange(batch_size)) {
      auto batch_output = output_matrix.select(0, batch);
      has_bias ? at::addmm_out(batch_output, bias_matrix, weight_matrix, input_matrix.select(0, batch))
               : at::mm_out(batch_output, weight_matrix, input_matrix.select(0, batch));
    }
  } else {
    // Channels-last output: matmul into a fresh buffer, then permute-copy.
    auto result = weight_matrix.unsqueeze(0).expand({batch_size, output_channels, input_channels}).bmm(input_matrix);
    if (has_bias) {
      result = result.add(converted_bias->reshape({1, output_channels, 1}));
    }
    output.copy_(result.reshape({batch_size, output_channels, input_depth, input_height, input_width}));
  }
}

// A no-padding Conv1d with one output position is a matrix multiplication over
// the dilated input window.
static void conv1d_single_output_matmul(const Tensor& input,
                                        const Tensor& weight,
                                        const std::optional<Tensor>& bias_opt,
                                        int64_t dilation,
                                        const Tensor& output) {
  const int64_t batch_size = input.size(0);
  const int64_t input_channels = input.size(1);
  const int64_t output_channels = weight.size(0);
  const int64_t kernel_size = weight.size(3);
  const auto input_3d = input.contiguous().squeeze(2);
  const auto window = input_3d.as_strided({batch_size, input_channels, kernel_size},
                                          {input_3d.stride(0), input_3d.stride(1), dilation});
  const auto input_matrix = window.reshape({batch_size, input_channels * kernel_size});
  const auto weight_matrix = weight.contiguous().reshape({output_channels, input_channels * kernel_size});
  auto output_matrix = output.view({batch_size, output_channels});
  if (bias_opt && bias_opt->defined()) {
    const auto bias = bias_opt->scalar_type() == output.scalar_type() ? *bias_opt : bias_opt->to(output.scalar_type());
    at::addmm_out(output_matrix, bias, input_matrix, weight_matrix.t());
  } else {
    at::mm_out(output_matrix, input_matrix, weight_matrix.t());
  }
}

// conv as unfold(im2col) + matmul; weight OIDHW flattens to output channels x reduction size.
static void conv3d_im2col_matmul(const Tensor& input,
                                 const Tensor& weight,
                                 const std::optional<Tensor>& bias_opt,
                                 IntArrayRef stride,
                                 IntArrayRef padding,
                                 const Tensor& output) {
  const int64_t batch_size = input.size(0);
  const int64_t input_channels = input.size(1);
  const int64_t output_channels = weight.size(0);
  const int64_t kernel_depth = weight.size(2);
  const int64_t kernel_height = weight.size(3);
  const int64_t kernel_width = weight.size(4);
  const auto padded_input =
      at::constant_pad_nd(input.contiguous(), {padding[2], padding[2], padding[1], padding[1], padding[0], padding[0]});
  const auto patches = padded_input.unfold(2, kernel_depth, stride[0])
                           .unfold(3, kernel_height, stride[1])
                           .unfold(4, kernel_width, stride[2]);
  const int64_t output_depth = patches.size(2);
  const int64_t output_height = patches.size(3);
  const int64_t output_width = patches.size(4);
  const int64_t output_positions = batch_size * output_depth * output_height * output_width;
  const int64_t reduction_size = input_channels * kernel_depth * kernel_height * kernel_width;
  const auto columns = patches.permute({0, 2, 3, 4, 1, 5, 6, 7}).reshape({output_positions, reduction_size});
  auto result = columns.matmul(weight.contiguous().reshape({output_channels, reduction_size}).t());
  if (bias_opt && bias_opt->defined()) {
    result =
        result.add(bias_opt->scalar_type() == result.scalar_type() ? *bias_opt : bias_opt->to(result.scalar_type()));
  }
  output.copy_(result.reshape({batch_size, output_depth, output_height, output_width, output_channels})
                   .permute({0, 4, 1, 2, 3}));
}

static bool conv1d_indexing_fits_int32(int64_t kernel_size,
                                       int64_t stride,
                                       int64_t padding,
                                       int64_t dilation,
                                       int64_t max_output_index) {
  constexpr int64_t max_int32 = std::numeric_limits<int32_t>::max();
  const auto max_output_offset = max_output_index * stride;
  const auto max_kernel_offset = (kernel_size - 1) * dilation;
  return max_output_offset <= max_int32 && max_kernel_offset <= max_int32 &&
      max_output_offset - padding + max_kernel_offset <= max_int32;
}

static bool conv1d_padded_plane_fits_int32(const Tensor& input_t, int64_t padding) {
  constexpr int64_t max_int32 = std::numeric_limits<int32_t>::max();
  const int64_t input_channels = input_t.size(1);
  const int64_t input_length = input_t.size(3);
  if (input_channels == 0) {
    return true;
  }
  if (padding < 0 || input_length > max_int32 / input_channels) {
    return false;
  }
  return padding <= (max_int32 / input_channels - input_length) / 2;
}

static bool conv1d_dw_indexing_fits_int32(const Tensor& weight_t,
                                          int64_t stride,
                                          int64_t padding,
                                          int64_t dilation,
                                          const Tensor& output_t) {
  const auto output_length = output_t.size(3);
  const auto vectorized = stride == 1 && output_length >= conv1d_dw_outputs_per_thread;
  const auto max_output_index =
      vectorized ? c10::metal::round_up(output_length, int64_t(conv1d_dw_outputs_per_thread)) - 1 : output_length - 1;
  return conv1d_indexing_fits_int32(weight_t.size(3), stride, padding, dilation, max_output_index);
}

// Depthwise Conv1d on the (N, C, 1, L) view4d form; each vectorized thread
// computes adjacent outputs directly in NCL without a layout transform.
// (N, C, 1, L) tensors that are dense as NHWC are computationally NLC.
static bool conv1d_is_nlc(const Tensor& t) {
  return !t.is_contiguous() && t.is_contiguous(MemoryFormat::ChannelsLast);
}

static void conv1d_dw_forward(const Tensor& input_t,
                              const Tensor& weight_t,
                              const std::optional<Tensor>& bias_opt,
                              int64_t stride,
                              int64_t padding,
                              int64_t dilation,
                              const Tensor& output_t) {
  using namespace mps;
  const bool has_bias = bias_opt && bias_opt->defined();
  const auto dtype = input_t.scalar_type();
  const bool in_nlc = conv1d_is_nlc(input_t);
  const auto contiguous_input = in_nlc ? input_t : input_t.contiguous();
  const bool out_nlc = conv1d_is_nlc(output_t);
  const auto contiguous_weight = weight_t.contiguous();
  std::optional<Tensor> bias;
  if (has_bias) {
    bias = bias_opt->scalar_type() == dtype ? bias_opt->contiguous() : bias_opt->to(dtype).contiguous();
  }
  Conv1dDwParams params;
  params.input_channels = c10::checked_convert<int32_t>(input_t.size(1), "input channels");
  params.input_length = c10::checked_convert<int32_t>(input_t.size(3), "input length");
  params.output_length = c10::checked_convert<int32_t>(output_t.size(3), "output length");
  params.batch_size = c10::checked_convert<int32_t>(input_t.size(0), "batch size");
  params.kernel_size = c10::checked_convert<int32_t>(weight_t.size(3), "kernel size");
  params.stride = c10::checked_convert<int32_t>(stride, "stride");
  params.padding = c10::checked_convert<int32_t>(padding, "padding");
  params.dilation = c10::checked_convert<int32_t>(dilation, "dilation");
  const auto output_channels = c10::checked_convert<int32_t>(output_t.size(1), "output channels");
  params.channel_multiplier = output_channels / params.input_channels;
  params.in_channel_stride = in_nlc ? 1 : params.input_length;
  params.in_pos_stride = in_nlc ? params.input_channels : 1;
  params.out_channel_stride = out_nlc ? 1 : params.output_length;
  params.out_pos_stride = out_nlc ? output_channels : 1;
  params.swap_grid = out_nlc;
  params.has_bias = has_bias;
  const bool plain_ncl = !in_nlc && !out_nlc;
  const bool vectorized = plain_ncl && stride == 1 && params.output_length >= conv1d_dw_outputs_per_thread;
  const bool valid_vectorized = vectorized && padding == 0 && params.output_length % conv1d_dw_outputs_per_thread == 0;
  using namespace std::string_view_literals;
  static_assert(conv1d_dw_outputs_per_thread == 8, "Conv1d depthwise kernel names encode the output width");
  const auto variant = valid_vectorized ? "_vec8_valid"sv : (vectorized ? "_vec8"sv : ""sv);
  auto pipeline = lib.getPipelineStateForFunc(fmt::format("conv1d_dw{}_{}", variant, scalarToMetalTypeString(input_t)));
  const auto output_threads =
      vectorized ? c10::metal::ceil_div(params.output_length, conv1d_dw_outputs_per_thread) : params.output_length;
  const auto x_threads = params.swap_grid ? output_channels : output_threads;
  const auto y_threads = params.swap_grid ? output_threads : output_channels;
  const auto threads_per_threadgroup = std::min(x_threads, 256);
  auto stream = getCurrentMPSStream();
  dispatch_sync_with_rethrow(stream->queue(), ^() {
    @autoreleasepool {
      auto encoder = stream->commandEncoder();
      getMPSProfiler().beginProfileKernel(pipeline, "conv1d_dw", {contiguous_input, contiguous_weight}, stream);
      [encoder setComputePipelineState:pipeline];
      mtl_setArgs(encoder, contiguous_input, contiguous_weight, output_t, params, bias ? *bias : contiguous_input);
      [encoder dispatchThreads:MTLSizeMake(x_threads, y_threads, params.batch_size)
          threadsPerThreadgroup:MTLSizeMake(threads_per_threadgroup, 1, 1)];
      getMPSProfiler().endProfileKernel(pipeline, stream);
    }
  });
}

struct Conv1dMatmulRegion {
  int64_t out_col0;
  int64_t out_cols;
  int64_t in_col0;
  int64_t taps;
  int64_t w_tap0;
};

// Unit-stride only: output col m accumulates taps j with 0 <= m - pad + j*d < L.
// Regions are maximal col ranges whose valid tap window [j0, j1) is constant, so
// each region dispatch reads activations strictly inside [0, L).
static std::vector<Conv1dMatmulRegion> conv1d_matmul_regions(int64_t outW,
                                                             int64_t L,
                                                             int64_t pad,
                                                             int64_t k,
                                                             int64_t d) {
  std::vector<Conv1dMatmulRegion> regions;
  int64_t m = 0;
  while (m < outW) {
    const int64_t j0 = pad > m ? (pad - m + d - 1) / d : 0;
    const int64_t j1 = std::min(k, L - 1 + pad >= m ? (L - 1 + pad - m) / d + 1 : int64_t(0));
    int64_t next = outW;
    if (j0 > 0) {
      next = std::min(next, pad - (j0 - 1) * d);
    }
    if (j1 > 0) {
      next = std::min(next, L + pad - (j1 - 1) * d);
    }
    next = std::max(next, m + 1);
    // zero taps: columns read only padding, the kernel writes bias/zero
    const bool has_taps = j1 > j0;
    regions.push_back({m, next - m, has_taps ? m - pad + j0 * d : 0, has_taps ? j1 - j0 : 0, has_taps ? j0 : 0});
    if (regions.size() > size_t(conv1d_matmul_max_regions)) {
      return regions;
    }
    m = next;
  }
  return regions;
}

static std::pair<int, int> conv1d_matmul_tile(int64_t og,
                                              int64_t cg,
                                              int64_t taps,
                                              int64_t groups,
                                              int64_t outW,
                                              int64_t N) {
  static const int64_t cores = []() {
    const unsigned device_cores = at::mps::MPSDevice::getInstance()->getCoreCount();
    return device_cores > 0 ? static_cast<int64_t>(device_cores) : 16;
  }();
  // thresholds fit against per-shape sweeps: thin channels want narrow
  // tiles, heavy reductions amortize wide ones
  int bm = 64;
  if (cg <= 64 || og <= 48) {
    bm = 32;
  } else if (og >= 128 && cg * taps >= 4096) {
    bm = 128;
  }
  const int64_t col_tiles = std::max<int64_t>(1, c10::metal::ceil_div(outW, int64_t(64)));
  while (bm > 32 && c10::metal::ceil_div(og, int64_t(bm)) * groups * col_tiles * N < 2 * cores) {
    bm /= 2;
  }
  return {bm, bm >= 128 ? 4 : 2};
}

// (kW, O, I) weight copy for conv1d_matmul; per-tap slabs keep the reduction
// dim contiguous. Takes the (O, I, 1, kW) view4d weight.
static Tensor conv1d_weights_to_koc(const Tensor& weight) {
  using namespace mps;
  constexpr int64_t kInt32Max = std::numeric_limits<int32_t>::max();
  const int64_t output_channels = weight.size(0);
  const int64_t input_channels_per_group = weight.size(1);
  const int64_t kernel_width = weight.size(3);
  const auto [min_stride, max_stride] = std::minmax_element(weight.strides().begin(), weight.strides().end());
  if (weight.numel() == 0 || weight.numel() > kInt32Max || *max_stride > kInt32Max || *min_stride < -kInt32Max) {
    return weight.squeeze(2).permute({2, 0, 1}).contiguous();
  }
  auto output = at::empty({kernel_width, output_channels, input_channels_per_group}, weight.options());
  ConvWeightPermuteParams params{};
  params.output_channels = static_cast<int32_t>(output_channels);
  params.input_channels_per_group = static_cast<int32_t>(input_channels_per_group);
  params.kernel_height = 1;
  params.kernel_width = static_cast<int32_t>(kernel_width);
  params.output_channel_stride = static_cast<int32_t>(weight.stride(0));
  params.input_channel_stride = static_cast<int32_t>(weight.stride(1));
  params.width_stride = static_cast<int32_t>(weight.stride(3));
  auto pipeline = lib.getPipelineStateForFunc(fmt::format("conv_weight_to_koc_{}", scalarToMetalTypeString(weight)));
  auto stream = getCurrentMPSStream();
  dispatch_sync_with_rethrow(stream->queue(), ^() {
    @autoreleasepool {
      auto encoder = stream->commandEncoder();
      getMPSProfiler().beginProfileKernel(pipeline, "conv_weight_to_koc", {weight}, stream);
      [encoder setComputePipelineState:pipeline];
      mtl_setArgs(encoder, weight, output, params);
      [encoder dispatchThreads:MTLSizeMake(input_channels_per_group, output_channels, 1)
          threadsPerThreadgroup:MTLSizeMake(std::min<int64_t>(input_channels_per_group, 256), 1, 1)];
      getMPSProfiler().endProfileKernel(pipeline, stream);
    }
  });
  return output;
}

// Unit-stride conv1d as `taps` accumulated shifted matmuls over the activation in
// its native NCL/NLC layout; no NHWC staging pass, geometry stays runtime.
static void conv1d_matmul_forward(const Tensor& input_t,
                                  const Tensor& weight_t,
                                  const std::optional<Tensor>& bias_opt,
                                  int64_t padding,
                                  int64_t dilation,
                                  int64_t groups,
                                  const Tensor& output_t) {
  using namespace mps;
  const auto dtype = input_t.scalar_type();
  const bool out_nlc = conv1d_is_nlc(output_t);
  // one kernel layout flag serves both operands, so coerce the input to match
  const bool in_matches = out_nlc ? input_t.is_contiguous(MemoryFormat::ChannelsLast) : input_t.is_contiguous();
  const auto src =
      in_matches ? input_t : input_t.contiguous(out_nlc ? MemoryFormat::ChannelsLast : MemoryFormat::Contiguous);
  const auto weights = conv1d_weights_to_koc(weight_t);
  const bool has_bias = bias_opt && bias_opt->defined();
  std::optional<Tensor> bias;
  if (has_bias) {
    bias = bias_opt->scalar_type() == dtype ? bias_opt->contiguous() : bias_opt->to(dtype).contiguous();
  }

  const int64_t L = src.size(3);
  const int64_t outW = output_t.size(3);
  const int64_t k = weight_t.size(3);
  auto regions = conv1d_matmul_regions(outW, L, padding, k, dilation);
  std::optional<Tensor> padded;
  // Each edge region occupies a mostly-padded 64-column destination tile, so
  // its cost is matmul work; padding the input instead costs three linear passes
  // plus fixed launches. Compare both in microseconds (measured rates: ~12.5e6
  // MACs/us tensor throughput, ~1e5 bytes/us copy) and take the cheaper one.
  const int64_t og = output_t.size(1) / groups;
  const int64_t cg = src.size(1) / groups;
  const double edge_us = double(int64_t(regions.size()) - 1) * 64.0 * double(og * cg) * double(k) / 12.5e6;
  const double pad_us = 45.0 + 3.0 * double(src.size(0) * src.size(1)) * double(L) * src.element_size() / 1e5;
  const bool can_materialize_padding = conv1d_padded_plane_fits_int32(input_t, padding);
  TORCH_INTERNAL_ASSERT(can_materialize_padding || regions.size() <= size_t(conv1d_matmul_max_regions));
  if (can_materialize_padding &&
      (regions.size() > size_t(conv1d_matmul_max_regions) || (regions.size() > 3 && pad_us < edge_us))) {
    padded = at::empty({src.size(0), src.size(1), 1, L + 2 * padding},
                       src.options(),
                       out_nlc ? std::optional<MemoryFormat>(MemoryFormat::ChannelsLast) : std::nullopt);
    if (padding > 0) {
      padded->narrow(3, 0, padding).zero_();
      padded->narrow(3, L + padding, padding).zero_();
    }
    padded->narrow(3, padding, L).copy_(src);
    regions = {{0, outW, 0, k, 0}};
  }
  const auto& activation = padded ? *padded : src;

  Conv1dMatmulParams params{};
  params.C_in = c10::checked_convert<int32_t>(src.size(1), "input channels");
  params.C_out = c10::checked_convert<int32_t>(output_t.size(1), "output channels");
  params.L = c10::checked_convert<int32_t>(activation.size(3), "input length");
  params.outW_total = c10::checked_convert<int32_t>(outW, "output length");
  params.dilation = c10::checked_convert<int32_t>(dilation, "dilation");
  params.groups = c10::checked_convert<int32_t>(groups, "groups");
  params.region_count = c10::checked_convert<int32_t>(regions.size(), "regions");
  params.has_bias = has_bias;
  int64_t col_tiles = 0;
  for (const auto i : c10::irange(regions.size())) {
    auto& out = params.regions[i];
    out.out_col0 = c10::checked_convert<int32_t>(regions[i].out_col0, "output column");
    out.out_cols = c10::checked_convert<int32_t>(regions[i].out_cols, "output columns");
    out.in_col0 = c10::checked_convert<int32_t>(regions[i].in_col0, "input column");
    out.taps = c10::checked_convert<int32_t>(regions[i].taps, "taps");
    out.w_tap0 = c10::checked_convert<int32_t>(regions[i].w_tap0, "tap offset");
    out.tile0 = c10::checked_convert<int32_t>(col_tiles, "tile offset");
    col_tiles += c10::metal::ceil_div(regions[i].out_cols, int64_t(64));
  }

  // hosts without MetalPerformancePrimitives run the simdgroup_matrix twin;
  // the env knob forces it on newer hosts to proxy-validate that fallback
  static const bool force_simd = c10::utils::check_env("PYTORCH_MPS_CONV1D_FORCE_SIMD") == true;
  const bool use_mpp = at::mps::has_mpp() && !force_simd;
  const auto tile = conv1d_matmul_tile(og, cg, k, groups, outW, src.size(0));
  const int bm = use_mpp ? tile.first : std::min(tile.first, 64);
  const int nsg = tile.second;
  const int group_threads = use_mpp ? nsg * 32 : (bm >= 64 ? 128 : 64);
  const auto pipeline_name = use_mpp ? fmt::format("conv1d_matmul_{}_bm{}_bn64_s{}_{}_{}_{}",
                                                   scalarToMetalTypeString(src),
                                                   bm,
                                                   nsg,
                                                   out_nlc ? "nlc" : "ncl",
                                                   has_bias ? "bias" : "nobias",
                                                   groups > 1 ? "grouped" : "ungrouped")
                                     : fmt::format("conv1d_matmul_simd_{}_bm{}_{}_{}_{}",
                                                   scalarToMetalTypeString(src),
                                                   bm,
                                                   out_nlc ? "nlc" : "ncl",
                                                   has_bias ? "bias" : "nobias",
                                                   groups > 1 ? "grouped" : "ungrouped");
  auto pipeline = lib.getPipelineStateForFunc(pipeline_name);
  const auto o_tiles = c10::metal::ceil_div(og, int64_t(bm));
  const auto grid_y = col_tiles;
  auto stream = getCurrentMPSStream();
  dispatch_sync_with_rethrow(stream->queue(), ^() {
    @autoreleasepool {
      auto encoder = stream->commandEncoder();
      getMPSProfiler().beginProfileKernel(pipeline, "conv1d_matmul", {activation, weights}, stream);
      [encoder setComputePipelineState:pipeline];
      mtl_setArgs(encoder, activation, weights, output_t, params, bias ? *bias : weights);
      [encoder dispatchThreadgroups:MTLSizeMake(o_tiles * groups, grid_y, activation.size(0))
              threadsPerThreadgroup:MTLSizeMake(group_threads, 1, 1)];
      getMPSProfiler().endProfileKernel(pipeline, stream);
    }
  });
}

// Small reductions leave the matmul tensor units idle tap-by-tap; merge the taps
// into the K dim by materializing the k-shifted rows. Serves C_in <= 8 and
// strided low-reduction geometries the direct kernels lose on.
static void conv1d_im2col_matmul_forward(const Tensor& input_t,
                                         const Tensor& weight_t,
                                         const std::optional<Tensor>& bias_opt,
                                         int64_t stride,
                                         int64_t padding,
                                         int64_t dilation,
                                         int64_t groups,
                                         const Tensor& output_t) {
  const int64_t N = input_t.size(0);
  const int64_t C = input_t.size(1);
  const int64_t k = weight_t.size(3);
  const int64_t outW = output_t.size(3);
  auto x = input_t.contiguous().squeeze(2);
  if (padding > 0) {
    auto padded = at::zeros({N, C, x.size(2) + 2 * padding}, x.options());
    padded.narrow(2, padding, x.size(2)).copy_(x);
    x = padded;
  }
  // rows keep the (channel, tap) order, so group g's block stays contiguous
  const int64_t padded_length = x.size(2);
  const auto columns = x.as_strided({N, C, k, outW}, {C * padded_length, padded_length, dilation, stride})
                           .reshape({N, C * k, outW})
                           .unsqueeze(2);
  const auto merged_weight = weight_t.reshape({weight_t.size(0), (C / groups) * k, 1, 1});
  conv1d_matmul_forward(columns, merged_weight, bias_opt, 0, 1, groups, output_t);
}

// The tap-merged im2col matmul handles any stride/dilation/groups while its
// padded activation and merged view fit int32 addressing.
static bool conv1d_im2col_matmul_eligible(const Tensor& input_t,
                                          const Tensor& weight_t,
                                          int64_t padding,
                                          const Tensor& output_t) {
  constexpr int64_t kInt32Max = std::numeric_limits<int32_t>::max();
  const int64_t input_channels = input_t.size(1);
  const int64_t output_length = output_t.size(3);
  if (input_channels == 0 || output_length == 0 || !conv1d_padded_plane_fits_int32(input_t, padding) ||
      output_length > kInt32Max - 2 || weight_t.size(3) > kInt32Max / input_channels) {
    return false;
  }
  const int64_t merged_channels = input_channels * weight_t.size(3);
  return merged_channels <= kInt32Max / (output_length + 2) && output_t.size(1) <= kInt32Max / output_length;
}

static bool conv1d_matmul_eligible(const Tensor& input_t,
                                   const Tensor& weight_t,
                                   int64_t stride,
                                   int64_t padding,
                                   int64_t dilation,
                                   int64_t groups,
                                   const Tensor& output_t) {
  if (stride != 1) {
    return false;
  }
  constexpr int64_t kInt32Max = std::numeric_limits<int32_t>::max();
  const int64_t input_channels = input_t.size(1);
  const int64_t input_length = input_t.size(3);
  const int64_t output_channels = output_t.size(1);
  const int64_t output_length = output_t.size(3);
  const int64_t padded_length = input_length + 2 * std::max<int64_t>(dilation, 1);
  if (input_channels == 0 || output_length == 0 || input_channels > kInt32Max / padded_length ||
      output_channels > kInt32Max / output_length) {
    return false;
  }
  const int64_t og = output_channels / groups;
  const int64_t cg = input_channels / groups;
  // sub-tile groups underfill the matmul; dw or the im2col catch-all take them
  if (groups > 1 && (og < 8 || cg < 8)) {
    return false;
  }
  if (!conv1d_padded_plane_fits_int32(input_t, padding) &&
      conv1d_matmul_regions(output_length, input_length, padding, weight_t.size(3), dilation).size() >
          conv1d_matmul_max_regions) {
    return false;
  }
  return true;
}

static void fill_depthwise_conv_desc(MPSGraphDepthwiseConvolution3DOpDescriptor* descriptor_,
                                     NSUInteger strideInX,
                                     NSUInteger strideInY,
                                     NSUInteger dilationRateInX,
                                     NSUInteger dilationRateInY,
                                     NSUInteger paddingHorizontal,
                                     NSUInteger paddingVertical) {
  descriptor_.strides =
      @[ @1, [[NSNumber alloc] initWithInteger:strideInY], [[NSNumber alloc] initWithInteger:strideInX] ];
  descriptor_.dilationRates =
      @[ @1, [[NSNumber alloc] initWithInteger:dilationRateInY], [[NSNumber alloc] initWithInteger:dilationRateInX] ];

  descriptor_.paddingStyle = MPSGraphPaddingStyleExplicit;
  descriptor_.paddingValues = @[
    @0,
    @0,
    [[NSNumber alloc] initWithInteger:paddingVertical],
    [[NSNumber alloc] initWithInteger:paddingVertical],
    [[NSNumber alloc] initWithInteger:paddingHorizontal],
    [[NSNumber alloc] initWithInteger:paddingHorizontal]
  ];
  descriptor_.channelDimensionIndex = -3LL;
}

// Create convolution descriptor
static void fill_conv_desc(MPSGraphConvolution2DOpDescriptor* descriptor_,
                           NSUInteger strideInX,
                           NSUInteger strideInY,
                           NSUInteger dilationRateInX,
                           NSUInteger dilationRateInY,
                           NSUInteger paddingHorizontal,
                           NSUInteger paddingVertical,
                           c10::MemoryFormat memory_format,
                           NSUInteger groups) {
  descriptor_.strideInX = strideInX;
  descriptor_.strideInY = strideInY;
  descriptor_.dilationRateInX = dilationRateInX;
  descriptor_.dilationRateInY = dilationRateInY;

  // TODO: Program the padding style
  descriptor_.paddingStyle = MPSGraphPaddingStyleExplicit;

  descriptor_.paddingLeft = paddingHorizontal;
  descriptor_.paddingRight = paddingHorizontal;
  descriptor_.paddingTop = paddingVertical;
  descriptor_.paddingBottom = paddingVertical;

  descriptor_.dataLayout = (memory_format == at::MemoryFormat::Contiguous) ? MPSGraphTensorNamedDataLayoutNCHW
                                                                           : MPSGraphTensorNamedDataLayoutNHWC;

  // PyTorch always uses OIHW memory layout for weights
  descriptor_.weightsLayout = MPSGraphTensorNamedDataLayoutOIHW;
  descriptor_.groups = groups;
}

// Forward-only Metal conv for filter dims >= 256 (MPSGraph miscomputes those); backward is unaffected.
static Tensor mps_convolution_2d_large_kernel(const Tensor& input_t,
                                              const Tensor& weight_t,
                                              const std::optional<Tensor>& bias_opt,
                                              IntArrayRef padding,
                                              IntArrayRef stride,
                                              IntArrayRef dilation,
                                              int64_t groups) {
  using namespace mps;
  const auto input = input_t.contiguous();
  const auto weight = weight_t.contiguous();
  const bool has_bias = bias_opt && bias_opt->defined();
  const auto bias = has_bias ? bias_opt->contiguous() : Tensor();

  auto output = at::empty(conv_output_size(input.sizes(), weight.sizes(), padding, stride, dilation), input.options());
  if (output.numel() == 0) {
    return output;
  }

  Conv2DParams params;
  params.N = static_cast<int32_t>(input.size(0));
  params.C_in = static_cast<int32_t>(input.size(1));
  params.C_out = static_cast<int32_t>(weight.size(0));
  params.H = static_cast<int32_t>(input.size(2));
  params.W = static_cast<int32_t>(input.size(3));
  params.outH = static_cast<int32_t>(output.size(2));
  params.outW = static_cast<int32_t>(output.size(3));
  params.kH = static_cast<int32_t>(weight.size(2));
  params.kW = static_cast<int32_t>(weight.size(3));
  params.sH = static_cast<int32_t>(stride[0]);
  params.sW = static_cast<int32_t>(stride[1]);
  params.padH = static_cast<int32_t>(padding[0]);
  params.padW = static_cast<int32_t>(padding[1]);
  params.dH = static_cast<int32_t>(dilation[0]);
  params.dW = static_cast<int32_t>(dilation[1]);
  params.C_in_per_group = static_cast<int32_t>(weight.size(1));
  params.C_out_per_group = params.C_out / static_cast<int32_t>(groups);
  params.has_bias = has_bias;

  // ow-blocking quarters the thread count; only use it when the grid still saturates the GPU.
  const auto outRows = output.numel() / output.size(3);
  const auto blockedThreads = outRows * ((output.size(3) + 3) / 4);
  const bool blocked = blockedThreads >= 32768;
  const auto nThreads = blocked ? blockedThreads : output.numel();
  const bool i32 = canUse32BitIndexMath(input) && canUse32BitIndexMath(weight) && canUse32BitIndexMath(output);

  MPSStream* stream = getCurrentMPSStream();
  dispatch_sync_with_rethrow(stream->queue(), ^() {
    @autoreleasepool {
      auto computeEncoder = stream->commandEncoder();
      auto PSO = lib.getPipelineStateForFunc(
          fmt::format("conv2d_r{}_{}_{}", blocked ? 4 : 1, i32 ? "i32" : "i64", scalarToMetalTypeString(input)));
      [computeEncoder setComputePipelineState:PSO];
      mtl_setArgs(computeEncoder, input, weight, has_bias ? std::optional<Tensor>(bias) : std::nullopt, output, params);
      mtl_dispatch1DJob(computeEncoder, PSO, nThreads);
    }
  });
  return output;
}

static Tensor _mps_convolution_impl(const Tensor& input_t,
                                    const Tensor& weight_t,
                                    const std::optional<Tensor>& bias_opt,
                                    IntArrayRef padding,
                                    IntArrayRef stride,
                                    IntArrayRef dilation,
                                    int64_t groups,
                                    std::optional<IntArrayRef> input_shape) {
  // conv1d arrives as (N, C, 1, L) via view1d_as_2d; any H == 1 window without
  // H padding is computationally 1D and takes the Metal conv path too.
  const bool is1DConv = input_t.dim() == 4 && input_t.size(2) == 1 && weight_t.size(2) == 1 && padding[0] == 0;
  // MPSGraph 2D conv miscomputes the output once a filter spatial dim reaches 256; use a Metal kernel instead.
  if (input_t.dim() == 4 && (weight_t.size(2) >= 256 || weight_t.size(3) >= 256) &&
      (!is1DConv || input_shape.has_value())) {
    const auto outH = (input_t.size(2) + 2 * padding[0] - dilation[0] * (weight_t.size(2) - 1) - 1) / stride[0] + 1;
    const auto outW = (input_t.size(3) + 2 * padding[1] - dilation[1] * (weight_t.size(3) - 1) - 1) / stride[1] + 1;
    // Degenerate shapes fall through to the normal path's shape check.
    if (outH >= 1 && outW >= 1) {
      return mps_convolution_2d_large_kernel(input_t, weight_t, bias_opt, padding, stride, dilation, groups);
    }
  }
  constexpr auto kChannelsLast = MemoryFormat::ChannelsLast;
  constexpr auto kChannelsLast3d = MemoryFormat::ChannelsLast3d;
  constexpr auto kContiguous = MemoryFormat::Contiguous;
  const bool is_macos_15_plus = is_macos_at_least(MacOSVersion::MACOS_15_0);

  const bool is3DConv = input_t.dim() == 5;
  const auto memory_format = input_t.suggest_memory_format(/*channels_last_strides_exact_match=*/true);
  const bool is_cl_input = is_macos_15_plus && memory_format == kChannelsLast && !is3DConv;
  const auto input_suggested_layout = is_cl_input ? kChannelsLast : kContiguous;
  // Allocate output in the user-requested layout regardless of fast-path gate.
  const bool is_channels_last = mps_conv_use_channels_last(input_t, weight_t);
  const bool bias_defined = bias_opt ? bias_opt->defined() : false;

  TORCH_CHECK(isFloatingType(input_t.scalar_type()), "Convolution is supported only for Floating types");

  using namespace at::native::mps;
  CheckedFrom c = "mps_convolution";
  TensorArg input{input_t, "input", 1}, weight{weight_t, "weight", 2};
  checkAllSameType(c, {input, weight});
  checkAllSameGPU(c, {input, weight});

  const auto output_size = input_shape.has_value()
      ? input_shape->vec()
      : conv_output_size(input->sizes(), weight->sizes(), padding, stride, dilation);
  if (is1DConv) {
    constexpr int64_t max_int32 = std::numeric_limits<int32_t>::max();
    bool fits_int32 = input_t.size(0) <= max_int32 && input_t.size(1) <= max_int32 && input_t.size(3) <= max_int32 &&
        weight_t.size(0) <= max_int32 && weight_t.size(1) <= max_int32 && weight_t.size(3) <= max_int32 &&
        output_size[0] <= max_int32 && output_size[1] <= max_int32 && output_size[3] <= max_int32 &&
        stride[1] <= max_int32 && padding[1] <= max_int32 && dilation[1] <= max_int32 && groups <= max_int32;
    if (fits_int32 && output_size[3] > 0) {
      fits_int32 = conv1d_indexing_fits_int32(weight_t.size(3), stride[1], padding[1], dilation[1], output_size[3] - 1);
    }
    TORCH_CHECK(fits_int32,
                "MPS Conv1d dimensions, parameters, and index math must fit in the signed 32-bit range, but got input ",
                input_t.sizes(),
                ", weight ",
                weight_t.sizes(),
                ", stride ",
                stride[1],
                ", padding ",
                padding[1],
                ", dilation ",
                dilation[1],
                ", and groups ",
                groups);
  }
  auto output_t = at::empty(output_size,
                            input->scalar_type(),
                            std::nullopt,
                            kMPS,
                            std::nullopt,
                            is_channels_last ? (is3DConv ? kChannelsLast3d : kChannelsLast) : kContiguous);
  if (output_t.numel() == 0) {
    return output_t;
  }
  TensorArg output{output_t, "result", 0};

  // TODO: Remove me when MacOS-14 is no longer supported
  std::optional<Tensor> output_c;
  if (!is_macos_15_plus && is_channels_last) {
    output_c = at::empty_like(output_t, output_t.options().memory_format(kContiguous));
  }

  if (!is_macos_at_least(MacOSVersion::MACOS_15_1) && !is3DConv && !is1DConv) {
    // On macOS < 15.1, MPS convolution kernel does not support output channels > 2^16
    for (auto elem : output_t.sizes()) {
      TORCH_CHECK_NOT_IMPLEMENTED(elem <= (1 << 16), "Output channels > 65536 not supported at the MPS device. ");
    }
  }

  convolution_shape_check(c, input, weight, output, padding, stride, dilation, groups);

  if (is3DConv) {
    if (conv3d_is_pointwise(weight_t, stride, padding, groups)) {
      conv3d_pointwise_matmul(input_t, weight_t, bias_opt, output_t);
    } else if (conv3d_prefer_im2col(input_t, weight_t, stride, padding, dilation, groups, output_t)) {
      conv3d_im2col_matmul(input_t, weight_t, bias_opt, stride, padding, output_t);
    } else {
      conv3d_metal_forward(input_t, weight_t, bias_opt, padding, stride, dilation, groups, output_t);
    }
    return output_t;
  }

  if (is1DConv) {
    // channel multipliers ride the same kernel: output channel o reads input
    // channel o / (C_out / groups)
    const bool is_depthwise = groups > 0 && groups == input_t.size(1) && weight_t.size(0) % groups == 0;
    // A groups == 1, C_in == 1 conv only matches the depthwise pattern
    // degenerately; strided catalog geometries with enough columns to fill a
    // destination tile run faster as the direct matmul than as C_out dw passes.
    const bool c1_direct = groups == 1 && input_t.size(1) == 1 && stride[1] > 1 && output_t.size(3) >= 64 &&
        conv1d_direct_metal_eligible(input_t, weight_t, bias_defined, stride[1], dilation[1], groups, output_t);
    const bool use_depthwise_metal = is_depthwise && !c1_direct &&
        (output_t.is_contiguous() || conv1d_is_nlc(output_t)) &&
        conv1d_dw_indexing_fits_int32(weight_t, stride[1], padding[1], dilation[1], output_t);
    if (use_depthwise_metal) {
      conv1d_dw_forward(input_t, weight_t, bias_opt, stride[1], padding[1], dilation[1], output_t);
      return output_t;
    }
    if (c1_direct) {
      conv3d_metal_forward(input_t.unsqueeze(2),
                           weight_t.unsqueeze(2),
                           bias_opt,
                           {0, 0, padding[1]},
                           {1, 1, stride[1]},
                           {1, 1, dilation[1]},
                           groups,
                           output_t.unsqueeze(2));
      return output_t;
    }
    if (groups == 1 && input_t.size(1) > 0 && output_t.size(3) == 1 && padding[1] == 0) {
      conv1d_single_output_matmul(input_t, weight_t, bias_opt, dilation[1], output_t);
      return output_t;
    }
    // unit depth axis: reuse the conv3d routing (pointwise / im2col / direct)
    const auto input_3d = input_t.unsqueeze(2);
    const auto weight_3d = weight_t.unsqueeze(2);
    const auto output_3d = output_t.unsqueeze(2);
    const std::array<int64_t, 3> padding_3d{0, 0, padding[1]};
    const std::array<int64_t, 3> stride_3d{1, 1, stride[1]};
    const std::array<int64_t, 3> dilation_3d{1, 1, dilation[1]};
    if (conv3d_is_pointwise(weight_3d, stride_3d, padding_3d, groups)) {
      conv3d_pointwise_matmul(input_3d, weight_3d, bias_opt, output_3d);
      return output_t;
    }
    // A strided k == 1 conv is a pointwise matmul over the decimated input.
    if (groups == 1 && input_t.size(1) > 0 && weight_t.size(3) == 1 && padding[1] == 0 && stride[1] > 1 &&
        output_t.size(3) > 0) {
      const auto sliced = input_t.slice(3, 0, (output_t.size(3) - 1) * stride[1] + 1, stride[1]);
      conv3d_pointwise_matmul(sliced.unsqueeze(2), weight_3d, bias_opt, output_3d);
      return output_t;
    }
    if (conv1d_padded_plane_fits_int32(input_t, padding[1]) &&
        conv3d_prefer_im2col(input_3d, weight_3d, stride_3d, padding_3d, dilation_3d, groups, output_3d)) {
      conv3d_im2col_matmul(input_3d, weight_3d, bias_opt, stride_3d, padding_3d, output_3d);
      return output_t;
    }
    if (conv1d_matmul_eligible(input_t, weight_t, stride[1], padding[1], dilation[1], groups, output_t)) {
      // Tap-by-tap matmuls underfill the tensor units for tiny channel counts or
      // near-empty destination tiles; merge the taps into the K dim instead.
      if (groups == 1 && (input_t.size(1) <= 8 || output_t.size(3) <= 8) &&
          conv1d_im2col_matmul_eligible(input_t, weight_t, padding[1], output_t)) {
        conv1d_im2col_matmul_forward(input_t, weight_t, bias_opt, 1, padding[1], dilation[1], 1, output_t);
      } else {
        conv1d_matmul_forward(input_t, weight_t, bias_opt, padding[1], dilation[1], groups, output_t);
      }
      return output_t;
    }
    // strided with tiny channel counts: every direct kernel underfills, the
    // tap-merged matmul does not
    if (stride[1] > 1 && groups == 1 && input_t.size(1) <= 8 &&
        conv1d_im2col_matmul_eligible(input_t, weight_t, padding[1], output_t)) {
      conv1d_im2col_matmul_forward(input_t, weight_t, bias_opt, stride[1], padding[1], dilation[1], 1, output_t);
      return output_t;
    }
    if (conv1d_direct_metal_eligible(input_t, weight_t, bias_defined, stride[1], dilation[1], groups, output_t)) {
      conv3d_metal_forward(input_3d, weight_3d, bias_opt, padding_3d, stride_3d, dilation_3d, groups, output_3d);
      return output_t;
    }
    // grouped strided conv with a catalog per-group geometry: run each group as
    // an ungrouped conv (N == 1 keeps every sliced view dense; many groups
    // would serialize into underfilled dispatches)
    if (groups > 1 && groups <= 8 && input_t.size(0) == 1 && input_t.size(1) % groups == 0) {
      const int64_t group_in = input_t.size(1) / groups;
      const int64_t group_out = output_t.size(1) / groups;
      const auto input_g0 = input_t.narrow(1, 0, group_in);
      const auto weight_g0 = weight_t.narrow(0, 0, group_out);
      const auto output_g0 = output_t.narrow(1, 0, group_out);
      // a channel-narrowed slice of a channels-last output is not NDHWC-packed,
      // so the ndhwc kernels cannot write it; those shapes take the catch-all
      const bool sliced_output_ok = output_g0.is_contiguous() || is_packed_channels_last_3d(output_g0.unsqueeze(2));
      if (group_out > 0 && sliced_output_ok &&
          conv1d_direct_metal_eligible(input_g0, weight_g0, bias_defined, stride[1], dilation[1], 1, output_g0)) {
        for (const auto g : c10::irange(groups)) {
          const auto bias_g =
              bias_defined ? std::optional<Tensor>(bias_opt->narrow(0, g * group_out, group_out)) : std::nullopt;
          conv3d_metal_forward(input_t.narrow(1, g * group_in, group_in).unsqueeze(2),
                               weight_t.narrow(0, g * group_out, group_out).unsqueeze(2),
                               bias_g,
                               padding_3d,
                               stride_3d,
                               dilation_3d,
                               1,
                               output_t.narrow(1, g * group_out, group_out).unsqueeze(2));
        }
        return output_t;
      }
    }
    // Tap-merged catch-all for remaining shapes whose temporary fits int32;
    // available on every macOS tier through the matmul kernels.
    if (input_t.size(1) % groups == 0 && conv1d_im2col_matmul_eligible(input_t, weight_t, padding[1], output_t)) {
      conv1d_im2col_matmul_forward(input_t, weight_t, bias_opt, stride[1], padding[1], dilation[1], groups, output_t);
      return output_t;
    }
    // The normal convolution caller does not supply input_shape; return here
    // instead of reaching the MPSGraph fallback used by ConvTranspose backward-input.
    if (!input_shape.has_value()) {
      // Direct indexing handles a logical im2col element count over INT32_MAX
      // without materializing it.
      conv3d_metal_forward(input_3d, weight_3d, bias_opt, padding_3d, stride_3d, dilation_3d, groups, output_3d);
      return output_t;
    }
    if (!is_macos_at_least(MacOSVersion::MACOS_15_1)) {
      for (auto elem : output_t.sizes()) {
        TORCH_CHECK_NOT_IMPLEMENTED(elem <= (1 << 16), "Output channels > 65536 not supported at the MPS device. ");
      }
    }
  }

  // Derive from MPSCachedGraph
  struct CachedGraph : public MPSCachedGraph {
    CachedGraph(MPSGraph* graph) : MPSCachedGraph(graph) {}
    MPSGraphTensor* inputTensor_ = nil;
    MPSGraphTensor* biasTensor_ = nil;
    MPSGraphTensor* weightTensor_ = nil;
    MPSGraphTensor* outputTensor_ = nil;
  };

  auto stream = at::mps::getCurrentMPSStream();

  @autoreleasepool {
    IntArrayRef bias_shape;
    if (bias_defined)
      bias_shape = bias_opt.value().sizes();

    std::string bias_shape_key;
    if (bias_defined) {
      bias_shape_key = std::to_string(bias_shape[0]);
    } else {
      bias_shape_key = "nobias";
    }

    std::string key = fmt::format("mps_convolution:{}:{}:{}:{}:{}:{}:{}:{}",
                                  getArrayRefString(stride),
                                  getArrayRefString(dilation),
                                  getArrayRefString(padding),
                                  groups,
                                  is_cl_input,
                                  mps::getTensorsStringKey({input_t, weight_t}),
                                  bias_defined,
                                  bias_shape_key);

    auto inputShape = mps::getMPSShape(input_t, input_suggested_layout);
    auto cachedGraph = LookUpOrCreateCachedGraph<CachedGraph>(key, [&](auto mpsGraph, auto newCachedGraph) {
      // 2D only past this point (3D returned above); depthwise conv2d is
      // expressed via the graph's 3D depthwise op with a dummy depth dim
      bool isDepthwiseConv =
          (groups > 1 && weight_t.size(1) == 1) && input_t.dim() >= 4 && weight_t.dim() >= 4 && !is_channels_last;

      auto inputTensor = mpsGraphRankedPlaceHolder(mpsGraph, getMPSScalarType(input_t), inputShape);
      auto weightTensor = mpsGraphRankedPlaceHolder(mpsGraph, weight_t);
      MPSGraphTensor* outputTensor = nil;
      if (isDepthwiseConv) {
        auto depthWiseConv3dDescriptor_ = [[MPSGraphDepthwiseConvolution3DOpDescriptor new] autorelease];
        fill_depthwise_conv_desc(
            depthWiseConv3dDescriptor_, stride[1], stride[0], dilation[1], dilation[0], padding[1], padding[0]);

        MPSGraphTensor* weightTransposeTensor = [mpsGraph transposeTensor:weightTensor
                                                                dimension:-3
                                                            withDimension:-4
                                                                     name:nil];
        outputTensor = [mpsGraph depthwiseConvolution3DWithSourceTensor:inputTensor
                                                          weightsTensor:weightTransposeTensor
                                                             descriptor:depthWiseConv3dDescriptor_
                                                                   name:nil];
      } else {
        MPSGraphConvolution2DOpDescriptor* conv2dDescriptor_ = [[MPSGraphConvolution2DOpDescriptor new] autorelease];
        fill_conv_desc(conv2dDescriptor_,
                       stride[1],
                       stride[0],
                       dilation[1],
                       dilation[0],
                       padding[1],
                       padding[0],
                       input_suggested_layout,
                       groups);

        outputTensor = [mpsGraph convolution2DWithSourceTensor:inputTensor
                                                 weightsTensor:weightTensor
                                                    descriptor:conv2dDescriptor_
                                                          name:nil];
      }

      MPSGraphTensor* biasTensor = nil;
      if (bias_defined) {
        biasTensor = mpsGraphUnrankedPlaceHolder(mpsGraph, getMPSDataType(bias_opt.value()));
        outputTensor = [mpsGraph additionWithPrimaryTensor:outputTensor secondaryTensor:biasTensor name:nil];
      }
      newCachedGraph->inputTensor_ = inputTensor;
      newCachedGraph->weightTensor_ = weightTensor;
      newCachedGraph->biasTensor_ = biasTensor;
      newCachedGraph->outputTensor_ = outputTensor;
    });

    const auto input_for_graph =
        output_c ? input_t.contiguous() : materialize_for_conv(input_t, input_suggested_layout);
    auto inputPlaceholder = make_conv_placeholder(cachedGraph->inputTensor_, input_for_graph, input_suggested_layout);
    auto outputPlaceholder = output_c
        ? Placeholder(cachedGraph->outputTensor_, *output_c)
        : make_conv_placeholder(cachedGraph->outputTensor_, output_t, input_suggested_layout);
    // MPSGraph conv miscomputes for non-dense (offset/gapped) weight views; gather instead of using the strided API.
    auto weightsPlaceholder =
        Placeholder(cachedGraph->weightTensor_, weight_t, nil, true, MPSDataTypeInvalid, /*useMPSStridedAPI=*/false);
    auto biasPlaceholder = Placeholder();
    // Reshape the bias to be broadcastable with output of conv2d or conv3d
    if (bias_defined) {
      const int64_t C = bias_shape[0];
      const auto bias_view =
          input_suggested_layout == kChannelsLast ? std::vector<int64_t>{1, 1, 1, C} : std::vector<int64_t>{1, C, 1, 1};
      biasPlaceholder = Placeholder(cachedGraph->biasTensor_, bias_opt->view(bias_view));
    }

    auto feeds = [[[NSMutableDictionary alloc] initWithCapacity:3] autorelease];
    feeds[inputPlaceholder.getMPSGraphTensor()] = inputPlaceholder.getMPSGraphTensorData();
    feeds[weightsPlaceholder.getMPSGraphTensor()] = weightsPlaceholder.getMPSGraphTensorData();
    if (bias_defined) {
      feeds[biasPlaceholder.getMPSGraphTensor()] = biasPlaceholder.getMPSGraphTensorData();
    }

    runMPSGraph(stream, cachedGraph->graph(), feeds, outputPlaceholder);
  }

  if (output_c) {
    output_t.copy_(*output_c);
  }

  return output_t;
}

Tensor _mps_convolution(const Tensor& input_t,
                        const Tensor& weight_t,
                        const std::optional<Tensor>& bias_opt,
                        IntArrayRef padding,
                        IntArrayRef stride,
                        IntArrayRef dilation,
                        int64_t groups) {
  return _mps_convolution_impl(input_t, weight_t, bias_opt, padding, stride, dilation, groups, std::nullopt);
}

static Tensor mps_convolution_backward_input(IntArrayRef input_size,
                                             const Tensor& grad_output_t,
                                             const Tensor& weight_t,
                                             IntArrayRef padding,
                                             IntArrayRef stride,
                                             IntArrayRef dilation,
                                             int64_t groups,
                                             bool bias_defined) {
  using namespace at::native::mps;
  using namespace mps;
  bool is3DConv = grad_output_t.dim() == 5;
  if (!is_macos_at_least(MacOSVersion::MACOS_15_1)) {
    // On macOS < 15.1, MPS convolution kernel does not support output channels > 2^16
    for (auto elem : grad_output_t.sizes()) {
      TORCH_CHECK_NOT_IMPLEMENTED(elem <= (1 << 16), "Output channels > 65536 not supported at the MPS device. ");
    }
  }

  TORCH_CHECK(isFloatingType(grad_output_t.scalar_type()), "Convolution is supported only for Floating types");
  CheckedFrom c = "mps_convolution_backward_input";
  TensorArg grad_output{grad_output_t, "grad_output", 1}, weight{weight_t, "weight", 2};
  checkAllSameType(c, {grad_output, weight});
  checkAllSameGPU(c, {grad_output, weight});
  constexpr auto kChannelsLast = at::MemoryFormat::ChannelsLast;
  constexpr auto kChannelsLast3d = at::MemoryFormat::ChannelsLast3d;
  constexpr auto kContiguous = at::MemoryFormat::Contiguous;
  const bool is_macos_15_plus = is_macos_at_least(MacOSVersion::MACOS_15_0);
  // Backward uses NDHWC+DHWIO only when the full fast path is beneficial; for
  // factorized kernels / small Cin / depthwise the NCDHW+OIDHW fallback wins.
  const bool use_dhwio = is3DConv && is_macos_15_plus && is_packed_channels_last_3d(grad_output_t) &&
      conv3d_dhwio_is_beneficial(weight_t.sizes());
  const auto desc_layout = use_dhwio ? kChannelsLast3d : kContiguous;
  // Allocate grad_input in the user-requested layout. The fast path writes
  // directly; the NCDHW fallback writes via a contig scratch + copy below.
  const bool is_channels_last = mps_conv_use_channels_last(grad_output_t, weight_t);
  auto grad_input_t =
      at::empty(input_size,
                grad_output_t.options(),
                is_channels_last ? std::optional(is3DConv ? kChannelsLast3d : kChannelsLast) : std::nullopt);

  // Avoid "grad_input" when this is being used as transposed convolution
  TensorArg grad_input{grad_input_t, "result", 0};
  convolution_shape_check(c, grad_input, weight, grad_output, padding, stride, dilation, groups);

  // Contig scratch when graph emits NCDHW but grad_input is CL3d -- covers
  // the macOS-14 fallback and the 3D NCDHW fallback on macOS 15+.
  std::optional<Tensor> grad_input_c;
  const bool needs_contig_scratch = is_channels_last && (!is_macos_15_plus || (is3DConv && !use_dhwio));
  if (needs_contig_scratch) {
    grad_input_c = at::empty_like(grad_input_t, grad_input_t.options().memory_format(MemoryFormat::Contiguous));
  }

  // Derive from MPSCachedGraph
  struct CachedGraph : public MPSCachedGraph {
    CachedGraph(MPSGraph* graph) : MPSCachedGraph(graph) {}
    MPSGraphTensor* gradOutputTensor_ = nil;
    MPSGraphTensor* weightTensor_ = nil;
    MPSGraphTensor* gradInputTensor_ = nil;
  };

  // Add backward with input
  @autoreleasepool {
    MPSStream* stream = getCurrentMPSStream();
    MPSShape* mps_input_shape = getMPSShape(input_size, desc_layout);
    std::string key = fmt::format("mps_{}_convolution_backward_input:{}:{}:{}:{}:{}:{}:{}",
                                  is3DConv ? "3d_" : "",
                                  getArrayRefString(stride),
                                  getArrayRefString(dilation),
                                  getArrayRefString(padding),
                                  groups,
                                  is_channels_last,
                                  use_dhwio,
                                  getTensorsStringKey({grad_output_t, weight_t}));
    auto cachedGraph = LookUpOrCreateCachedGraph<CachedGraph>(key, [&](auto mpsGraph, auto newCachedGraph) {
      auto gradOutputShape = getMPSShape(grad_output_t, desc_layout);
      auto gradOutputTensor = mpsGraphRankedPlaceHolder(mpsGraph, getMPSScalarType(grad_output_t), gradOutputShape);
      auto weightTensor = mpsGraphRankedPlaceHolder(mpsGraph, weight_t);

      MPSGraphTensor* gradInputTensor;
      MPSShape* weightOutputShape = mps::getMPSShape(weight_t);
      // Depthwise conv is input feature channels = groups. So I in OIHW has to be 1.
      bool isDepthwiseConv = ((groups > 1 && (weightOutputShape[1].intValue == 1)) && grad_output_t.ndimension() >= 4 &&
                              weightOutputShape.count >= 4 && !is_channels_last);

      if (is3DConv) {
        MPSGraphConvolution3DOpDescriptor* conv3dDescriptor_ = [[MPSGraphConvolution3DOpDescriptor new] autorelease];
        fill_conv3d_desc(conv3dDescriptor_,
                         stride[2],
                         stride[1],
                         stride[0],
                         dilation[2],
                         dilation[1],
                         dilation[0],
                         padding[2],
                         padding[1],
                         padding[0],
                         desc_layout,
                         use_dhwio,
                         groups);
        MPSGraphTensor* convWeightTensor = use_dhwio
            ? [mpsGraph transposeTensor:weightTensor permutation:@[ @2, @3, @4, @1, @0 ] name:nil]
            : weightTensor;
        gradInputTensor = [mpsGraph convolution3DDataGradientWithIncomingGradientTensor:gradOutputTensor
                                                                          weightsTensor:convWeightTensor
                                                                            outputShape:mps_input_shape
                                                           forwardConvolutionDescriptor:conv3dDescriptor_
                                                                                   name:nil];
      } else if (isDepthwiseConv) {
        MPSGraphDepthwiseConvolution3DOpDescriptor* depthWiseConv3dDescriptor_ =
            [[MPSGraphDepthwiseConvolution3DOpDescriptor new] autorelease];
        fill_depthwise_conv_desc(
            depthWiseConv3dDescriptor_, stride[1], stride[0], dilation[1], dilation[0], padding[1], padding[0]);
        MPSGraphTensor* weightTransposeTensor = [mpsGraph transposeTensor:weightTensor
                                                                dimension:-3
                                                            withDimension:-4
                                                                     name:nil];
        gradInputTensor =
            [mpsGraph depthwiseConvolution3DDataGradientWithIncomingGradientTensor:gradOutputTensor
                                                                     weightsTensor:weightTransposeTensor
                                                                       outputShape:mps_input_shape
                                                                        descriptor:depthWiseConv3dDescriptor_
                                                                              name:nil];
      } else {
        MPSGraphConvolution2DOpDescriptor* conv2dDescriptor_ = [[MPSGraphConvolution2DOpDescriptor new] autorelease];
        fill_conv_desc(conv2dDescriptor_,
                       stride[1],
                       stride[0],
                       dilation[1],
                       dilation[0],
                       padding[1],
                       padding[0],
                       at::MemoryFormat::Contiguous,
                       groups);

        gradInputTensor = [mpsGraph convolution2DDataGradientWithIncomingGradientTensor:gradOutputTensor
                                                                          weightsTensor:weightTensor
                                                                            outputShape:mps_input_shape
                                                           forwardConvolutionDescriptor:conv2dDescriptor_
                                                                                   name:nil];
      }

      newCachedGraph->gradOutputTensor_ = gradOutputTensor;
      newCachedGraph->weightTensor_ = weightTensor;
      newCachedGraph->gradInputTensor_ = gradInputTensor;
    });

    const auto grad_out_for_graph =
        grad_input_c ? grad_output_t.contiguous() : materialize_for_conv(grad_output_t, desc_layout);
    auto gradOutputPlaceholder = make_conv_placeholder(cachedGraph->gradOutputTensor_, grad_out_for_graph, desc_layout);
    // MPSGraph conv miscomputes for non-dense (offset/gapped) weight views; gather instead of using the strided API.
    auto weightsPlaceholder =
        Placeholder(cachedGraph->weightTensor_, weight_t, nil, true, MPSDataTypeInvalid, /*useMPSStridedAPI=*/false);
    auto outputPlaceholder = grad_input_c
        ? Placeholder(cachedGraph->gradInputTensor_, *grad_input_c)
        : make_conv_placeholder(cachedGraph->gradInputTensor_, grad_input_t, desc_layout);

    auto feeds = dictionaryFromPlaceholders(gradOutputPlaceholder, weightsPlaceholder);
    runMPSGraph(stream, cachedGraph->graph(), feeds, outputPlaceholder);
  }
  if (grad_input_c) {
    grad_input_t.copy_(*grad_input_c);
  }
  return grad_input_t;
}

static Tensor mps_convolution_backward_weights(IntArrayRef weight_size,
                                               const Tensor& grad_output_t,
                                               const Tensor& input_t,
                                               IntArrayRef padding,
                                               IntArrayRef stride,
                                               IntArrayRef dilation,
                                               int64_t groups,
                                               bool bias_defined) {
  using namespace at::native::mps;
  using namespace mps;
  const bool is3DConv = input_t.dim() == 5;
  TORCH_CHECK(isFloatingType(grad_output_t.scalar_type()), "Convolution is supported only for Floating types");
  CheckedFrom c = "mps_convolution_backward_weights";
  constexpr auto kChannelsLast = at::MemoryFormat::ChannelsLast;
  constexpr auto kChannelsLast3d = at::MemoryFormat::ChannelsLast3d;
  constexpr auto kContiguous = at::MemoryFormat::Contiguous;
  const bool is_macos_15_plus = is_macos_at_least(MacOSVersion::MACOS_15_0);
  // Half-precision WG regresses on NDHWC+DHWIO; force NCDHW+OIDHW.
  const bool half_precision_wg =
      grad_output_t.scalar_type() == at::kBFloat16 || grad_output_t.scalar_type() == at::kHalf;
  // Require BOTH inputs CL3d-packed; otherwise we'd permute the non-packed one each call.
  const bool use_dhwio = is3DConv && is_macos_15_plus && !half_precision_wg && is_packed_channels_last_3d(input_t) &&
      is_packed_channels_last_3d(grad_output_t) && conv3d_dhwio_is_beneficial(weight_size);
  const auto desc_layout = use_dhwio ? kChannelsLast3d : kContiguous;
  // grad_weight allocation: 2D follows the standard CL convention; 3D always
  // stays contiguous OIDHW (the graph already transposes DHWIO -> OIDHW).
  const bool allocate_grad_weight_cl = mps_conv_use_channels_last(input_t, grad_output_t) && !is3DConv;

  // For uniformity with everything else, although it seems grad_weight
  // would be unambiguous too.
  TensorArg grad_output{grad_output_t, "grad_output", 1};
  TensorArg input{input_t, "input", 2};

  checkAllSameType(c, {grad_output, input});
  checkAllSameGPU(c, {grad_output, input});

  auto grad_weight_t = at::empty(
      weight_size, grad_output_t.options(), allocate_grad_weight_cl ? std::optional(kChannelsLast) : std::nullopt);

  TensorArg grad_weight{grad_weight_t, "result", 0};

  convolution_shape_check(c, input, grad_weight, grad_output, padding, stride, dilation, groups);

  // Derive from MPSCachedGraph
  struct CachedGraph : public MPSCachedGraph {
    CachedGraph(MPSGraph* graph) : MPSCachedGraph(graph) {}
    MPSGraphTensor* gradOutputTensor_ = nil;
    MPSGraphTensor* inputTensor_ = nil;
    MPSGraphTensor* gradWeightTensor_ = nil;
  };

  // TODO: Remove me when MacOS-14 is no longer supported
  std::optional<Tensor> grad_weight_c;
  if (!is_macos_at_least(MacOSVersion::MACOS_15_0) && allocate_grad_weight_cl) {
    grad_weight_c = at::empty_like(grad_weight_t, grad_weight_t.options().memory_format(MemoryFormat::Contiguous));
  }

  @autoreleasepool {
    MPSStream* stream = getCurrentMPSStream();

    // Under DHWIO the graph emits weight grad in DHWIO order; the op output
    // shape must match, and we transpose back to OIDHW after.
    MPSShape* mps_weight_shape = use_dhwio
        ? @[ @(weight_size[2]), @(weight_size[3]), @(weight_size[4]), @(weight_size[1]), @(weight_size[0]) ]
        : getMPSShape(weight_size);
    std::string key = fmt::format("mps_{}convolution_backward_weights:{}:{}:{}:{}:{}:{}:{}",
                                  is3DConv ? "3d_" : "",
                                  getArrayRefString(stride),
                                  getArrayRefString(dilation),
                                  getArrayRefString(padding),
                                  groups,
                                  allocate_grad_weight_cl,
                                  use_dhwio,
                                  getTensorsStringKey({grad_output_t, input_t, grad_weight_t}));
    auto cachedGraph = LookUpOrCreateCachedGraph<CachedGraph>(key, [&](auto mpsGraph, auto newCachedGraph) {
      MPSShape* inputShape = getMPSShape(input_t, desc_layout);
      MPSShape* gradOutputShape = getMPSShape(grad_output_t, desc_layout);
      // For the non-CL path the depthwise heuristic inspects the OIHW weight shape.
      MPSShape* weight_shape_OIDHW = getMPSShape(weight_size);
      bool isDepthwiseConv = ((groups > 1 && (weight_shape_OIDHW[1].intValue == 1)) && inputShape.count >= 4 &&
                              weight_shape_OIDHW.count >= 4);

      MPSGraphTensor* gradOutputTensor =
          mpsGraphRankedPlaceHolder(mpsGraph, getMPSScalarType(grad_output_t), gradOutputShape);
      MPSGraphTensor* inputTensor = mpsGraphRankedPlaceHolder(mpsGraph, getMPSScalarType(input_t), inputShape);

      MPSGraphTensor* gradWeightTensor;
      if (is3DConv) {
        MPSGraphConvolution3DOpDescriptor* conv3dDescriptor_ = [[MPSGraphConvolution3DOpDescriptor new] autorelease];
        fill_conv3d_desc(conv3dDescriptor_,
                         stride[2],
                         stride[1],
                         stride[0],
                         dilation[2],
                         dilation[1],
                         dilation[0],
                         padding[2],
                         padding[1],
                         padding[0],
                         desc_layout,
                         use_dhwio,
                         groups);
        gradWeightTensor = [mpsGraph convolution3DWeightsGradientWithIncomingGradientTensor:gradOutputTensor
                                                                               sourceTensor:inputTensor
                                                                                outputShape:mps_weight_shape
                                                               forwardConvolutionDescriptor:conv3dDescriptor_
                                                                                       name:nil];
        if (use_dhwio) {
          gradWeightTensor = [mpsGraph transposeTensor:gradWeightTensor permutation:@[ @4, @3, @0, @1, @2 ] name:nil];
        }
      } else if (isDepthwiseConv) {
        MPSGraphDepthwiseConvolution3DOpDescriptor* depthWiseConv3dDescriptor_ =
            [[MPSGraphDepthwiseConvolution3DOpDescriptor new] autorelease];
        fill_depthwise_conv_desc(
            depthWiseConv3dDescriptor_, stride[1], stride[0], dilation[1], dilation[0], padding[1], padding[0]);
        NSNumber* outputFeatChannelDim = mps_weight_shape[0];
        MPSShape* weightShapeTranspose = @[ @1, outputFeatChannelDim, mps_weight_shape[2], mps_weight_shape[3] ];
        MPSGraphTensor* gradWeightTensorTranspose =
            [mpsGraph depthwiseConvolution3DWeightsGradientWithIncomingGradientTensor:gradOutputTensor
                                                                         sourceTensor:inputTensor
                                                                          outputShape:weightShapeTranspose
                                                                           descriptor:depthWiseConv3dDescriptor_
                                                                                 name:nil];
        gradWeightTensor = [mpsGraph transposeTensor:gradWeightTensorTranspose dimension:-3 withDimension:-4 name:nil];
      } else {
        MPSGraphConvolution2DOpDescriptor* conv2dDescriptor_ = [[MPSGraphConvolution2DOpDescriptor new] autorelease];
        fill_conv_desc(conv2dDescriptor_,
                       stride[1],
                       stride[0],
                       dilation[1],
                       dilation[0],
                       padding[1],
                       padding[0],
                       at::MemoryFormat::Contiguous,
                       groups);

        gradWeightTensor = [mpsGraph convolution2DWeightsGradientWithIncomingGradientTensor:gradOutputTensor
                                                                               sourceTensor:inputTensor
                                                                                outputShape:mps_weight_shape
                                                               forwardConvolutionDescriptor:conv2dDescriptor_
                                                                                       name:nil];
      }

      newCachedGraph->gradOutputTensor_ = gradOutputTensor;
      newCachedGraph->inputTensor_ = inputTensor;
      newCachedGraph->gradWeightTensor_ = gradWeightTensor;
    });

    const auto grad_out_for_graph =
        grad_weight_c ? grad_output_t.contiguous() : materialize_for_conv(grad_output_t, desc_layout);
    const auto input_for_graph = grad_weight_c ? input_t.contiguous() : materialize_for_conv(input_t, desc_layout);
    auto gradOutputPlaceholder = make_conv_placeholder(cachedGraph->gradOutputTensor_, grad_out_for_graph, desc_layout);
    auto inputPlaceholder = make_conv_placeholder(cachedGraph->inputTensor_, input_for_graph, desc_layout);
    auto outputPlaceholder =
        Placeholder(cachedGraph->gradWeightTensor_, grad_weight_c ? *grad_weight_c : grad_weight_t);

    auto feeds = dictionaryFromPlaceholders(gradOutputPlaceholder, inputPlaceholder);
    runMPSGraph(stream, cachedGraph->graph(), feeds, outputPlaceholder);
  }

  if (grad_weight_c) {
    grad_weight_t.copy_(*grad_weight_c);
  }
  return grad_weight_t;
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> mps_convolution_backward(const at::Tensor& input,
                                                                        const at::Tensor& grad_output,
                                                                        const at::Tensor& weight,
                                                                        IntArrayRef padding,
                                                                        IntArrayRef stride,
                                                                        IntArrayRef dilation,
                                                                        int64_t groups,
                                                                        std::array<bool, 3> output_mask) {
  Tensor grad_input, grad_weight, grad_bias;
  if (input.numel() == 0) {
    if (output_mask[0]) {
      grad_input = at::empty_like(input, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
    }
    if (output_mask[1]) {
      grad_weight = at::zeros_like(weight, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
    }
  } else {
    if (output_mask[0]) {
      grad_input = mps_convolution_backward_input(
          input.sizes(), grad_output, weight, padding, stride, dilation, groups, output_mask[2]);
    }
    if (output_mask[1]) {
      grad_weight = mps_convolution_backward_weights(
          weight.sizes(), grad_output, input, padding, stride, dilation, groups, output_mask[2]);
    }
  }

  return std::tuple<Tensor, Tensor, Tensor>{grad_input, grad_weight, grad_bias};
}

static Tensor mps_convolution_transpose_forward(const Tensor& grad_output,
                                                const Tensor& weight,
                                                IntArrayRef padding,
                                                IntArrayRef output_padding,
                                                IntArrayRef stride,
                                                IntArrayRef dilation,
                                                int64_t groups) {
  auto input_size =
      conv_input_size(grad_output.sizes(), weight.sizes(), padding, output_padding, stride, dilation, groups);
  return mps_convolution_backward_input(input_size, grad_output, weight, padding, stride, dilation, groups, false);
}

Tensor _mps_convolution_transpose(const Tensor& input_t,
                                  const Tensor& weight_t,
                                  IntArrayRef padding,
                                  IntArrayRef output_padding,
                                  IntArrayRef stride,
                                  IntArrayRef dilation,
                                  int64_t groups) {
  bool is_unsupported_3d_dtype =
      (input_t.dim() == 5 && (input_t.scalar_type() == kHalf || input_t.scalar_type() == kBFloat16));
  TORCH_CHECK(!is_unsupported_3d_dtype, "ConvTranspose 3D with BF16 or FP16 types is not supported on MPS");

  auto output_t =
      mps_convolution_transpose_forward(input_t, weight_t, padding, output_padding, stride, dilation, groups);
  return output_t;
}

static Tensor mps_convolution_transpose_backward_input(const Tensor& grad_output_t,
                                                       const Tensor& weight_t,
                                                       IntArrayRef padding,
                                                       IntArrayRef stride,
                                                       IntArrayRef dilation,
                                                       int64_t groups,
                                                       IntArrayRef input_shape) {
  return _mps_convolution_impl(grad_output_t, weight_t, std::nullopt, padding, stride, dilation, groups, input_shape);
}

static Tensor mps_convolution_transpose_backward_weight(IntArrayRef weight_size,
                                                        const Tensor& grad_output_t,
                                                        const Tensor& input_t,
                                                        IntArrayRef padding,
                                                        IntArrayRef stride,
                                                        IntArrayRef dilation,
                                                        int64_t groups) {
  return mps_convolution_backward_weights(
      weight_size, input_t, grad_output_t, padding, stride, dilation, groups, false);
}

std::tuple<Tensor, Tensor> mps_convolution_transpose_backward(const Tensor& input,
                                                              const Tensor& grad_output,
                                                              const Tensor& weight,
                                                              IntArrayRef padding,
                                                              IntArrayRef output_padding,
                                                              IntArrayRef stride,
                                                              IntArrayRef dilation,
                                                              int64_t groups,
                                                              std::array<bool, 2> output_mask) {
  Tensor grad_input, grad_weight;
  if (output_mask[0]) {
    grad_input =
        mps_convolution_transpose_backward_input(grad_output, weight, padding, stride, dilation, groups, input.sizes());
  }
  if (output_mask[1]) {
    grad_weight = mps_convolution_transpose_backward_weight(
        weight.sizes(), grad_output, input, padding, stride, dilation, groups);
  }

  return std::tuple<Tensor, Tensor>{grad_input, grad_weight};
}

} // namespace at::native
