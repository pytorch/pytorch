//  Copyright © 2023 Apple Inc.
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/native/UpSample.h>
#include <ATen/native/mps/MPSGraphVenturaOps.h>
#include <ATen/native/mps/OperationUtils.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_upsample_nearest_exact1d.h>
#include <ATen/ops/_upsample_nearest_exact1d_backward.h>
#include <ATen/ops/_upsample_nearest_exact1d_backward_native.h>
#include <ATen/ops/_upsample_nearest_exact1d_native.h>
#include <ATen/ops/_upsample_nearest_exact2d.h>
#include <ATen/ops/_upsample_nearest_exact2d_backward.h>
#include <ATen/ops/_upsample_nearest_exact2d_backward_native.h>
#include <ATen/ops/_upsample_nearest_exact2d_native.h>
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
#endif
namespace at::native {
namespace mps {

// Upsampling operations (1D/2D forward and backward)
// supported resize_mode: 'nearest' | 'bilinear' | 'nearest-exact'
static void upsample_out_template(const Tensor& input,
                                  IntArrayRef output_size,
                                  std::optional<IntArrayRef> input_size_opt, // only used for backward pass
                                  std::optional<double> scale_h_opt,
                                  std::optional<double> scale_w_opt,
                                  const Tensor& output,
                                  bool align_corners,
                                  const c10::string_view resize_mode_str) {
  if (input.numel() == 0) {
    return;
  }
  const auto input_dim = input.sizes();
  if (input_dim.size() <= 3) {
    native::upsample_1d_common_check(input.sizes(), output_size);
  } else {
    native::upsample_2d_common_check(input.sizes(), output_size);
  }
  Tensor out;
  if (needsGather(output)) {
    out = at::empty_like(output, MemoryFormat::Contiguous);
  }

  bool centerResults = false;
  MPSGraphResizeMode resizeMode = MPSGraphResizeNearest;
  MPSGraphResizeNearestRoundingMode nearestRoundingMode = MPSGraphResizeNearestRoundingModeFloor;
  MPSGraphTensorNamedDataLayout dataLayout =
      input_dim.size() > 3 ? MPSGraphTensorNamedDataLayoutNCHW : MPSGraphTensorNamedDataLayoutCHW;
  if (resize_mode_str == "nearest") {
    resizeMode = MPSGraphResizeNearest;
  } else if (resize_mode_str == "bilinear") {
    resizeMode = MPSGraphResizeBilinear;
    centerResults = true;
  } else if (resize_mode_str == "nearest-exact") {
    centerResults = true;
    nearestRoundingMode = MPSGraphResizeNearestRoundingModeRoundPreferCeil;
  } else {
    TORCH_CHECK(false, "Unsupported resize mode ", resize_mode_str);
  }

  const int64_t output_width = output_size.size() > 1 ? output_size[1] : output_size[0];
  const int64_t output_height = output_size.size() > 1 ? output_size[0] : (output.dim() > 2 ? output.size(-2) : 1);
  const float scale_w = (scale_w_opt.value_or(0.) > 0.) ? static_cast<float>(scale_w_opt.value()) : 0.;
  const float scale_h = (scale_h_opt.value_or(0.) > 0.) ? static_cast<float>(scale_h_opt.value()) : 1.;
  const float offset_y = centerResults ? (scale_h - 1.0f) / 2.0f : 0.0f;
  const float offset_x = centerResults ? (scale_w - 1.0f) / 2.0f : 0.0f;

  IntArrayRef input_size;
  const bool is_backward_pass = input_size_opt.has_value();
  if (is_backward_pass) {
    input_size = input_size_opt.value();
  }
  struct CachedGraph : public MPSCachedGraph {
    CachedGraph(MPSGraph* graph) : MPSCachedGraph(graph) {}
    MPSGraphTensor *inputTensor = nil, *outputTensor = nil;
    MPSGraphTensor* outputSizeTensor = nil;
  };
  MPSStream* stream = getCurrentMPSStream();

  @autoreleasepool {
    string key = "upsample_" + std::string(resize_mode_str) + (align_corners ? "_aligned_corners" : "") +
        getTensorsStringKey({input}) + ":[" + std::to_string(scale_h) + "," + std::to_string(scale_w) + "]:[" +
        (is_backward_pass ? getArrayRefString(input_size) : "Undefined") + "]";

    auto cachedGraph = LookUpOrCreateCachedGraph<CachedGraph>(key, [&](auto mpsGraph, auto newCachedGraph) {
      newCachedGraph->inputTensor = mpsGraphRankedPlaceHolder(mpsGraph, input);
      newCachedGraph->outputSizeTensor = mpsGraphRankedPlaceHolder(mpsGraph, MPSDataTypeInt32, @[ @(2) ]);

      MPSGraphTensor* scaleOffsetTensor = nullptr;
      MPSGraphTensor* inputSizeTensor = nullptr;

      if (scale_w > 0.0) {
        const float outScales[4] = {scale_h, scale_w, offset_y, offset_x};
        scaleOffsetTensor = [mpsGraph constantWithData:[NSData dataWithBytes:outScales length:sizeof(outScales)]
                                                 shape:@[ @4 ]
                                              dataType:MPSDataTypeFloat32];
      }
      if (is_backward_pass) {
        std::vector<NSNumber*> inputSizeVec(4);
        inputSizeVec[0] = @(input_size[0]);
        inputSizeVec[1] = @(input_size[1]);
        inputSizeVec[2] = @(input_size[2]);
        inputSizeVec[3] = @(input_dim.size() > 3 ? input_size[3] : 1);
        inputSizeTensor = [mpsGraph constantWithScalar:0
                                                 shape:[NSArray arrayWithObjects:inputSizeVec.data()
                                                                           count:input_dim.size()]
                                              dataType:getMPSDataType(input)];
      }
      if (!is_backward_pass) {
        if (scaleOffsetTensor && !align_corners) {
          if (resizeMode == MPSGraphResizeNearest) {
            newCachedGraph->outputTensor = [mpsGraph resizeNearestWithTensor:newCachedGraph->inputTensor
                                                                  sizeTensor:newCachedGraph->outputSizeTensor
                                                           scaleOffsetTensor:scaleOffsetTensor
                                                         nearestRoundingMode:nearestRoundingMode
                                                                      layout:dataLayout
                                                                        name:nil];
          } else { // bilinear forward
            newCachedGraph->outputTensor = [mpsGraph resizeBilinearWithTensor:newCachedGraph->inputTensor
                                                                   sizeTensor:newCachedGraph->outputSizeTensor
                                                            scaleOffsetTensor:scaleOffsetTensor
                                                                       layout:dataLayout
                                                                         name:nil];
          }
        } else { // scaleOffsetTensor == nil || align_corners
          if (resizeMode == MPSGraphResizeNearest) {
            newCachedGraph->outputTensor = [mpsGraph resizeNearestWithTensor:newCachedGraph->inputTensor
                                                                  sizeTensor:newCachedGraph->outputSizeTensor
                                                         nearestRoundingMode:nearestRoundingMode
                                                                centerResult:centerResults
                                                                alignCorners:align_corners
                                                                      layout:dataLayout
                                                                        name:nil];
          } else { // bilinear forward
            newCachedGraph->outputTensor = [mpsGraph resizeBilinearWithTensor:newCachedGraph->inputTensor
                                                                   sizeTensor:newCachedGraph->outputSizeTensor
                                                                 centerResult:centerResults
                                                                 alignCorners:align_corners
                                                                       layout:dataLayout
                                                                         name:nil];
          }
        }
      } else { // is_backward_pass == true
        if (scaleOffsetTensor && !align_corners) {
          if (resizeMode == MPSGraphResizeNearest) {
            newCachedGraph->outputTensor = [mpsGraph resizeNearestWithGradientTensor:newCachedGraph->inputTensor
                                                                               input:inputSizeTensor
                                                                   scaleOffsetTensor:scaleOffsetTensor
                                                                 nearestRoundingMode:nearestRoundingMode
                                                                              layout:dataLayout
                                                                                name:nil];
          } else { // bilinear backward
            newCachedGraph->outputTensor = [mpsGraph resizeBilinearWithGradientTensor:newCachedGraph->inputTensor
                                                                                input:inputSizeTensor
                                                                    scaleOffsetTensor:scaleOffsetTensor
                                                                               layout:dataLayout
                                                                                 name:nil];
          }
        } else { // scaleOffsetTensor == nil || align_corners
          if (resizeMode == MPSGraphResizeNearest) {
            newCachedGraph->outputTensor = [mpsGraph resizeNearestWithGradientTensor:newCachedGraph->inputTensor
                                                                               input:inputSizeTensor
                                                                 nearestRoundingMode:nearestRoundingMode
                                                                        centerResult:centerResults
                                                                        alignCorners:align_corners
                                                                              layout:dataLayout
                                                                                name:nil];
          } else { // bilinear backward
            newCachedGraph->outputTensor = [mpsGraph resizeBilinearWithGradientTensor:newCachedGraph->inputTensor
                                                                                input:inputSizeTensor
                                                                         centerResult:centerResults
                                                                         alignCorners:align_corners
                                                                               layout:dataLayout
                                                                                 name:nil];
          }
        }
      }
    });
    MPSNDArrayDescriptor* sizeDesc = [MPSNDArrayDescriptor descriptorWithDataType:MPSDataTypeInt32 shape:@[ @(2) ]];
    MPSNDArray* sizeNDArray = [[[MPSNDArray alloc] initWithDevice:stream->device() descriptor:sizeDesc] autorelease];
    [sizeNDArray writeBytes:(int32_t[]){(int32_t)output_height, (int32_t)output_width} strideBytes:nil];
    MPSGraphTensorData* sizeTensorData = [[[MPSGraphTensorData alloc] initWithMPSNDArray:sizeNDArray] autorelease];

    Placeholder inputPlaceholder = Placeholder(cachedGraph->inputTensor, input);
    Placeholder outputPlaceholder =
        Placeholder(cachedGraph->outputTensor, out.has_storage() ? out : output, nil, false);

    NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* feeds = @{
      inputPlaceholder.getMPSGraphTensor() : inputPlaceholder.getMPSGraphTensorData(),
      cachedGraph->outputSizeTensor : sizeTensorData,
    };
    runMPSGraph(stream, cachedGraph->graph(), feeds, outputPlaceholder);

    if (out.has_storage()) {
      output.copy_(out);
    }
  }
}

#ifndef PYTORCH_JIT_COMPILE_SHADERS
static auto& lib = mps::MetalShaderLibrary::getBundledLibrary();
#else
#include <ATen/native/mps/Upsample_metallib.h>
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

static void upsample_bicubic2d_out_template(const Tensor& input,
                                            IntArrayRef output_size,
                                            bool align_corners,
                                            std::optional<double> scale_h_opt,
                                            std::optional<double> scale_w_opt,
                                            const Tensor& output) {
  if (output.numel() == 0) {
    return;
  }
  std::array<float, 2> scales = {
      area_pixel_compute_scale<float>(input.size(3), output.size(3), align_corners, scale_w_opt),
      area_pixel_compute_scale<float>(input.size(2), output.size(2), align_corners, scale_h_opt)};
  auto upsamplePSO = lib.getPipelineStateForFunc("upsample_bicubic2d_" + mps::scalarToMetalTypeString(input));
  auto stream = getCurrentMPSStream();
  dispatch_sync_with_rethrow(stream->queue(), ^() {
    @autoreleasepool {
      std::array<int64_t, 4> output_strides = {output.stride(3), output.stride(2), output.stride(1), output.stride(0)};
      std::array<int64_t, 4> output_sizes = {output.size(3), output.size(2), output.size(1), output.size(0)};
      std::array<int64_t, 4> input_sizes = {input.size(3), input.size(2), input.size(1), input.size(0)};
      std::array<int64_t, 4> input_strides = {input.stride(3), input.stride(2), input.stride(1), input.stride(0)};
      auto computeEncoder = stream->commandEncoder();
      [computeEncoder setComputePipelineState:upsamplePSO];
      mtl_setBuffer(computeEncoder, input, 0);
      mtl_setBuffer(computeEncoder, output, 1);
      mtl_setBytes(computeEncoder, input_strides, 2);
      mtl_setBytes(computeEncoder, output_strides, 3);
      mtl_setBytes(computeEncoder, input_sizes, 4);
      mtl_setBytes(computeEncoder, output_sizes, 5);
      mtl_setBytes(computeEncoder, scales, 6);
      mtl_setBytes(computeEncoder, align_corners, 7);
      mtl_dispatch1DJob(computeEncoder, upsamplePSO, output_size[0] * output_size[1]);
    }
  });
}

static void upsample_bicubic2d_backward_out_template(const Tensor& grad_input,
                                                     const Tensor& grad_output,
                                                     IntArrayRef output_size,
                                                     IntArrayRef input_size,
                                                     bool align_corners,
                                                     std::optional<double> scale_h_opt,
                                                     std::optional<double> scale_w_opt) {
  grad_input.zero_();
  if (grad_output.numel() == 0) {
    return;
  }
  std::array<float, 2> scales = {
      area_pixel_compute_scale<float>(grad_input.size(3), grad_output.size(3), align_corners, scale_w_opt),
      area_pixel_compute_scale<float>(grad_input.size(2), grad_output.size(2), align_corners, scale_h_opt)};
  auto upsamplePSO =
      lib.getPipelineStateForFunc("upsample_bicubic2d_backward_" + mps::scalarToMetalTypeString(grad_input));
  auto stream = getCurrentMPSStream();
  dispatch_sync_with_rethrow(stream->queue(), ^() {
    @autoreleasepool {
      std::array<int64_t, 4> output_strides = {
          grad_output.stride(3), grad_output.stride(2), grad_output.stride(1), grad_output.stride(0)};
      std::array<int64_t, 4> output_sizes = {
          grad_output.size(3), grad_output.size(2), grad_output.size(1), grad_output.size(0)};
      std::array<int64_t, 4> input_sizes = {
          grad_input.size(3), grad_input.size(2), grad_input.size(1), grad_input.size(0)};
      std::array<int64_t, 4> input_strides = {
          grad_input.stride(3), grad_input.stride(2), grad_input.stride(1), grad_input.stride(0)};
      auto computeEncoder = stream->commandEncoder();
      [computeEncoder setComputePipelineState:upsamplePSO];
      mtl_setBuffer(computeEncoder, grad_input, 0);
      mtl_setBuffer(computeEncoder, grad_output, 1);
      mtl_setBytes(computeEncoder, input_strides, 2);
      mtl_setBytes(computeEncoder, output_strides, 3);
      mtl_setBytes(computeEncoder, input_sizes, 4);
      mtl_setBytes(computeEncoder, output_sizes, 5);
      mtl_setBytes(computeEncoder, scales, 6);
      mtl_setBytes(computeEncoder, align_corners, 7);
      mtl_dispatch1DJob(computeEncoder, upsamplePSO, output_size[0] * output_size[1]);
    }
  });
}

} // namespace mps

TORCH_IMPL_FUNC(upsample_nearest1d_out_mps)
(const Tensor& input, IntArrayRef output_size, std::optional<double> scale, const Tensor& output) {
  mps::upsample_out_template(input, output_size, std::nullopt, std::nullopt, scale, output, false, "nearest");
}

TORCH_IMPL_FUNC(upsample_nearest1d_backward_out_mps)
(const Tensor& grad_output,
 IntArrayRef output_size,
 IntArrayRef input_size,
 std::optional<double> scale,
 const Tensor& grad_input) {
  mps::upsample_out_template(grad_output, output_size, input_size, std::nullopt, scale, grad_input, false, "nearest");
}

TORCH_IMPL_FUNC(_upsample_nearest_exact1d_out_mps)
(const Tensor& input, IntArrayRef output_size, std::optional<double> scale, const Tensor& output) {
  mps::upsample_out_template(input, output_size, std::nullopt, std::nullopt, scale, output, false, "nearest-exact");
}

TORCH_IMPL_FUNC(_upsample_nearest_exact1d_backward_out_mps)
(const Tensor& grad_output,
 IntArrayRef output_size,
 IntArrayRef input_size,
 std::optional<double> scale,
 const Tensor& grad_input) {
  mps::upsample_out_template(
      grad_output, output_size, input_size, std::nullopt, scale, grad_input, false, "nearest-exact");
}

TORCH_IMPL_FUNC(upsample_nearest2d_out_mps)
(const Tensor& input,
 IntArrayRef output_size,
 std::optional<double> scales_h,
 std::optional<double> scales_w,
 const Tensor& output) {
  mps::upsample_out_template(input, output_size, std::nullopt, scales_h, scales_w, output, false, "nearest");
}

TORCH_IMPL_FUNC(upsample_nearest2d_backward_out_mps)
(const Tensor& grad_output,
 IntArrayRef output_size,
 IntArrayRef input_size,
 std::optional<double> scales_h,
 std::optional<double> scales_w,
 const Tensor& grad_input) {
  mps::upsample_out_template(grad_output, output_size, input_size, scales_h, scales_w, grad_input, false, "nearest");
}

TORCH_IMPL_FUNC(_upsample_nearest_exact2d_out_mps)
(const Tensor& input,
 IntArrayRef output_size,
 std::optional<double> scales_h,
 std::optional<double> scales_w,
 const Tensor& output) {
  mps::upsample_out_template(input, output_size, std::nullopt, scales_h, scales_w, output, false, "nearest-exact");
}

TORCH_IMPL_FUNC(_upsample_nearest_exact2d_backward_out_mps)
(const Tensor& grad_output,
 IntArrayRef output_size,
 IntArrayRef input_size,
 std::optional<double> scales_h,
 std::optional<double> scales_w,
 const Tensor& grad_input) {
  mps::upsample_out_template(
      grad_output, output_size, input_size, scales_h, scales_w, grad_input, false, "nearest-exact");
}

TORCH_IMPL_FUNC(upsample_linear1d_out_mps)
(const Tensor& input, IntArrayRef output_size, bool align_corners, std::optional<double> scale, const Tensor& output) {
  mps::upsample_out_template(input, output_size, std::nullopt, std::nullopt, scale, output, align_corners, "bilinear");
}

TORCH_IMPL_FUNC(upsample_linear1d_backward_out_mps)
(const Tensor& grad_output,
 IntArrayRef output_size,
 IntArrayRef input_size,
 bool align_corners,
 std::optional<double> scale,
 const Tensor& grad_input) {
  mps::upsample_out_template(
      grad_output, output_size, input_size, std::nullopt, scale, grad_input, align_corners, "bilinear");
}

TORCH_IMPL_FUNC(upsample_bilinear2d_out_mps)
(const Tensor& input,
 IntArrayRef output_size,
 bool align_corners,
 std::optional<double> scales_h,
 std::optional<double> scales_w,
 const Tensor& output) {
  mps::upsample_out_template(input, output_size, std::nullopt, scales_h, scales_w, output, align_corners, "bilinear");
}

TORCH_IMPL_FUNC(upsample_bilinear2d_backward_out_mps)
(const Tensor& grad_output,
 IntArrayRef output_size,
 IntArrayRef input_size,
 bool align_corners,
 std::optional<double> scales_h,
 std::optional<double> scales_w,
 const Tensor& grad_input) {
  mps::upsample_out_template(
      grad_output, output_size, input_size, scales_h, scales_w, grad_input, align_corners, "bilinear");
}

TORCH_IMPL_FUNC(upsample_bicubic2d_out_mps)
(const Tensor& input,
 IntArrayRef output_size,
 bool align_corners,
 std::optional<double> scales_h,
 std::optional<double> scales_w,
 const Tensor& output) {
  mps::upsample_bicubic2d_out_template(input, output_size, align_corners, scales_h, scales_w, output);
}

TORCH_IMPL_FUNC(upsample_bicubic2d_backward_out_mps)
(const Tensor& grad_output,
 IntArrayRef output_size,
 IntArrayRef input_size,
 bool align_corners,
 std::optional<double> scales_h,
 std::optional<double> scales_w,
 const Tensor& grad_input) {
  mps::upsample_bicubic2d_backward_out_template(
      grad_input, grad_output, output_size, input_size, align_corners, scales_h, scales_w);
}

} // namespace at::native
