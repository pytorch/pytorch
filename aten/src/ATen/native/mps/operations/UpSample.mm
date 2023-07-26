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
#include <ATen/ops/upsample_bilinear2d.h>
#include <ATen/ops/upsample_bilinear2d_backward.h>
#include <ATen/ops/upsample_bilinear2d_backward_native.h>
#include <ATen/ops/upsample_bilinear2d_native.h>
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
void upsample_out_template(const Tensor& input,
                           IntArrayRef output_size,
                           c10::optional<IntArrayRef> input_size_opt, // only used for backward pass
                           c10::optional<double> scale_h_opt,
                           c10::optional<double> scale_w_opt,
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
  if (!output.is_contiguous()) {
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
    AT_ERROR("Unsupported resize mode ", resize_mode_str);
  }

  const bool is_macOS_13_0_or_newer = is_macos_13_or_newer();
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
        getTensorsStringKey({input}) + ":[" + to_string(scale_h) + "," + to_string(scale_w) + "]:[" +
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
      if (is_macOS_13_0_or_newer) {
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
      } else { // if macOS version < 13.0 (for backwards compatibility)
        if (!is_backward_pass) {
          newCachedGraph->outputTensor = [mpsGraph resizeTensor:newCachedGraph->inputTensor
                                                     sizeTensor:newCachedGraph->outputSizeTensor
                                                           mode:resizeMode
                                                   centerResult:centerResults
                                                   alignCorners:align_corners
                                                         layout:dataLayout
                                                           name:nil];
        } else {
          newCachedGraph->outputTensor = [mpsGraph resizeWithGradientTensor:newCachedGraph->inputTensor
                                                                      input:inputSizeTensor
                                                                       mode:resizeMode
                                                               centerResult:centerResults
                                                               alignCorners:align_corners
                                                                     layout:dataLayout
                                                                       name:nil];
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
    NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* results =
        @{outputPlaceholder.getMPSGraphTensor() : outputPlaceholder.getMPSGraphTensorData()};
    runMPSGraph(stream, cachedGraph->graph(), feeds, results);

    if (out.has_storage()) {
      output.copy_(out);
    }
  }
}

} // namespace mps

static bool check_mps_compatibility(const c10::string_view resize_mode_str, c10::optional<double> scale) {
  static const bool is_macOS_13_0_or_newer = is_macos_13_or_newer();
  if (!is_macOS_13_0_or_newer) {
    // passing scale factors to MPS's resize APIs is not supported on macOS < 13
    if (scale.value_or(0.) > 0.) {
      TORCH_WARN_ONCE("MPS: passing scale factor to upsample ops is supported natively starting from macOS 13.0. ",
                      "Falling back on CPU. This may have performance implications.");
      return false;
      // nearest mode on Monterey uses round() to compute source indices which
      // is incompatible with PyTorch that uses floor(). So we fallback to CPU on Monterey.
      // The nearest mode should work fine on Ventura.
    } else if (resize_mode_str == "nearest" || resize_mode_str == "nearest-exact") {
      TORCH_WARN_ONCE("MPS: '",
                      resize_mode_str,
                      "' mode upsampling is supported natively starting from macOS 13.0. ",
                      "Falling back on CPU. This may have performance implications.");
      return false;
    }
  }
  return true;
}

TORCH_IMPL_FUNC(upsample_nearest1d_out_mps)
(const Tensor& input, IntArrayRef output_size, c10::optional<double> scale, const Tensor& output) {
  if (check_mps_compatibility("nearest", scale)) {
    mps::upsample_out_template(input, output_size, c10::nullopt, c10::nullopt, scale, output, false, "nearest");
  } else {
    output.copy_(at::upsample_nearest1d(input.to("cpu"), output_size, scale));
  }
}

TORCH_IMPL_FUNC(upsample_nearest1d_backward_out_mps)
(const Tensor& grad_output,
 IntArrayRef output_size,
 IntArrayRef input_size,
 c10::optional<double> scale,
 const Tensor& grad_input) {
  if (check_mps_compatibility("nearest", scale)) {
    mps::upsample_out_template(grad_output, output_size, input_size, c10::nullopt, scale, grad_input, false, "nearest");
  } else {
    grad_input.copy_(at::upsample_nearest1d_backward(grad_output.to("cpu"), output_size, input_size, scale));
  }
}

TORCH_IMPL_FUNC(_upsample_nearest_exact1d_out_mps)
(const Tensor& input, IntArrayRef output_size, c10::optional<double> scale, const Tensor& output) {
  if (check_mps_compatibility("nearest-exact", scale)) {
    mps::upsample_out_template(input, output_size, c10::nullopt, c10::nullopt, scale, output, false, "nearest-exact");
  } else {
    output.copy_(at::_upsample_nearest_exact1d(input.to("cpu"), output_size, scale));
  }
}

TORCH_IMPL_FUNC(_upsample_nearest_exact1d_backward_out_mps)
(const Tensor& grad_output,
 IntArrayRef output_size,
 IntArrayRef input_size,
 c10::optional<double> scale,
 const Tensor& grad_input) {
  if (check_mps_compatibility("nearest-exact", scale)) {
    mps::upsample_out_template(
        grad_output, output_size, input_size, c10::nullopt, scale, grad_input, false, "nearest-exact");
  } else {
    grad_input.copy_(at::_upsample_nearest_exact1d_backward(grad_output.to("cpu"), output_size, input_size, scale));
  }
}

TORCH_IMPL_FUNC(upsample_nearest2d_out_mps)
(const Tensor& input,
 IntArrayRef output_size,
 c10::optional<double> scales_h,
 c10::optional<double> scales_w,
 const Tensor& output) {
  if (check_mps_compatibility("nearest", scales_w)) {
    mps::upsample_out_template(input, output_size, c10::nullopt, scales_h, scales_w, output, false, "nearest");
  } else {
    output.copy_(at::upsample_nearest2d(input.to("cpu"), output_size, scales_h, scales_w));
  }
}

TORCH_IMPL_FUNC(upsample_nearest2d_backward_out_mps)
(const Tensor& grad_output,
 IntArrayRef output_size,
 IntArrayRef input_size,
 c10::optional<double> scales_h,
 c10::optional<double> scales_w,
 const Tensor& grad_input) {
  if (check_mps_compatibility("nearest", scales_w)) {
    mps::upsample_out_template(grad_output, output_size, input_size, scales_h, scales_w, grad_input, false, "nearest");
  } else {
    grad_input.copy_(
        at::upsample_nearest2d_backward(grad_output.to("cpu"), output_size, input_size, scales_h, scales_w));
  }
}

TORCH_IMPL_FUNC(_upsample_nearest_exact2d_out_mps)
(const Tensor& input,
 IntArrayRef output_size,
 c10::optional<double> scales_h,
 c10::optional<double> scales_w,
 const Tensor& output) {
  if (check_mps_compatibility("nearest-exact", scales_w)) {
    mps::upsample_out_template(input, output_size, c10::nullopt, scales_h, scales_w, output, false, "nearest-exact");
  } else {
    output.copy_(at::_upsample_nearest_exact2d(input.to("cpu"), output_size, scales_h, scales_w));
  }
}

TORCH_IMPL_FUNC(_upsample_nearest_exact2d_backward_out_mps)
(const Tensor& grad_output,
 IntArrayRef output_size,
 IntArrayRef input_size,
 c10::optional<double> scales_h,
 c10::optional<double> scales_w,
 const Tensor& grad_input) {
  if (check_mps_compatibility("nearest-exact", scales_w)) {
    mps::upsample_out_template(
        grad_output, output_size, input_size, scales_h, scales_w, grad_input, false, "nearest-exact");
  } else {
    grad_input.copy_(
        at::_upsample_nearest_exact2d_backward(grad_output.to("cpu"), output_size, input_size, scales_h, scales_w));
  }
}

TORCH_IMPL_FUNC(upsample_bilinear2d_out_mps)
(const Tensor& input,
 IntArrayRef output_size,
 bool align_corners,
 c10::optional<double> scales_h,
 c10::optional<double> scales_w,
 const Tensor& output) {
  if (check_mps_compatibility("bilinear", scales_w)) {
    mps::upsample_out_template(input, output_size, c10::nullopt, scales_h, scales_w, output, align_corners, "bilinear");
  } else {
    output.copy_(at::upsample_bilinear2d(input.to("cpu"), output_size, align_corners, scales_h, scales_w));
  }
}

TORCH_IMPL_FUNC(upsample_bilinear2d_backward_out_mps)
(const Tensor& grad_output,
 IntArrayRef output_size,
 IntArrayRef input_size,
 bool align_corners,
 c10::optional<double> scales_h,
 c10::optional<double> scales_w,
 const Tensor& grad_input) {
  if (check_mps_compatibility("bilinear", scales_w)) {
    mps::upsample_out_template(
        grad_output, output_size, input_size, scales_h, scales_w, grad_input, align_corners, "bilinear");
  } else {
    grad_input.copy_(at::upsample_bilinear2d_backward(
        grad_output.to("cpu"), output_size, input_size, align_corners, scales_h, scales_w));
  }
}

} // namespace at::native
