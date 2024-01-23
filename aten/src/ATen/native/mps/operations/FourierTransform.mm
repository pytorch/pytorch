//  Copyright © 2023 Apple Inc.
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/native/UpSample.h>
#include <ATen/native/mps/MPSGraphVenturaOps.h>
#include <ATen/native/mps/MPSGraphSonomaOps.h>
#include <ATen/native/mps/OperationUtils.h>

#include <MetalPerformanceShadersGraph/MPSGraph.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_fft_c2c.h>
#include <ATen/ops/_fft_c2r.h>
#include <ATen/ops/_fft_r2c.h>
#endif

#if !defined(__MAC_14_0) && \
    (!defined(MAC_OS_X_VERSION_14_0) || (MAC_OS_X_VERSION_MIN_REQUIRED < MAC_OS_X_VERSION_14_0))
@implementation FakeMPSGraphFFTDescriptor
+(nullable instancetype) descriptor {
  // This should never be called
  return nil;
}
@end
#endif
    
namespace at::native {
namespace mps {

enum class FFTType { R2C, C2R, C2C };

// Prototypes
Tensor _fft_r2c_mps(const Tensor& input, int64_t signal_ndim, bool normalized, IntArrayRef signal_sizes);
Tensor& _fft_r2c_mps_out(const Tensor& input,
                         Tensor& out,
                         int64_t signal_ndim,
                         bool normalized,
                         IntArrayRef signal_sizes);
Tensor _fft_c2r_mps(const Tensor& input, int64_t signal_ndim, bool normalized, IntArrayRef signal_sizes);
Tensor& _fft_c2r_mps_out(const Tensor& input,
                         Tensor& out,
                         int64_t signal_ndim,
                         bool normalized,
                         IntArrayRef signal_sizes);
Tensor _fft_c2c_mps(const Tensor& input, int64_t signal_ndim, bool normalized, bool forward, IntArrayRef signal_sizes);
Tensor& _fft_c2c_mps_out(const Tensor& input,
                         Tensor& out,
                         int64_t signal_ndim,
                         bool normalized,
                         bool forward,
                         IntArrayRef signal_sizes);
static Tensor runFFTGraph(const Tensor& input,
                          const Tensor& out,
                          bool out_provided,
                          const std::string& key_prefix,
                          mps::FFTType fftType,
                          int64_t signal_ndim,
                          bool normalized,
                          bool forward);

static MPSGraphTensor* createFFTGraph(MPSGraph* graph,
                                      MPSGraphTensor* inputTensor,
                                      int64_t signal_ndim,
                                      bool normalized,
                                      mps::FFTType fftType,
                                      bool forward) {
  MPSGraphFFTDescriptor* descriptor = [MPSGraphFFTDescriptor descriptor];
  descriptor.inverse = (fftType == mps::FFTType::C2C) ? !forward : (fftType == mps::FFTType::C2R);
  descriptor.scalingMode = normalized ? MPSGraphFFTScalingModeUnitary : MPSGraphFFTScalingModeNone;

  switch (fftType) {
    case mps::FFTType::R2C:
      return [graph realToHermiteanFFTWithTensor:inputTensor axes:@[ @(signal_ndim) ] descriptor:descriptor name:nil];
    case mps::FFTType::C2R:
      return [graph HermiteanToRealFFTWithTensor:inputTensor axes:@[ @(signal_ndim) ] descriptor:descriptor name:nil];
    case mps::FFTType::C2C:
      return [graph fastFourierTransformWithTensor:inputTensor axes:@[ @(signal_ndim) ] descriptor:descriptor name:nil];
  }
}

static Tensor runFFTGraph(const Tensor& input,
                          const Tensor& output,
                          bool out_provided,
                          const std::string& key_prefix,
                          mps::FFTType fftType,
                          int64_t signal_ndim,
                          bool normalized,
                          bool forward) {
  TORCH_CHECK(input.is_mps(), key_prefix + ": Expected MPS tensor");
  if (out_provided) {
    TORCH_CHECK(output.is_mps(), key_prefix + ": Expected MPS tensor for output");
  }

  Tensor out;
  if (!output.is_contiguous()) {
    out = at::empty_like(output, MemoryFormat::Contiguous);
  }

  struct CachedGraph : public MPSCachedGraph {
    CachedGraph(MPSGraph* graph) : MPSCachedGraph(graph) {}
    MPSGraphTensor *inputTensor = nil, *outputTensor = nil;
    MPSGraphTensor* outputSizeTensor = nil;
  };

  MPSStream* stream = getCurrentMPSStream();

  @autoreleasepool {
    string key = key_prefix + getTensorsStringKey({input}) + getMPSTypeString(input);

    auto cachedGraph = LookUpOrCreateCachedGraph<CachedGraph>(key, [&](auto mpsGraph, auto newCachedGraph) {
      MPSGraphTensor* inputTensor = mpsGraphRankedPlaceHolder(mpsGraph, getMPSDataType(input), getMPSShape(input));
      MPSGraphTensor* outputTensor = createFFTGraph(mpsGraph, inputTensor, signal_ndim, normalized, fftType, forward);
      newCachedGraph->inputTensor = inputTensor;
      newCachedGraph->outputTensor = outputTensor;
    });

    // Use utility functions to determine the shape of the output tensor
    auto output_shape = getMPSShape(input);
    int64_t output_height = output_shape[0].integerValue;
    int64_t output_width = output_shape[1].integerValue;

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

static bool check_mps_compatibility() {
  static const bool is_macOS_14_0_or_newer = is_macos_13_or_newer(MacOSVersion::MACOS_VER_14_0_PLUS);
  if (!is_macOS_14_0_or_newer) {
    TORCH_WARN_ONCE("MPS: FFT operations are only supported natively starting from macOS 14.0. ",
                    "Falling back on CPU. This may have performance implications.");
    return false;
  }
  return true;
}

// Real-to-Hermitean FFT
Tensor _fft_r2c_mps(const Tensor& input, int64_t signal_ndim, bool normalized, IntArrayRef signal_sizes) {
  return runFFTGraph(input, {}, false, "fft_r2c_mps", mps::FFTType::R2C, signal_ndim, normalized, true);
}

// Real-to-Hermitean FFT, writing result to the provided output tensor
Tensor& _fft_r2c_mps_out(const Tensor& input,
                         Tensor& out,
                         int64_t signal_ndim,
                         bool normalized,
                         IntArrayRef signal_sizes) {
  runFFTGraph(input, out, true, "fft_r2c_out_mps", mps::FFTType::R2C, signal_ndim, normalized, true);
  return out;
}

// Hermitean-to-Real FFT
Tensor _fft_c2r_mps(const Tensor& input, int64_t signal_ndim, bool normalized, IntArrayRef signal_sizes) {
  return runFFTGraph(input, {}, false, "fft_c2r_mps", mps::FFTType::C2R, signal_ndim, normalized, false);
}

// Hermitean-to-Real FFT, writing result to the provided output tensor
Tensor& _fft_c2r_mps_out(const Tensor& input,
                         Tensor& out,
                         int64_t signal_ndim,
                         bool normalized,
                         IntArrayRef signal_sizes) {
  runFFTGraph(input, out, true, "fft_c2r_out_mps", mps::FFTType::C2R, signal_ndim, normalized, false);
  return out;
}

// complex-to-complex FFT
Tensor _fft_c2c_mps(const Tensor& input, int64_t signal_ndim, bool normalized, bool forward, IntArrayRef signal_sizes) {
  return runFFTGraph(input, {}, false, "fft_c2c_mps", mps::FFTType::C2C, signal_ndim, normalized, forward);
}

// complex-to-complex FFT, writing result to the provided output tensor
Tensor& _fft_c2c_mps_out(const Tensor& input,
                         Tensor& out,
                         int64_t signal_ndim,
                         bool normalized,
                         bool forward,
                         IntArrayRef signal_sizes) {
  runFFTGraph(input, out, true, "fft_c2c_out_mps", mps::FFTType::C2C, signal_ndim, normalized, forward);
  return out;
}

} // namespace at::native
