#include <ATen/native/SpectralOpsUtils.h>
#include <ATen/native/mps/MPSGraphSonomaOps.h>
#include <ATen/native/mps/MPSGraphVenturaOps.h>
#include <ATen/native/mps/OperationUtils.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_fft_c2c_native.h>
#include <ATen/ops/_fft_c2r_native.h>
#include <ATen/ops/_fft_r2c_native.h>
#endif

#if !defined(__MAC_14_0) && (!defined(MAC_OS_X_VERSION_14_0) || (MAC_OS_X_VERSION_MIN_REQUIRED < MAC_OS_X_VERSION_14_0))
@implementation FakeMPSGraphFFTDescriptor
+ (nullable instancetype)descriptor {
  // Redispatch the constructor to the actual implementation
  id desc = NSClassFromString(@"MPSGraphFFTDescriptor");
  return (FakeMPSGraphFFTDescriptor*)[desc descriptor];
}

- (nonnull id)copyWithZone:(nullable NSZone*)zone {
  return self;
}
@end
#endif

namespace at::native {
namespace {
MPSGraphFFTScalingMode normalization_to_ScalingMode(int64_t normalization) {
  switch (static_cast<fft_norm_mode>(normalization)) {
    case fft_norm_mode::none:
      return MPSGraphFFTScalingModeNone;
    case fft_norm_mode::by_n:
      return MPSGraphFFTScalingModeSize;
    case fft_norm_mode::by_root_n:
      return MPSGraphFFTScalingModeUnitary;
    default:
      break;
  }
  TORCH_CHECK(false, "Unsupported normalization type", normalization);
}

NSArray<NSNumber*>* IntArrayToNSArray(IntArrayRef arr) {
  auto rc = [NSMutableArray<NSNumber*> arrayWithCapacity:arr.size()];
  for (const auto idx : c10::irange(arr.size())) {
    rc[idx] = [NSNumber numberWithInteger:arr[idx]];
  }
  return rc;
}

} // anonymous namespace

Tensor _fft_c2r_mps(const Tensor& self, IntArrayRef dim, int64_t normalization, int64_t last_dim_size) {
  TORCH_CHECK(self.is_complex());
  auto in_sizes = self.sizes();
  DimVector out_sizes(in_sizes.begin(), in_sizes.end());
  out_sizes[dim.back()] = last_dim_size;
  auto out = at::empty(out_sizes, self.options().dtype(c10::toRealValueType(self.scalar_type())));
  return _fft_c2r_mps_out(self, dim, normalization, last_dim_size, out);
}

Tensor _fft_r2c_mps(const Tensor& self, IntArrayRef dim, int64_t normalization, bool onesided) {
  TORCH_CHECK(self.is_floating_point());
  auto input_sizes = self.sizes();
  DimVector out_sizes(input_sizes.begin(), input_sizes.end());
  auto last_dim = dim.back();
  auto last_dim_halfsize = (input_sizes[last_dim]) / 2 + 1;
  if (onesided) {
    out_sizes[last_dim] = last_dim_halfsize;
  }

  auto out = at::empty(out_sizes, self.options().dtype(c10::toComplexType(self.scalar_type())));
  return _fft_r2c_mps_out(self, dim, normalization, onesided, out);
}

Tensor _fft_c2c_mps(const Tensor& self, IntArrayRef dim, int64_t normalization, bool forward) {
  TORCH_CHECK(self.is_complex());
  if (dim.empty()) {
    return self.clone();
  }
  auto out = at::empty(self.sizes(), self.options());
  return _fft_c2c_mps_out(self, dim, normalization, forward, out);
}

using namespace mps;

// TODO: Investigate numerical discrepancies see https://github.com/pytorch/pytorch/issues/120237
Tensor& _fft_r2c_mps_out(const Tensor& self, IntArrayRef dim, int64_t normalization, bool onesided, Tensor& out) {
  TORCH_CHECK(supportsComplex(), "FFT operations are only supported on MacOS 14+");
  auto key = __func__ + getTensorsStringKey({self, out}) + ":" + getArrayRefString(dim) + ":" +
      std::to_string(normalization) + ":" + std::to_string(onesided);
  @autoreleasepool {
    auto cachedGraph = LookUpOrCreateCachedGraph<MPSUnaryCachedGraph>(key, [&](auto mpsGraph, auto newCachedGraph) {
      auto inputTensor = mpsGraphRankedPlaceHolder(mpsGraph, self);
      auto descriptor = [MPSGraphFFTDescriptor descriptor];
      descriptor.scalingMode = normalization_to_ScalingMode(normalization);
      MPSGraphTensor* outputTensor;
      if (onesided) {
        // Return only unique results:
        outputTensor = [mpsGraph realToHermiteanFFTWithTensor:inputTensor
                                                         axes:IntArrayToNSArray(dim)
                                                   descriptor:descriptor
                                                         name:nil];
      } else {
        // Return with Hermitean conjugate results:
        auto useDataType =
            (inputTensor.dataType == MPSDataTypeFloat16) ? MPSDataTypeComplexFloat16 : MPSDataTypeComplexFloat32;
        auto cTensor = [mpsGraph castTensor:inputTensor toType:useDataType name:nil];
        outputTensor = [mpsGraph fastFourierTransformWithTensor:cTensor
                                                           axes:IntArrayToNSArray(dim)
                                                     descriptor:descriptor
                                                           name:nil];
      }
      newCachedGraph->inputTensor_ = inputTensor;
      newCachedGraph->outputTensor_ = outputTensor;
    });
    auto inputPlaceholder = Placeholder(cachedGraph->inputTensor_, self);
    auto outputPlaceholder = Placeholder(cachedGraph->outputTensor_, out);
    auto feeds = dictionaryFromPlaceholders(inputPlaceholder);
    runMPSGraph(getCurrentMPSStream(), cachedGraph->graph(), feeds, outputPlaceholder);
  }
  return out;
}

Tensor& _fft_c2r_mps_out(const Tensor& self,
                         IntArrayRef dim,
                         int64_t normalization,
                         int64_t last_dim_size,
                         Tensor& out) {
  TORCH_CHECK(supportsComplex(), "FFT operations are only supported on MacOS 14+");
  auto key = __func__ + getTensorsStringKey({self}) + ":" + getArrayRefString(dim) + ":" +
      std::to_string(normalization) + ":" + std::to_string(last_dim_size);
  @autoreleasepool {
    auto cachedGraph = LookUpOrCreateCachedGraph<MPSUnaryCachedGraph>(key, [&](auto mpsGraph, auto newCachedGraph) {
      auto inputTensor = mpsGraphRankedPlaceHolder(mpsGraph, self);
      auto descriptor = [MPSGraphFFTDescriptor descriptor];
      descriptor.scalingMode = normalization_to_ScalingMode(normalization);
      descriptor.inverse = YES;
      descriptor.roundToOddHermitean = ((last_dim_size % 2) == 1) ? YES : NO;
      auto outputTensor = [mpsGraph HermiteanToRealFFTWithTensor:inputTensor
                                                            axes:IntArrayToNSArray(dim)
                                                      descriptor:descriptor
                                                            name:nil];
      newCachedGraph->inputTensor_ = inputTensor;
      newCachedGraph->outputTensor_ = outputTensor;
    });
    auto inputPlaceholder = Placeholder(cachedGraph->inputTensor_, self);
    auto outputPlaceholder = Placeholder(cachedGraph->outputTensor_, out);
    auto feeds = dictionaryFromPlaceholders(inputPlaceholder);
    runMPSGraph(getCurrentMPSStream(), cachedGraph->graph(), feeds, outputPlaceholder);
  }
  return out;
}

Tensor& _fft_c2c_mps_out(const Tensor& self, IntArrayRef dim, int64_t normalization, bool forward, Tensor& out) {
  TORCH_CHECK(supportsComplex(), "FFT operations are only supported on MacOS 14+");
  auto key = __func__ + getTensorsStringKey({self}) + ":" + getArrayRefString(dim) + ":" +
      std::to_string(normalization) + ":" + std::to_string(forward);
  @autoreleasepool {
    auto cachedGraph = LookUpOrCreateCachedGraph<MPSUnaryCachedGraph>(key, [&](auto mpsGraph, auto newCachedGraph) {
      auto inputTensor = mpsGraphRankedPlaceHolder(mpsGraph, self);
      auto descriptor = [MPSGraphFFTDescriptor descriptor];
      descriptor.scalingMode = normalization_to_ScalingMode(normalization);
      descriptor.inverse = !forward;
      auto outputTensor = [mpsGraph fastFourierTransformWithTensor:inputTensor
                                                              axes:IntArrayToNSArray(dim)
                                                        descriptor:descriptor
                                                              name:nil];
      newCachedGraph->inputTensor_ = inputTensor;
      newCachedGraph->outputTensor_ = outputTensor;
    });
    auto inputPlaceholder = Placeholder(cachedGraph->inputTensor_, self);
    auto outputPlaceholder = Placeholder(cachedGraph->outputTensor_, out);
    auto feeds = dictionaryFromPlaceholders(inputPlaceholder);
    runMPSGraph(getCurrentMPSStream(), cachedGraph->graph(), feeds, outputPlaceholder);
  }
  return out;
}

} // namespace at::native
