#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/WrapDimUtilsMulti.h>
#include <ATen/native/Resize.h>
#include <ATen/native/SpectralOpsUtils.h>
#include <ATen/native/mps/OperationUtils.h>
#include <algorithm>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_fft_c2c_native.h>
#include <ATen/ops/_fft_c2r_native.h>
#include <ATen/ops/_fft_r2c_native.h>
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

// MPSGraph FFT can only transform axes within the last four dimensions, so when
// an axis falls outside that window we permute it into the trailing dims, transform, then invert.
struct FFTAxisPlan {
  bool needsTranspose = false;
  NSArray<NSNumber*>* axes = nil;
  NSArray<NSNumber*>* permutation = nil;
  NSArray<NSNumber*>* inversePermutation = nil;
};

FFTAxisPlan computeFFTAxisPlan(IntArrayRef dim, int64_t ndim) {
  TORCH_CHECK(static_cast<int64_t>(dim.size()) <= 4,
              "MPS FFT only supports transforming up to 4 dimensions, but got ",
              dim.size(),
              " dimensions");
  FFTAxisPlan plan;
  if (std::all_of(dim.begin(), dim.end(), [&](int64_t d) { return d >= ndim - 4; })) {
    plan.axes = IntArrayToNSArray(dim);
    return plan;
  }

  plan.needsTranspose = true;
  const auto isTransformDim = at::dim_list_to_bitset(dim, ndim);
  std::vector<int64_t> perm;
  perm.reserve(ndim);
  for (const auto i : c10::irange(ndim)) {
    if (!isTransformDim[i]) {
      perm.push_back(i);
    }
  }
  perm.insert(perm.end(), dim.begin(), dim.end());
  std::vector<int64_t> inverse(ndim);
  for (const auto i : c10::irange(ndim)) {
    inverse[perm[i]] = i;
  }
  std::vector<int64_t> remappedAxes;
  remappedAxes.reserve(dim.size());
  for (const auto i : c10::irange(ndim - static_cast<int64_t>(dim.size()), ndim)) {
    remappedAxes.push_back(i);
  }
  plan.axes = IntArrayToNSArray(remappedAxes);
  plan.permutation = IntArrayToNSArray(perm);
  plan.inversePermutation = IntArrayToNSArray(inverse);
  return plan;
}

MPSGraphTensor* applyFFTInputPlan(MPSGraph* mpsGraph, MPSGraphTensor* input, const FFTAxisPlan& plan) {
  if (!plan.needsTranspose) {
    return input;
  }
  return [mpsGraph transposeTensor:input permutation:plan.permutation name:nil];
}

MPSGraphTensor* applyFFTOutputPlan(MPSGraph* mpsGraph, MPSGraphTensor* output, const FFTAxisPlan& plan) {
  if (!plan.needsTranspose) {
    return output;
  }
  return [mpsGraph transposeTensor:output permutation:plan.inversePermutation name:nil];
}

} // anonymous namespace

Tensor _fft_c2r_mps(const Tensor& self, IntArrayRef dim, int64_t normalization, int64_t last_dim_size) {
  auto out = at::empty({0}, self.options().dtype(c10::toRealValueType(self.scalar_type())));
  return _fft_c2r_mps_out(self, dim, normalization, last_dim_size, out);
}

Tensor _fft_r2c_mps(const Tensor& self, IntArrayRef dim, int64_t normalization, bool onesided) {
  auto out = at::empty({0}, self.options().dtype(c10::toComplexType(self.scalar_type())));
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
  TORCH_CHECK(self.scalar_type() == kFloat || self.scalar_type() == kHalf, "Only float and half dtypes are supported");
  TORCH_CHECK(out.scalar_type() == c10::toComplexType(self.scalar_type()));
  const auto input_sizes = self.sym_sizes();
  SymDimVector out_sizes(input_sizes.begin(), input_sizes.end());
  auto last_dim = dim.back();
  auto last_dim_halfsize = (input_sizes[last_dim]) / 2 + 1;
  if (onesided) {
    out_sizes[last_dim] = last_dim_halfsize;
  }
  at::native::resize_output_symint(out, out_sizes);

  auto key = __func__ + getTensorsStringKey({self, out}) + ":" + getArrayRefString(dim) + ":" +
      std::to_string(normalization) + ":" + std::to_string(onesided);
  @autoreleasepool {
    auto cachedGraph = LookUpOrCreateCachedGraph<MPSUnaryCachedGraph>(key, [&](auto mpsGraph, auto newCachedGraph) {
      auto inputTensor = mpsGraphRankedPlaceHolder(mpsGraph, self);
      auto axisPlan = computeFFTAxisPlan(dim, self.dim());
      auto fftInput = applyFFTInputPlan(mpsGraph, inputTensor, axisPlan);
      auto descriptor = [MPSGraphFFTDescriptor descriptor];
      descriptor.scalingMode = normalization_to_ScalingMode(normalization);
      MPSGraphTensor* outputTensor;
      if (onesided) {
        // Return only unique results:
        outputTensor = [mpsGraph realToHermiteanFFTWithTensor:fftInput
                                                         axes:axisPlan.axes
                                                   descriptor:descriptor
                                                         name:nil];
      } else {
        // Return with Hermitean conjugate results:
        auto useDataType =
            (fftInput.dataType == MPSDataTypeFloat16) ? MPSDataTypeComplexFloat16 : MPSDataTypeComplexFloat32;
        auto cTensor = [mpsGraph castTensor:fftInput toType:useDataType name:nil];
        outputTensor = [mpsGraph fastFourierTransformWithTensor:cTensor
                                                           axes:axisPlan.axes
                                                     descriptor:descriptor
                                                           name:nil];
      }
      outputTensor = applyFFTOutputPlan(mpsGraph, outputTensor, axisPlan);
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
  TORCH_CHECK(self.is_complex(), "Input must be complex");
  TORCH_CHECK(out.scalar_type() == c10::toRealValueType(self.scalar_type()), "Unexpected output type");
  const auto in_sizes = self.sym_sizes();
  SymDimVector out_sizes(in_sizes.begin(), in_sizes.end());
  out_sizes[dim.back()] = last_dim_size;
  at::native::resize_output_symint(out, out_sizes);
  auto key = __func__ + getTensorsStringKey({self}) + ":" + getArrayRefString(dim) + ":" +
      std::to_string(normalization) + ":" + std::to_string(last_dim_size);
  @autoreleasepool {
    auto cachedGraph = LookUpOrCreateCachedGraph<MPSUnaryCachedGraph>(key, [&](auto mpsGraph, auto newCachedGraph) {
      auto inputTensor = mpsGraphRankedPlaceHolder(mpsGraph, self);
      auto axisPlan = computeFFTAxisPlan(dim, self.dim());
      auto fftInput = applyFFTInputPlan(mpsGraph, inputTensor, axisPlan);
      auto descriptor = [MPSGraphFFTDescriptor descriptor];
      descriptor.scalingMode = normalization_to_ScalingMode(normalization);
      descriptor.inverse = YES;
      descriptor.roundToOddHermitean = ((last_dim_size % 2) == 1) ? YES : NO;
      auto outputTensor = [mpsGraph HermiteanToRealFFTWithTensor:fftInput
                                                            axes:axisPlan.axes
                                                      descriptor:descriptor
                                                            name:nil];
      outputTensor = applyFFTOutputPlan(mpsGraph, outputTensor, axisPlan);
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
  auto key = __func__ + getTensorsStringKey({self}) + ":" + getArrayRefString(dim) + ":" +
      std::to_string(normalization) + ":" + std::to_string(forward);
  @autoreleasepool {
    auto cachedGraph = LookUpOrCreateCachedGraph<MPSUnaryCachedGraph>(key, [&](auto mpsGraph, auto newCachedGraph) {
      auto inputTensor = mpsGraphRankedPlaceHolder(mpsGraph, self);
      auto axisPlan = computeFFTAxisPlan(dim, self.dim());
      auto fftInput = applyFFTInputPlan(mpsGraph, inputTensor, axisPlan);
      auto descriptor = [MPSGraphFFTDescriptor descriptor];
      descriptor.scalingMode = normalization_to_ScalingMode(normalization);
      descriptor.inverse = !forward;
      auto outputTensor = [mpsGraph fastFourierTransformWithTensor:fftInput
                                                              axes:axisPlan.axes
                                                        descriptor:descriptor
                                                              name:nil];
      outputTensor = applyFFTOutputPlan(mpsGraph, outputTensor, axisPlan);
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
